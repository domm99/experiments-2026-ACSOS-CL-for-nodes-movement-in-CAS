from typing import Callable, List, Optional, Union

import torch
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.models.bic_model import BiasLayer
from avalanche.training.plugins import BiCPlugin, EvaluationPlugin, SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.supervised.strategy_wrappers import CriterionType
from avalanche.training.templates import SupervisedTemplate
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader


class PatchedBiCPlugin(BiCPlugin):
    """Local Avalanche BiC patch for newer PyTorch MultiStepLR APIs."""

    def bias_correction_step(
        self,
        strategy: SupervisedTemplate,
        persistent_workers: bool = False,
        num_workers: int = 0,
    ):
        strategy.model.eval()

        targets = getattr(strategy.adapted_dataset, "targets")
        self.bias_layer = BiasLayer(targets.uniques)
        self.bias_layer.to(strategy.device)
        self.bias_layer.train()
        for param in self.bias_layer.parameters():
            param.requires_grad = True

        bic_optimizer = torch.optim.SGD(
            self.bias_layer.parameters(), lr=self.lr, momentum=0.9
        )
        scheduler = MultiStepLR(
            bic_optimizer,
            milestones=[50, 100, 150],
            gamma=0.1,
        )

        list_subsets: List[AvalancheDataset] = []
        for _, class_buf in self.val_buffer.items():
            list_subsets.append(class_buf.buffer)

        stage_set = concat_datasets(list_subsets)
        stage_loader = DataLoader(
            stage_set,
            batch_size=strategy.train_mb_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        for e in range(self.stage_2_epochs):
            total, t_acc, t_loss = 0, 0, 0
            for inputs in stage_loader:
                x = inputs[0].to(strategy.device)
                y_real = inputs[1].to(strategy.device)

                with torch.no_grad():
                    outputs = strategy.model(x)

                outputs = self.bias_layer(outputs)
                loss = torch.nn.functional.cross_entropy(outputs, y_real)

                _, preds = torch.max(outputs, 1)
                t_acc += torch.sum(preds == y_real.data)
                t_loss += loss.item() * x.size(0)
                total += x.size(0)

                loss += 0.1 * ((self.bias_layer.beta.sum() ** 2) / 2)

                bic_optimizer.zero_grad()
                loss.backward()
                bic_optimizer.step()

            scheduler.step()
            if self.verbose and (self.stage_2_epochs // 4) > 0:
                if (e + 1) % (self.stage_2_epochs // 4) == 0:
                    print(
                        "| E {:3d} | Train: loss={:.3f}, S2 acc={:5.1f}% |".format(
                            e + 1, t_loss / total, 100 * t_acc / total
                        )
                    )

        self.bias_layer.eval()
        for param in self.bias_layer.parameters():
            param.requires_grad = False

        if self.verbose:
            print(
                "Bias correction done: alpha={}, beta={}".format(
                    self.bias_layer.alpha.item(), self.bias_layer.beta.item()
                )
            )


class PatchedBiC(SupervisedTemplate):
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        mem_size: int = 200,
        val_percentage: float = 0.1,
        T: int = 2,
        stage_2_epochs: int = 200,
        lamb: float = -1,
        lr: float = 0.1,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every: int = -1,
        **base_kwargs,
    ):
        bic = PatchedBiCPlugin(
            mem_size=mem_size,
            val_percentage=val_percentage,
            T=T,
            stage_2_epochs=stage_2_epochs,
            lamb=lamb,
            lr=lr,
        )

        if plugins is None:
            plugins = [bic]
        else:
            plugins.append(bic)

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs,
        )
