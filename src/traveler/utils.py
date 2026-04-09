from typing import Iterable

import torch

from src.traveler.types import StateDict


def clone_state_dict(state_dict: StateDict) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _format_names(names: Iterable[str], limit: int = 10) -> str:
    names = list(names)
    if len(names) <= limit:
        return ", ".join(names)
    shown = ", ".join(names[:limit])
    return f"{shown}, ... (+{len(names) - limit} more)"


def _tensor_difference_summary(name: str, left: torch.Tensor, right: torch.Tensor) -> str:
    if left.shape != right.shape:
        return f"{name}: shape mismatch {tuple(left.shape)} != {tuple(right.shape)}"
    if left.dtype != right.dtype:
        return f"{name}: dtype mismatch {left.dtype} != {right.dtype}"

    if torch.equal(left, right):
        return f"{name}: identical"

    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()

    if left_cpu.is_floating_point() or left_cpu.is_complex():
        delta = (left_cpu - right_cpu).abs()
        max_abs_diff = delta.max().item()
        mean_abs_diff = delta.mean().item()
        diff_mask = delta != 0
    else:
        diff_mask = left_cpu != right_cpu
        max_abs_diff = None
        mean_abs_diff = None

    differing_entries = int(diff_mask.sum().item())
    total_entries = left_cpu.numel()
    first_flat_index = int(diff_mask.reshape(-1).nonzero()[0].item())
    first_index = tuple(int(i) for i in torch.unravel_index(torch.tensor(first_flat_index), left_cpu.shape))
    left_value = left_cpu[first_index].item()
    right_value = right_cpu[first_index].item()

    summary = (
        f"{name}: {differing_entries}/{total_entries} entries differ; "
        f"first difference at {first_index}: {left_value} != {right_value}"
    )
    if max_abs_diff is not None and mean_abs_diff is not None:
        summary += f"; max_abs_diff={max_abs_diff:.6g}; mean_abs_diff={mean_abs_diff:.6g}"
    return summary


def state_dicts_equal(
    left: StateDict,
    right: StateDict,
    *,
    left_name: str = "left",
    right_name: str = "right",
) -> tuple[bool, str | None]:
    left_keys = set(left.keys())
    right_keys = set(right.keys())

    missing_in_right = sorted(left_keys - right_keys)
    missing_in_left = sorted(right_keys - left_keys)
    if missing_in_right or missing_in_left:
        message_parts = [
            f"{left_name} and {right_name} do not have the same parameter keys."
        ]
        if missing_in_right:
            message_parts.append(
                f"Missing in {right_name}: {_format_names(missing_in_right)}"
            )
        if missing_in_left:
            message_parts.append(
                f"Missing in {left_name}: {_format_names(missing_in_left)}"
            )
        return False, " ".join(message_parts)

    difference_summaries: list[str] = []
    for name in sorted(left.keys()):
        if torch.equal(left[name], right[name]):
            continue
        difference_summaries.append(_tensor_difference_summary(name, left[name], right[name]))

    if not difference_summaries:
        return True, None

    limit = 8
    shown_summaries = difference_summaries[:limit]
    extra = len(difference_summaries) - len(shown_summaries)
    message = (
        f"{left_name} and {right_name} differ in {len(difference_summaries)} parameter tensors. "
        + " | ".join(shown_summaries)
    )
    if extra > 0:
        message += f" | ... (+{extra} more differing tensors)"
    return False, message


def traveler_log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[traveler] {message}", flush=True)
