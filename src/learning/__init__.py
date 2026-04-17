import copy
import torch
from torch import nn
from torch.nn import functional as F
from src.learning.models import CnnEMNIST
from torch.utils.data import DataLoader, Dataset

def initialize_model(dataset_name):
    if dataset_name == 'EMNIST':
        return CnnEMNIST()
    elif dataset_name == 'CIFAR100':
        raise Exception("CIFAR100 model not implemented yet")
    else:
        raise Exception("Unknown dataset")

def local_training(model, epochs, data, batch_size, device):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    epoch_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        batch_loss = []
        for batch_index, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        mean_epoch_loss = sum(batch_loss) / len(batch_loss)
        epoch_loss.append(mean_epoch_loss)
    return model.cpu().state_dict(), sum(epoch_loss) / len(epoch_loss)

def model_evaluation(model_params, data, batch_size, device, dataset_name):
    model = initialize_model(dataset_name)
    model.load_state_dict(model_params)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    accuracy = correct / total
    return accuracy, loss

def average_weights(models_params, weights):
    w_avg = copy.deepcopy(models_params[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], 0.0)
    sum_weights = sum(weights)
    for key in w_avg.keys():
        for i in range(0, len(models_params)):
            w_avg[key] += models_params[i][key] * weights[i]
        w_avg[key] = torch.div(w_avg[key], sum_weights)
    return w_avg


def local_distillation(
    student_weights,
    teacher_weights,
    data,
    batch_size,
    device,
    dataset_name,
    epochs=1,
    alpha=0.5,
    temperature=2.0,
    verbose=False,
):
    criterion = nn.CrossEntropyLoss()
    student_model = initialize_model(dataset_name)
    teacher_model = initialize_model(dataset_name)
    student_model.load_state_dict(student_weights)
    teacher_model.load_state_dict(teacher_weights)

    student_model.to(device)
    teacher_model.to(device)
    student_model.train()
    teacher_model.eval()
    epoch_loss = []

    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001, weight_decay=1e-4)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    for epoch_index in range(epochs):
        batch_loss = []
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.inference_mode():
                teacher_logits = teacher_model(images)

            student_logits = student_model(images)
            student_loss = criterion(student_logits, labels)
            distillation_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction='batchmean',
            ) * (temperature ** 2)
            loss = alpha * student_loss + (1 - alpha) * distillation_loss
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        mean_epoch_loss = sum(batch_loss) / len(batch_loss)
        epoch_loss.append(mean_epoch_loss)

    return student_model.state_dict(), sum(epoch_loss) / len(epoch_loss)
