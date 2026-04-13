import copy
import torch
from torch import nn
from src.learning.models import CnnEMNIST
from torch.utils.data import DataLoader

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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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
