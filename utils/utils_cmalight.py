import torch
from tqdm import tqdm

def closure(data_loader: torch.utils.data.DataLoader,
            device: torch.device,
            mod,
            loss_fun = torch.nn.CrossEntropyLoss()): #mettere in utils
    loss = 0
    P = len(data_loader)
    with torch.no_grad():
        with tqdm(data_loader, unit="step", position=0, leave=True) as tepoch:
            for batch in tepoch:
                tepoch.set_description("Validation")
                x, y = batch[0].to(device), batch[1].to(device)
                batch_loss = loss_fun(input=mod(x), target=y).item()
                loss += (len(x) / P) * batch_loss
    return loss

"""
def accuracy(model, dataloader: torch.utils.data.DataLoader, device: str): #mettere in utils
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    return accuracy
"""

