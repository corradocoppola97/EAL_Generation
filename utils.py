import os

import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from torchsummaryX import summary
import csv
import ast
import torchvision
from typing import Union
from cmalight import CMA_L
from utils_diffusion_model import T
from diffusionModel import forward_diffusion_sample
import torch.nn.functional as F
from tqdm import tqdm

#Used to compute the loss function over the entire data set
def closure(data_loader: torch.utils.data.DataLoader,
            model: torchvision.models,
            criterion: torch.nn,
            device: Union[torch.device,str]):
    
    model.eval()
    loss = 0
    with torch.no_grad():
        for x,y in data_loader:
            x,y = x.to(device), y.to(device)
            y_pred = model(x)
            batch_loss = criterion(y_pred, y)
            loss += batch_loss.item()*(len(x)/len(data_loader.dataset))
    return loss

#Used to compute the loss function for DDPM over the entire data set
def closure_diffusion(data_loader: torch.utils.data.DataLoader,
            model: torchvision.models,
            criterion: torch.nn,
            device: Union[torch.device,str]):
    
    model.eval()
    loss = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total = len(data_loader), desc='closure_loop', position = 1, leave=False):
            x = batch[0].to(device)
            batch_size = x.shape[0]

            t = torch.randint(0, T, (batch_size,), device=device).long()

            x_noisy, noise = forward_diffusion_sample(x, t, device)
            noise_pred = model(x_noisy, t)
            batch_loss = criterion(noise_pred, noise)
            loss += batch_loss.item()*(len(x)/len(data_loader.dataset))

    return loss

#Used to compute the accuracy over the entire data set
def accuracy(data_loader: torch.utils.data.DataLoader,
            model: torchvision.models,
            device: Union[torch.device,str]):
    
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    return accuracy

#Used to compute the accuracy for DDPM over the entire data set
def accuracy_diffusion(data_loader: torch.utils.data.DataLoader,
            model: torchvision.models,
            device: Union[torch.device,str]):
    
    model.eval()
    correct_predictions = 0
    total_samples = 0
    tolerance = 1e-1

    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total = len(data_loader), desc='accuracy_loop', position = 1, leave=False):
            x = batch[0].to(device)
            batch_size = x.shape[0]

            t = torch.randint(0, T, (batch_size,), device=device).long()

            x_noisy, noise = forward_diffusion_sample(x, t, device)
            noise_pred = model(x_noisy, t)

            noise = noise.view(-1)
            noise_pred = noise_pred.view(-1)

            correct_predictions += torch.sum(torch.abs(noise - noise_pred) < tolerance).item()
            total_samples += noise.shape[0]

    accuracy = correct_predictions / total_samples
    return accuracy

def get_w(model):
    weights = [p.ravel().detach() for p in model.parameters()]
    return torch.cat(weights)

def set_w(model, w):
    index = 0
    for param in model.parameters():
        param_size = torch.prod(torch.tensor(param.size())).item()
        param.data = w[index:index+param_size].view(param.size()).to(param.device)
        index += param_size


def set_optimizer(opt:str, model: torchvision.models, *args, **kwargs):
    if opt == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(),*args,**kwargs)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),*args,**kwargs)
    elif opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),*args,**kwargs)
    elif opt == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(),*args,**kwargs)
    elif opt == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(),*args,**kwargs)
    elif opt == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(),*args,**kwargs)
    elif opt == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(),*args,**kwargs)
    elif opt == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(),*args,**kwargs)
    elif opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(),*args,**kwargs)
    elif opt == 'rprop':
        optimizer = torch.optim.Rprop(model.parameters(),*args,**kwargs)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),*args,**kwargs)
    elif opt == 'cmal':
        optimizer = CMA_L(model.parameters(),momentum=0.9, nesterov=True)
    else:
        raise SystemError('RICORDATI DI SCEGLIERE L OTTIMIZZATORE!  :)')
    return optimizer

def set_scheduler(slr:str, optimizer: torch.optim.Optimizer, *args, **kwargs):
    if slr == 'stepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.1, *args, **kwargs)
    elif slr == 'MultiplicativeLR':
        lambda1 = lambda epoch: 0.95
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda1, *args, **kwargs)
    elif slr == 'MultiStepLR':
        milestones = [3, 6, 9]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1, *args, **kwargs)
    elif slr == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, *args, **kwargs)
    elif slr == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3 *args, **kwargs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.1, *args, **kwargs)
    return scheduler

def hardware_check():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Actual device: ", device)
    if 'cuda' in device:
        print("Device info: {}".format(str(torch.cuda.get_device_properties(device)).split("(")[1])[:-1])

    return device
    
def print_model(model, device, save_model_root, input_shape):
    info = summary(model, torch.ones((1, 3, 32, 32)).to(device))
    info.to_csv(save_model_root + 'model_summary.csv')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

def save_csv_history(path):
    objects = []
    with (open(path + 'history.pkl', "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    df = pd.DataFrame(objects)
    df.to_csv(path + 'history.csv', header=False, index=False, sep=" ")
    

def plot_graph(data, label, title, path):
    epochs = range(0, len(data))
    plt.plot(epochs, data, 'orange', label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid('on', color='#cfcfcf')
    plt.tight_layout()
    plt.savefig(path + title + '.pdf')
    plt.close()


def plot_history(history, path):
    plot_graph(history['train_loss'], 'Train Loss', 'Train_loss', path)
    plot_graph(history['val_acc'], 'Val Acc.', 'Val_acc', path)


def extract_history(history_file):
    with open(history_file) as file_obj:
        reader_obj = csv.reader(file_obj)
        for row in reader_obj:
            l = row[0]
            break
        l = l[:-9]
        train_loss = ast.literal_eval(l)
        val_accuracy = row[498:498+250]
        val_accuracy[-1] = val_accuracy[-1][:4]
        val_accuracy[0] = val_accuracy[0][-5:]
        val_accuracy = [float(c.strip('[ ]')) for c in val_accuracy]
    return train_loss, val_accuracy

# path = 'C:\\Users\\corra\\PycharmProjects\\CMA_light_code\\Results\\Results\\'
# datasets = ['CIFAR10','CIFAR100']
# networks = [os.listdir(path+datasets[0])[k].strip('-bs128-ep250') for k in range(len(os.listdir(path+datasets[0])))]
# t,v = extract_history('C:\\Users\\corra\\PycharmProjects\\CMA_light_code\\Results\\Results\\CIFAR10\\SwinT-bs128-ep250\\cmal\\history.csv')
# algos = ['adadelta','adagrad','adam','adamax','cmal']
# colors = ['red','blue','green','orange','cyan']
#


# os.chdir('Results')
net = 'swin_t'
dataset = 'cifar10'
nets = ['resnet18', 'resnet34', 'resnet50','resnet152','swin_t','swin_b','efficientnet_v2_l','mobilenet_v2']
algos = ['cmal','cmalNoRRSh']
colors = ['red','blue','green','orange','purple','cyan','brown']
def make_plot(dataset, net, epochs):
    plt.figure()
    for j in range(len(algos)):
        stats = torch.load('history_'+algos[j]+'_'+dataset+'_'+net+'_prova.txt')
        if epochs == True:
            plt.plot([_ for _ in range(1,len(stats['train_loss'])+1)], stats['train_loss'], color=colors[j])
        else:
            plt.plot(stats['elapsed_time_noVAL'], stats['train_loss'], color=colors[j])
    plt.legend(algos)
    if epochs == True: plt.xlabel('Epochs')
    else: plt.xlabel('Elapsed time (s)')
    plt.ylabel('Training Loss')
    plt.title(net + '  Training Loss')
    if epochs==True: plt.savefig('Loss_'+dataset+'_'+net+'_epochs.png')
    else: plt.savefig('Loss_'+dataset+'_'+net+'_time.png')


    plt.figure()
    for j in range(len(algos)):
        stats = torch.load('history_'+algos[j]+'_'+dataset+'_'+net+'_prova.txt')
        if epochs == True:
            plt.plot([_ for _ in range(1,len(stats['train_loss'])+1)], stats['val_acc'], color=colors[j])
        else:
            plt.plot(stats['elapsed_time_noVAL'], stats['val_acc'], color=colors[j])
    plt.legend(algos)
    if epochs == True: plt.xlabel('Epochs')
    else: plt.xlabel('Elapsed time (s)')
    plt.ylabel('Validation Accuracy')
    plt.title(net + '  Validation Accuracy')
    if epochs== True: plt.savefig('Accuracy_'+dataset+'_'+net+'_epochs.png')
    else: plt.savefig('Accuracy_'+dataset+'_'+net+'_time.png')


"""for net in nets:
    try:
        make_plot(dataset,net,epochs=True)
        make_plot(dataset, net, epochs=False)
    except:
        print(f'PASS with {net}')
    print('DONE')"""