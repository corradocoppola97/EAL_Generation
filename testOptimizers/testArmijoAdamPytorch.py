import time
import torch
from torch import nn
from torch.nn import Parameter
from linesearchTest import armijoMonotone, armijoMonotonePytorch, armijoMonotonePytorchNew
from adam import Adam
from torch.utils.data import Dataset
import numpy as np
import random

seed = 1234

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # Imposta la seed per tutti i dispositivi (se si utilizzano più GPU)

# Imposta la seed per il generatore casuale di Python
random.seed(seed)

# Imposta la seed per il generatore casuale di NumPy
np.random.seed(seed)

# Imposta alcuni parametri aggiuntivi per garantire la riproducibilità
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



x = torch.tensor([1,2,3,4], dtype=torch.float)
y = torch.tensor([11,12,13,14], dtype=torch.float)

# x = torch.tensor([1,2,3,4, 5, 6, 7, 8], dtype=torch.float)
# y = torch.tensor([11,12,13,14, 15, 16, 17, 18], dtype=torch.float)

dataset = (x, y)
device = 'cpu'
opt = 'sgd'

a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
# model = [Parameter(a), Parameter(b)]
model = nn.Linear(1,1)

lr = 0.1
curr_lr = lr
delta=1e-3
gamma=1e-3
criterion = nn.MSELoss()
if opt == 'adam':
    optimizer = Adam(model.parameters(), lr = lr)
elif opt == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# print(model[0].data.item())
# print(optimizer)
# print(len(optimizer.param_groups))




class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,2)
        self.fc2 = nn.Linear(2,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CustomImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset[0])
    
    def __getitem__(self, idx):
        return self.dataset[0][idx], self.dataset[1][idx]
    
trainset = CustomImageDataset(dataset)

dataloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=False)
elem = next(iter(dataloader))
print(len(elem[0]))
print(len(dataloader.dataset))


# model = SimpleLinear()
# print(list(model.parameters()))


# print(optimizer.param_groups[0])

def flatten_hats(m_hats):
    weights = [p.ravel().detach() for p in m_hats]
    return torch.cat(weights)

def get_w(model):
    weights = [p.ravel().detach() for p in model.parameters()]
    return torch.cat(weights)

def set_w(model, w):
    index = 0
    for param in model.parameters():
        param_size = torch.prod(torch.tensor(param.size())).item()
        param.data = w[index:index+param_size].view(param.size()).to(param.device)
        index += param_size

def set_lr(optimizer, lr):
    for param in optimizer.param_groups:
        param['lr'] = lr

def closure(dataset, model, criterion, device='cpu'):
    x, y = dataset
    x = x.unsqueeze(-1)
    y_pred = model(x)
    y_pred = y_pred.reshape(-1)
    loss = criterion(y_pred, y)
    return loss.item()

def closureStochastic(dataset, model, criterion, device='cpu'):
    xi, yi = dataset
    loss = 0
    for x, y in zip(xi, yi):
        # print(f"x: {x}, y: {y}")
        x, y = torch.tensor([x.item()]), torch.tensor([y.item()])
        x = x.unsqueeze(-1)
        y_pred = model(x)
        y_pred = y_pred.reshape(-1)
        # print(f"y_pred: {y_pred.size()}, y: {y.size()}")
        batch_loss = criterion(y_pred, y)
        loss += batch_loss.item() / len(xi)
    return loss

def closureMiniBatch(dataloader, model, criterion, device='cpu'):
    
    loss = 0
    for x, y in dataloader:
        # print(f"x: {x}, y: {y}")
        x = x.unsqueeze(-1)
        y_pred = model(x)
        y_pred = y_pred.reshape(-1)
        # print(f"y_pred: {y_pred.size()}, y: {y.size()}")
        batch_loss = criterion(y_pred, y)
        loss += batch_loss.item() * (len(x) / len(dataloader.dataset))
    return loss


# ============= BATCH(FULL) GRADIENT ==============

#Lineasearch arrives to 0.01 at 25-30 epochs
#classi stochastic arrives to 0.01 at 50 epochs

#NEW

#SGD
#linesearch 0.02 at 30 epochs
#no linesearch 1.85 t 30 epochs


#ADAM
#linesearch 0.02 at 300 epochs
#no linesearch 1.09 t 300 epochs

# doLinesearch = True
# start_time = time.time()
# for epoch in range(30):
#     model.train()
#     w_before = get_w(model)
#     # print(f"w_before: {w_before}")

#     optimizer.zero_grad()
    
    
#     x, y = dataset
#     x = x.unsqueeze(-1)
#     y_pred = model(x)
#     y_pred = y_pred.squeeze(-1)
#     loss = criterion(y_pred, y)
    
#     f_tilde = loss.item()

#     loss.backward()
#     optimizer.step()

    

#     # print(optimizer.m_hats)
#     # print(flat_m_hats)
#     if opt == 'sgd':
#         w_after = get_w(model)
#         direction = ((w_after - w_before) / curr_lr)

#     if opt == 'adam':
#         direction = -flatten_hats(optimizer.m_hats)
#         v_hats = flatten_hats(optimizer.v_hats)

#     if doLinesearch:
#         model.eval()

#         gradient_dir = torch.dot(direction, direction)
#         alfa, f_alfa = armijoMonotonePytorch(f_tilde, criterion, w_before, gamma, direction, gradient_dir, model, dataset, closure)
#         curr_lr = alfa
#         set_lr(optimizer, alfa)
#         print(f"final_f_alpha: {f_alfa}")
#         print()
#         # print(optimizer.param_groups)
#         # print(f"alfa: {alfa}")

# end_time = time.time()
# print(f"loss: {loss.item():.2f}, x: {x}, y: {y}, y_pred: {y_pred}")
# print(f"total_time: {end_time - start_time}")




# ============= STOCHASTIC(INCREMENTAL) GRADIENT ==============

# Lineasearch arrives to 0.00 at 40 epochs
# classi stochastic arrives to 0.01 at 100 epochs

#NEW 

#SGD
# Lineasearch arrives to 0.00 at 40 epochs
# classi stochastic arrives to 0.01 at 100 epochs

#ADAM 
#ALMOST THE SAME


doLinesearch = True
start_time = time.time()
for epoch in range(40):
    model.train()


    xi, yi = dataset
    total_loss = 0
    f_tilde = 0
    w_before = get_w(model)

    for x, y in zip(xi, yi):
        # print(x,y)
        x, y = torch.tensor([x.item()]), torch.tensor([y.item()])
        # w_before = get_w(model)
        # print(f"w_before: {w_before}")

        optimizer.zero_grad()
        
        
        x = x.unsqueeze(-1)
        y_pred = model(x)
        y_pred = y_pred.squeeze(-1)
        # print(f"y_pred: {y_pred.size()}, y: {y.size()}")
        loss = criterion(y_pred, y)
        
        f_tilde += loss.item() / len(xi)

        loss.backward()
        optimizer.step()

        total_loss += f_tilde
        

        # print(optimizer.m_hats)
        # print(flat_m_hats)
    if opt == 'sgd':
        w_after = get_w(model)
        direction = ((w_after - w_before) / curr_lr)

    if opt == 'adam':
        direction = -flatten_hats(optimizer.m_hats)

    if doLinesearch:
        model.eval()

        gradient_dir = torch.dot(direction, direction)
        alfa, f_alfa = armijoMonotonePytorchNew(f_tilde, criterion, w_before, gamma, direction, gradient_dir, model, dataset, closureStochastic)
        curr_lr = alfa
        set_lr(optimizer, alfa)
        print()
        # print(optimizer.param_groups)
        # print(f"alfa: {alfa}")
        # print(f"loss: {f_tilde:.2f}, x: {x}, y: {y}, y_pred: {y_pred}")
        

end_time = time.time()
print(f"loss: {total_loss:.2f}, x: {x}, y: {y}, y_pred: {y_pred}")
print(f"total_time: {end_time - start_time}")





# ============= MINI BATCH GRADIENT ==============
#Linesearch arrives at 200 while not line arrives at 300

#NEW 

# SGD
#ALMOST THE SAME (slightly worse with linesearch)

# ADAM
# linesearch arrives at 0.07 at 200
#linesearch arrives at 0.51 at 200

# doLinesearch = True
# start_time = time.time()
# for epoch in range(40):
#     model.train()


#     total_loss = 0
#     f_tilde = 0
#     w_before = get_w(model)
#     for index, (x, y) in enumerate(dataloader):
#         # print(x,y)
#         # x, y = torch.tensor([x.item()]), torch.tensor([y.item()])
        
#         # print(f"w_before: {w_before}")

#         optimizer.zero_grad()
        
        
#         x = x.unsqueeze(-1)
#         y_pred = model(x)
#         y_pred = y_pred.squeeze(-1)
#         # print(f"y_pred: {y_pred.size()}, y: {y.size()}")
#         loss = criterion(y_pred, y)
        
#         f_tilde += loss.item() * (len(x) / len(dataloader.dataset))

#         loss.backward()
#         optimizer.step()

#         # print(f"len(x): {len(x)}, len(dataloader.dataset): {len(dataloader.dataset)}")
#         # print(f"loss_inter {index}: {loss.item()}")

#         # total_loss += f_tilde
        

#         # print(optimizer.m_hats)
#         # print(flat_m_hats)

#     if opt == 'sgd':
#         w_after = get_w(model)
#         direction = ((w_after - w_before) / curr_lr)

#     if opt == 'adam':
#         direction = -flatten_hats(optimizer.m_hats)

#     if doLinesearch:
#         model.eval()
#         gradient_dir = torch.dot(direction, direction)
#         alfa, f_alfa = armijoMonotonePytorchNew(f_tilde, criterion, w_before, gamma, direction, gradient_dir, model, dataloader, closureMiniBatch)
#         curr_lr = alfa
#         set_lr(optimizer, alfa)
#         print()
#         # print(optimizer.param_groups)
#         # print(f"alfa: {alfa}")
#     # print(f"loss: {f_tilde:.2f}, x: {x}, y: {y}, y_pred: {y_pred}")
        

# end_time = time.time()
# print(f"loss: {f_tilde:.2f}, x: {x}, y: {y}, y_pred: {y_pred}")
# print(f"total_time: {end_time - start_time}")


'''
#Tabelle risultati
#change metrics (SSIM), FID

0)metrics per DDPM
1)sub dataset omogoneo(test 10, 20, 30 %)(Nel linesearch)
2.0)linesearch all'inizio test su ogni epoca
2)attivazione linesearch se soddisfatta una certa condizione(abbassamento della loss è proporzionale a lr non entrare in linesearch)
3)alternative di linesearch
4)set alpha iniziale linesearch un multiplo(non troppo grande) di attuale lr

4)Errore adam non accumulo m_hat
5)adam correzione alpha, tests(reset parametri se loss peggiora, update alpha per ogni parametro)
'''