import time
import torch
from torch import nn
from torch.nn import Parameter
from linesearchTest import armijoMonotone, armijoMonotonePytorch


x = torch.tensor([1,2,3,4], dtype=torch.float)
y = torch.tensor([11,12,13,14], dtype=torch.float)

dataset = (x, y)
device = 'cpu'


a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
# model = [Parameter(a), Parameter(b)]
model = nn.Linear(1,1)

lr = 0.001
curr_lr = lr
delta=1e-3
gamma=1e-3
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# print(model[0].data.item())
# print(optimizer)
# print(len(optimizer.param_groups))






# print(optimizer.param_groups[0])

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
    return loss


# print(list(model.parameters()))
# w_test = torch.tensor([1., 10.])
# set_w(model, w_test)
# print(model.weight)
# print(model.bias)

# loss = closure(dataset, model, criterion, device)
# print(loss)

doLinesearch = True
start_time = time.time()
for epoch in range(30):
    w_before = get_w(model)

    optimizer.zero_grad()
    
    x, y = dataset
    x = x.unsqueeze(-1)
    y_pred = model(x)
    y_pred = y_pred.squeeze(-1)
    loss = criterion(y_pred, y)
    
    f_tilde = loss.item()

    loss.backward()
    optimizer.step()

    w_after = get_w(model)

    direction = ((w_after - w_before) / curr_lr)

    if doLinesearch:
        gradient_dir = torch.dot(direction, direction)
        alfa, f_alfa = armijoMonotonePytorch(f_tilde, criterion, w_before, gamma, direction, gradient_dir, model, dataset, closure)
        curr_lr = alfa
        set_lr(optimizer, alfa)
        print(f"alfa: {alfa}")

end_time = time.time()
print(f"loss: {loss.item():.2f}, x: {x}, y: {y}, y_pred: {y_pred}")
print(f"total_time: {end_time - start_time}")



