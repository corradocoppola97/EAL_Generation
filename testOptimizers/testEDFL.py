import time
import torch
from torch import nn
from torch.nn import Parameter
from linesearchTest import armijoMonotone, EDFL


x = torch.tensor([1,2,3,4])
y = torch.tensor([11,12,13,14], dtype=torch.float)

dataset = (x, y)
device = 'cpu'

a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
model = [Parameter(a), Parameter(b)]

lr = 0.01
curr_lr = lr
delta=1e-3
gamma=1e-3
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model, lr=lr)
doLinesearch = True
# print(model[0].data.item())
# print(optimizer)
# print(len(optimizer.param_groups))

# print(optimizer.param_groups[0])

def get_w(model):
    weights = [p.ravel().detach() for p in model]
    return torch.cat(weights)

def set_w(weights):
    return [weights[0], weights[1]]

def set_lr(optimizer, lr):
    for param in optimizer.param_groups:
        param['lr'] = lr

def closure(dataset, model, criterion, device):
    x, y = dataset
    y_pred = model[0] + model[1] * x
    loss = criterion(y_pred, y)
    return loss

losses = []
start_time = time.time()
for epoch in range(20):
    w_before = get_w(model)

    optimizer.zero_grad()
    y_pred = model[0] + model[1] * x
    loss = criterion(y_pred, y)
    
    f_tilde = loss.item()

    loss.backward()
    optimizer.step()

    w_after = get_w(model)

    direction = ((w_after - w_before) / curr_lr)

    if doLinesearch:
        # print(f"direction: {direction}")
        alfa, nf, f_alfa = EDFL(model, dataset, w_before, f_tilde, direction, closure, device, criterion, curr_lr, gamma, delta)
        # curr_lr = alfa
        # print(model)
        # set_lr(optimizer, alfa)
        # print(f"alfa: {alfa}")

end_time = time.time()
print(f"loss: {loss.item():.2f}, x: {x}, y: {y}, y_pred: {y_pred}")
print(f"total_time: {end_time - start_time}")



