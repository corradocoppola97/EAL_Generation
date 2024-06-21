import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import copy

from linesearchAdam import armijo
from linesearchAdam import armijoCustomEnforcedReduction, armijoDecreasingZeta



# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = 'cpu'


lr = 0.1
curr_lr = lr
delta=1e-3
gamma=1e-4
eps = 5 * (1e-8)


# Load the dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

dataset = (X_train, y_train)

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

def flatten_hats(m_hats):
    weights = [p.ravel().detach() for p in m_hats]
    return torch.cat(weights)

def closure(dataset, model, criterion, device='cpu'):
    x, y = dataset
    predictions = model(x)
    test_loss = criterion(predictions, y)
    return test_loss.item()

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Initialize the model, loss function, and optimizer
model = MLP(input_dim=X_train.shape[1])
criterion = nn.MSELoss()

opt = 'sgd'
if opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr = lr)
elif opt == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Training loop
num_epochs = 1000

doLinesearch = False
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    w_before = get_w(model)

    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    f_tilde = loss.item()

    loss.backward()
    optimizer.step()

    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    if doLinesearch:
        if opt == 'sgd':
            w_after = get_w(model)
            m_hats = flatten_hats(optimizer.m_hats)
            v_hats = flatten_hats(optimizer.v_hats)

            direction = -m_hats / (torch.sqrt(v_hats) + eps)

            with torch.no_grad():
                sample_model = copy.deepcopy(model)
                set_w(sample_model, w_before)

                f_w = closure(dataset, sample_model, criterion)

                #TODO da completare

        # gradient_dir = torch.dot(direction, direction)
        if curr_lr * 10 > 1:
            alfa = 1
        else:
            alfa = curr_lr * 10
        alfa, f_alfa = armijo(f_tilde, criterion, w_before, gamma, direction, model, dataset, closure)
        curr_lr = alfa
        set_lr(optimizer, alfa)
        # set_w(model, w_prova)
        
        # print()

end_time = time.time()
print(f"total_time: {end_time - start_time}")

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    r2 = r2_score(y_test.numpy(), predictions.numpy())
    print(f'Test MSE: {test_loss.item():.4f}')
    print(f'R-squared: {r2:.4f}')

# print(closure((X_test, y_test), model, criterion))
