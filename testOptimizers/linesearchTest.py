import numpy as np
import torch
import torchvision
import copy

max_it_EDFL = 1e2
max_iter_armijo = 10

def set_w(model, w):
    index = 0
    for param in model.parameters():
        param_size = torch.prod(torch.tensor(param.size())).item()
        param.data = w[index:index+param_size].view(param.size()).to(param.device)
        index += param_size

def armijoMonotone(f, criterion, w_before, gamma, direction, gradient_dir, x, y, alfa=None) :
    
    if alfa == None:
        alfa = 1

    w_prova = w_before + alfa * direction

    y_pred = w_prova[0] + w_prova[1] * x

    f_alfa = criterion(y_pred, y)
    
    while(f_alfa > (f + alfa * gamma * gradient_dir)):
        alfa = alfa/2
        w_prova = w_before + alfa * direction
        y_pred = w_prova[0] + w_prova[1] * x
        f_alfa = criterion(y_pred, y)
        
        
    return alfa, f_alfa

def armijoMonotonePytorchNew(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, alfa=None, device='cpu') :
    
    #alpha un ordine di grandezza maggiore(un multiplo > 1) dell'attuale lr
    if alfa == None:
        alfa = 1

    sample_model = copy.deepcopy(mod)

    w_prova = w_before + alfa * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)

    f_alfa = closure(dataset, sample_model, criterion, device)
    
    counter = 0
    best_alfa = alfa
    best_f = f_alfa
    while(f_alfa > (f + alfa * gamma * gradient_dir)):
        if counter == max_iter_armijo:
            return best_alfa, best_f
        print(f"f_alfa: {f_alfa}, step: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")

        if f_alfa < best_f:
            best_f = f_alfa
            best_alfa = alfa

        alfa = alfa/2
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)
        
        counter += 1
        
    return alfa, f_alfa

def armijoMonotonePytorch(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, alfa=None, device='cpu') :
    
    if alfa == None:
        alfa = 1

    sample_model = copy.deepcopy(mod)

    w_prova = w_before + alfa * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)

    f_alfa = closure(dataset, sample_model, criterion, device)
    
    while(f_alfa > (f + alfa * gamma * gradient_dir)):
        print(f"f_alfa: {f_alfa}, step: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")
        alfa = alfa * 0.5
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)

        
        
    return alfa, f_alfa

def EDFL(mod,
             dl_train,
             w_before: torch.Tensor,
             f_tilde: float,
             d_k: torch.Tensor,
             closure: callable,
             device: torch.device,
             criterion: torch.nn, 
             zeta,
             gamma,
             delta,
             verbose = True):

        alpha = zeta
        nfev = 0
        sample_model = copy.deepcopy(mod)
        real_loss = closure(dl_train,sample_model,criterion,device)
        if verbose: print(f'Starting EDFL  with alpha =  {alpha}    f_tilde = {f_tilde}    real_loss_before = {real_loss}')
        nfev += 1
        if f_tilde > real_loss - gamma * alpha * torch.linalg.norm(d_k) ** 2:
            if verbose: print('fail: ALPHA = 0')
            alpha = 0
            return alpha, nfev, f_tilde

        w_prova = w_before + d_k * (alpha / delta)
        # print(f'w_prova = {w_prova}, w_before: {w_before}, d_k: {d_k}, alpha: {alpha}, delta: {delta}')

        sample_model = [w_prova[0], w_prova[1]]

        cur_loss = closure(dl_train,sample_model,criterion,device)
        # print(f'sample_model = {sample_model}')
        # print(f'real_loss = {real_loss}')
        print(f'cur loss = {cur_loss}')
        nfev += 1

        idx = 0
        f_j = f_tilde
        while cur_loss <= min(f_j,real_loss - gamma * alpha * torch.linalg.norm(d_k) ** 2) and idx <= max_it_EDFL:
            if verbose: print(f'idx = {idx}   cur_loss = {cur_loss}')
            f_j = cur_loss
            alpha = alpha / delta
            w_prova = w_before + d_k * (alpha / delta)
            sample_model = [w_prova[0], w_prova[1]]
            cur_loss = closure(dl_train,sample_model,criterion,device)
            nfev += 1
            idx += 1

        return alpha, nfev, f_j
