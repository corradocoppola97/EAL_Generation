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

def armijo(f, criterion, w_before, gamma, direction, mod, dataset, closure, alfa=None, device='cpu') :
    '''
    Here we enforce the armijo condition, in order to accept only alpha_k that respect alpha_0*gamma*d_k^2
    This will enforce a better solution for alpha
    '''
    
    #alpha un ordine di grandezza maggiore(un multiplo > 1) dell'attuale lr
    if alfa == None:
        alfa_0 = 1
        alfa = 1
    else:
        alfa_0 = alfa

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
        # print(f"f_alfa: {f_alfa}, step: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")

        if f_alfa < best_f:
            best_f = f_alfa
            best_alfa = alfa

        alfa = alfa * 0.5
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)
        
        counter += 1
        
    return alfa, f_alfa

def armijoCustomEnforcedReduction(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, delta = 0.5, alfa=None, device='cpu') :
    '''
    Here we enforce the armijo reduction step, keeping to decrease alfa even if the armijo condition is True
    '''
    
    #alpha un ordine di grandezza maggiore(un multiplo > 1) dell'attuale lr
    if alfa == None:
        alfa_0 = 1
        alfa = 1
    else:
        alfa_0 = alfa

    sample_model = copy.deepcopy(mod)

    w_prova = w_before + alfa * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)

    f_alfa = closure(dataset, sample_model, criterion, device)
    f_increment = f + alfa * gamma * gradient_dir

    counter = 0
    best_alfa = alfa
    best_f = f_alfa
    last_alfa = f_alfa

    while(f_alfa > f_increment):

        if counter == max_iter_armijo:
            return best_alfa, best_f
        # print(f"f_alfa: {f_alfa}, step: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")

        if f_alfa < best_f:
            best_f = f_alfa
            best_alfa = alfa

        alfa = alfa * delta
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        
        last_alfa = f_alfa
        f_alfa = closure(dataset, sample_model, criterion, device)
        f_increment = f + alfa * gamma * gradient_dir

        counter += 1
        
    while(f_alfa <= last_alfa):

        alfa = alfa * delta
        w_prova = w_before + alfa * direction

        with torch.no_grad():
            set_w(sample_model, w_prova)
        
        last_alfa = f_alfa
        f_alfa = closure(dataset, sample_model, criterion, device)

    f_increment = f + alfa * gamma * gradient_dir
    if (last_alfa > f_increment):
        print(f"armijo criterion False, f_alfa: {f_alfa:.4f}, f_increment: {f_increment:.4f}")
    return alfa / delta, last_alfa

def armijoDecreasingZeta(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, delta = 0.5, alfa=None, device='cpu') :
    '''
    Here we enforce the armijo reduction step, keeping to decrease alfa even if the armijo condition is True
    '''
    
    #alpha un ordine di grandezza maggiore(un multiplo > 1) dell'attuale lr
    if alfa == None:
        alfa = 1

    sample_model = copy.deepcopy(mod)

    w_prova = w_before + alfa * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)

    zeta = 0.5
    last_zeta_increment = 0.5
    f_alfa = closure(dataset, sample_model, criterion, device)
    f_last = f_alfa
    f_increment = f + alfa * gamma * gradient_dir

    counter = 0
    condition_tol = False

    while(f_alfa > f_increment):

        #the func_value decreased
        alfa = alfa * zeta
            
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)

        f_alfa = closure(dataset, sample_model, criterion, device)
        f_increment = f + alfa * gamma * gradient_dir

    f_lasts = []
    n_increased = 0
    is_decrease = True

    while(not condition_tol):
        if n_increased >= 3:
            alfa = alfa / zeta
            break

        #the func_value decreased
        if f_alfa <= f_last:
            print(f"f_alfa: {f_alfa}, f_last: {f_last}, alfa: {alfa}, zeta: {zeta}")
            print()
            if np.abs(f_alfa - f_last) < 1e-4:
                break
            alfa = alfa * zeta
            n_increased = 0
            is_decrease = True
        
        #the func_value increased
        else:
            n_increased += 1
            alfa = alfa / zeta
            #TODO aggiornamento di zeta è sbagliato, cosi passa da 2 a 1 e si ferma, da cambiare
            incr_zeta = last_zeta_increment * 0.5
            zeta = zeta + incr_zeta
            last_zeta_increment = incr_zeta
            alfa = alfa * zeta
            is_decrease = False
            
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)

        if is_decrease:
            f_last = f_alfa
            is_decrease = True

        f_alfa = closure(dataset, sample_model, criterion, device)
        f_increment = f + alfa * gamma * gradient_dir

        # counter += 1

    return alfa, f_alfa

