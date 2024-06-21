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

def armijo(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, alfa=None, device='cpu') :
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

def armijoGeneralized(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, delta = 0.5, alfa=None, device='cpu') :
    '''
    Armijo Generalized: TODO: not working
    It seems as the range is far away and it brings the function to diverge
    '''
    
    #alpha un ordine di grandezza maggiore(un multiplo > 1) dell'attuale lr
    if alfa == None:
        alfa = 1

    sample_model = copy.deepcopy(mod)

    w_prova = w_before + alfa * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)

    f_alfa = closure(dataset, sample_model, criterion, device)
    

    while(f_alfa > (f + alfa * gamma * gradient_dir)):

        print(f"prima fase, f_alfa: {f_alfa}, step: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")

        alfa = alfa * delta
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)
        
    print(f"first stop, alfa: {alfa}, f_alfa: {f_alfa}, step: {f + alfa * gamma * gradient_dir}")

    w_prova = w_before + (alfa/delta) * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)
    
    f_alfa = closure(dataset, sample_model, criterion, device)

    while(f_alfa <= (f + (alfa/delta) * gamma * gradient_dir)):

        print(f"second phase, f_alfa: {f_alfa}, step: {f + (alfa/delta) * gamma * gradient_dir}, alfa: {alfa/delta}")

        alfa = alfa/delta
        w_prova = w_before + (alfa/delta) * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)
    
    print(f"second stop, f_alfa: {f_alfa}, step: {f + (alfa/delta) * gamma * gradient_dir}, alfa: {alfa}")

    w_prova = w_before + (alfa) * direction
    with torch.no_grad():
        set_w(sample_model, w_prova)
    f_double = closure(dataset, sample_model, criterion, device)
    print(f"double check first stop, f_double: {f_double}, step: {f + (alfa) * gamma * gradient_dir}, alfa: {alfa}")
    print()

    # if alfa > 1:
    #     return 1, f_alfa
    # else:
    #     return alfa, f_alfa
    
    return alfa, f_alfa

def armijoGoldstein(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, delta = 0.5, alfa=None, device='cpu') :
    '''
    Armijo with goldstein conditions: TODO: not working
    It doesn't move, remains always in the same point returning alfa = 0 almost always
    '''

    # gamma1 = 1e-4
    # gamma2 = 1e-1
    
    #alpha un ordine di grandezza maggiore(un multiplo > 1) dell'attuale lr
    if alfa == None:
        alfa = 1

    sample_model = copy.deepcopy(mod)

    w_prova = w_before + alfa * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)

    f_alfa = closure(dataset, sample_model, criterion, device)
    
    while(True):
        if (f_alfa <= (f + alfa * gamma * gradient_dir)) and (f_alfa >= (f + alfa * (1-gamma) * gradient_dir)):
            break
        
        alfa = alfa * 0.5
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)

    # while(f_alfa > (f + alfa * gamma1 * gradient_dir) or f_alfa < (f + alfa * gamma2 * gradient_dir)):

    #     # print(f"f_alfa: {f_alfa}, step: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")

    #     alfa = alfa / 2
    #     w_prova = w_before + alfa * direction
    #     with torch.no_grad():
    #         set_w(sample_model, w_prova)
    #     f_alfa = closure(dataset, sample_model, criterion, device)
        
    print(f"returned alfa: {alfa}")
    return alfa, f_alfa

def armijoGoldstein2(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, delta = 0.5, alfa=None, device='cpu') :
    '''
    Armijo with goldstein conditions: TODO: not working
    I range del check sembrano invertiti, non soddisfa mai le condizioni
    '''

    # gamma1 = 1e-4
    # gamma2 = 1e-1

    eta = 1.1
    
    #alpha un ordine di grandezza maggiore(un multiplo > 1) dell'attuale lr
    if alfa == None:
        alfa = 1

    sample_model = copy.deepcopy(mod)

    w_prova = w_before + alfa * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)

    f_alfa = closure(dataset, sample_model, criterion, device)
    
    while(True):
        if (f_alfa > (f + alfa * gamma * gradient_dir)):
            alfa = alfa * 0.75
            print(f"decrease alfa: {alfa}, f_alfa: {f_alfa}, check_gamma: {f + alfa * gamma * gradient_dir}, check_1meno_gamma: {f + alfa * (1-gamma) * gradient_dir}")
            
        elif (f_alfa < (f + alfa * (1-gamma) * gradient_dir)):
            alfa = min(alfa * eta, 1)
            print(f"increase alfa: {alfa}, f_alfa: {f_alfa}, check_1meno_gamma: {f + alfa * (1-gamma) * gradient_dir}, check_gamma: {f + alfa * gamma * gradient_dir}")



        else:
            break

        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)

        
    print(f"returned alfa: {alfa}")
    return alfa, f_alfa

def armijoBestAlfa(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, alfa=None, device='cpu') :
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

        alfa = alfa/2
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)
        
        counter += 1
        
    return alfa, f_alfa

def armijoCustomEnforcedCondition(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, alfa=None, device='cpu') :
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

    while(f_alfa > (f + alfa_0 * gamma * gradient_dir)):

        if counter == max_iter_armijo:
            return best_alfa, best_f
        # print(f"f_alfa: {f_alfa}, step: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")

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

def armijoBinarySearch(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, delta = 0.5, alfa=None, device='cpu') :
    '''
    Here we enforce the armijo reduction step, keeping to decrease alfa even if the armijo condition is True
    '''

    tol = 1e-4
    
    #alpha un ordine di grandezza maggiore(un multiplo > 1) dell'attuale lr
    if alfa == None:
        alfa = 1

    sample_model = copy.deepcopy(mod)

    w_prova = w_before + alfa * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)

    f_alfa = closure(dataset, sample_model, criterion, device)
    f_increment = f + alfa * gamma * gradient_dir

    zeta = delta
    last_zeta = zeta
    counter = 0

    last_f = f_alfa

    is_decreasing = True
    index_power = 1

    while(True):

        if counter != 0: 
            print()

            if f_alfa <= f_increment:
                if f_alfa <= last_f and np.abs(f_alfa - f_increment) > tol:
                    best_f_alfa = f_alfa
                    best_alfa = alfa
                    last_f = f_alfa


                if is_decreasing:
                    alfa = alfa + zeta

                else:
                    print()

        else:
            if f_alfa <= f_increment:
                best_f_alfa = f_alfa
                best_alfa = alfa


            #the func_value decreased
            alfa = alfa * zeta
            
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)

        f_alfa = closure(dataset, sample_model, criterion, device)
        f_increment = f + alfa * gamma * gradient_dir

        counter += 1

    return alfa, f_alfa