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

def armijoBase(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, alfa=None, device='cpu') :
    '''
    Here we enforce the armijo condition, in order to accept only alpha_k that respect alpha_0*gamma*d_k^2
    This will enforce a better solution for alpha
    '''
    
    #alpha un ordine di grandezza maggiore(un multiplo > 1) dell'attuale lr
    if alfa == None:
        alfa = 1

    sample_model = copy.deepcopy(mod)

    w_prova = w_before + alfa * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)

    f_alfa = closure(dataset, sample_model, criterion, device)
    
    n_func_eval = 0
    while(f_alfa > (f + alfa * gamma * gradient_dir)):
        n_func_eval += 1

        print(f"f_alfa: {f_alfa}, f_increment: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")


        alfa = alfa * 0.8
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)
        
        
    return alfa, f_alfa, n_func_eval

def armijoBaseNorm(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, alfa=None, device='cpu') :
    '''
    Here we enforce the armijo condition, in order to accept only alpha_k that respect alpha_0*gamma*d_k^2
    This will enforce a better solution for alpha
    '''
    tol = 1e-2
    
    #alpha un ordine di grandezza maggiore(un multiplo > 1) dell'attuale lr
    if alfa == None:
        alfa = 1

    sample_model = copy.deepcopy(mod)

    w_prova = w_before + alfa * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)

    f_alfa = closure(dataset, sample_model, criterion, device)
    
    n_func_eval = 0
    while(np.linalg.norm(f_alfa - (f + alfa * gamma * gradient_dir)) < tol):
        n_func_eval += 1

        print(f"f_alfa: {f_alfa}, f_increment: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")


        alfa = alfa * 0.5
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)
        
        
    return alfa, f_alfa, n_func_eval

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
        print(f" step: {alfa * gamma * gradient_dir}")

        if f_alfa < best_f:
            best_f = f_alfa
            best_alfa = alfa

        alfa = alfa * 0.5
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)
        
        counter += 1
    # print(f"alfa: {alfa}")
    return alfa, f_alfa

def armijoGeneralized(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, delta = 0.5, alfa=None, device='cpu') :
    '''
    Armijo Generalized: TODO: not working
    It seems as the range is far away and it brings the function to diverge
    '''
    
    #alpha un ordine di grandezza maggiore(un multiplo > 1) dell'attuale lr
    if alfa == None:
        alfa = 1
    
    eta = 0.5




    sample_model = copy.deepcopy(mod)

    w_prova = w_before + alfa * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)

    f_alfa = closure(dataset, sample_model, criterion, device)
    

    while(f_alfa > (f + alfa * gamma * gradient_dir)):

        # print(f"prima fase, f_alfa: {f_alfa}, step: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")

        alfa = alfa * delta
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)
        
    # print(f"first stop, alfa: {alfa}, f_alfa: {f_alfa}, step: {f + alfa * gamma * gradient_dir}")

    w_prova = w_before + (alfa/eta) * direction

    with torch.no_grad():
        set_w(sample_model, w_prova)
    
    f_alfa = closure(dataset, sample_model, criterion, device)


    while(f_alfa <= (f + (alfa/eta) * gamma * gradient_dir)):

        # print(f"second phase, f_alfa: {f_alfa}, step: {f + (alfa/eta) * gamma * gradient_dir}, alfa: {alfa/eta}")

        alfa = alfa/eta
        if alfa >= 1:
            break

        w_prova = w_before + (alfa/eta) * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)
    
    # print(f"second stop, f_alfa: {f_alfa}, step: {f + (alfa/delta) * gamma * gradient_dir}, alfa: {alfa}")

    # w_prova = w_before + (alfa) * direction
    # with torch.no_grad():
    #     set_w(sample_model, w_prova)
    # f_double = closure(dataset, sample_model, criterion, device)
    # print(f"double check first stop, f_double: {f_double}, step: {f + (alfa) * gamma * gradient_dir}, alfa: {alfa}")
    # print()

    # if alfa > 1:
    #     return 1, f_alfa
    # else:
    #     return alfa, f_alfa
    
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
            # print(f"decrease alfa: {alfa}, f_alfa: {f_alfa}, check_gamma: {f + alfa * gamma * gradient_dir}, check_1meno_gamma: {f + alfa * (1-gamma) * gradient_dir}")
            
        elif (f_alfa < (f + alfa * (1-gamma) * gradient_dir)):
            alfa = min(alfa * eta, 1)
            # print(f"increase alfa: {alfa}, f_alfa: {f_alfa}, check_gamma: {f + alfa * gamma * gradient_dir}, check_1meno_gamma: {f + alfa * (1-gamma) * gradient_dir},")



        else:
            break

        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)
        f_alfa = closure(dataset, sample_model, criterion, device)

        
    # print(f"returned alfa: {alfa}")
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

def armijoDecreasingZetaBackup(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, delta = 0.5, alfa=None, device='cpu') :
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

    n_func_eval = 1

    while(f_alfa > f_increment):
        n_func_eval += 1

        #the func_value decreased
        alfa = alfa * zeta
            
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)

        f_alfa = closure(dataset, sample_model, criterion, device)
        f_increment = f + alfa * gamma * gradient_dir

        print(f"f_alfa: {f_alfa}, f_increment: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")


    f_lasts = []
    n_increased = 0
    is_decrease = True

    counter = 0
    while(not condition_tol and counter < 10):
        counter+=1
        # print(f"counter: {counter}")

        if n_increased >= 3:
            alfa = alfa / zeta
            break

        #the func_value decreased
        if f_alfa <= f_last:
            # print(f"f_alfa: {f_alfa}, f_last: {f_last}, alfa: {alfa}, zeta: {zeta}")
            # print()
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

        n_func_eval += 1
        # counter += 1

    return alfa, f_alfa, n_func_eval

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

    n_func_eval = 1

    best_f_alfa = f_alfa
    best_alfa = alfa

    while(f_alfa > f_increment and counter < 5):
        n_func_eval += 1

        #the func_value decreased
        alfa = alfa * zeta
            
        w_prova = w_before + alfa * direction
        with torch.no_grad():
            set_w(sample_model, w_prova)

        f_alfa = closure(dataset, sample_model, criterion, device)
        f_increment = f + alfa * gamma * gradient_dir

        if f_alfa < best_alfa:
            best_f_alfa = f_alfa
            best_alfa = alfa

        counter += 1
        print(f"f_alfa: {f_alfa}, f_increment: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")


    f_lasts = []
    n_increased = 0
    is_decrease = True

    # f_alfa = best_f_alfa
    # alfa = best_alfa

    while(not condition_tol and counter < 10):
        counter+=1
        # print(f"counter: {counter}")

        if n_increased >= 3:
            alfa = alfa / zeta
            break

        #the func_value decreased
        if f_alfa <= f_last:
            # print(f"f_alfa: {f_alfa}, f_last: {f_last}, alfa: {alfa}, zeta: {zeta}")
            # print()
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

        n_func_eval += 1

        print(f"f_alfa: {f_alfa}, f_increment: {f + alfa * gamma * gradient_dir}, alfa: {alfa}")
        # counter += 1

    print(f"returning alfa: {alfa}")
    return alfa, f_alfa, n_func_eval

def armijo_improved(f, criterion, w_before, gamma, direction, gradient_dir, mod, dataset, closure, alfa=None, device='cpu'):
    '''
    Improved version of the Armijo line search.
    Continues searching for a better alpha even if the condition is satisfied.
    Saves the best f_alfa obtained and increments alpha if function value worsens.
    Stops if the function value does not improve for 3 consecutive times.
    '''

    if alfa is None:
        alfa = 1

    sample_model = copy.deepcopy(mod)

    f_alfa = 0
    best_alfa = alfa
    best_f_alfa = float('inf')
    no_improve_count = 0

    power = 0

    armijo_condition = False

    while no_improve_count < 3:
        power += 1
        increment_alfa = 1 / (2**power)


        w_prova = w_before + alfa * direction

        with torch.no_grad():
            set_w(sample_model, w_prova)

        f_alfa = closure(dataset, sample_model, criterion, device)
        # print(f"f_alfa: {f_alfa}, alfa: {alfa}, best_alfa: {best_alfa}, step: {(f + alfa * gamma * gradient_dir)}")


        if armijo_condition:
            if f_alfa < best_f_alfa and f_alfa <= (f + alfa * gamma * gradient_dir):
                best_alfa = alfa
                best_f_alfa = f_alfa
                no_improve_count = 0
                alfa -= increment_alfa
                
            else:
                if alfa >= 1:
                    break
                
                no_improve_count += 1
                alfa += increment_alfa

        if not armijo_condition:
            if f_alfa > (f + alfa * gamma * gradient_dir):
                alfa *= 0.5
            else:
                best_alfa = alfa
                best_f_alfa = f_alfa
                armijo_condition = True
                alfa *= 0.5
                # print(f"armijo condition True")
        
        
        
        
            
    # print()
    return best_alfa, best_f_alfa


