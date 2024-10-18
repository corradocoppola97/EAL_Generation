import time, os , argparse
import torch
import torchvision
from torch.utils.data import Subset, SubsetRandomSampler, DataLoader
from warnings import filterwarnings
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import copy

from utils.utils import closure, count_parameters, set_optimizer, accuracy, hardware_check
from optimizers.cmalight import get_w, set_w
from networks.network import get_pretrained_net
from testLinesearch.linesearchSGD import armijoDecreasingZeta


filterwarnings('ignore')

def set_lr(optimizer, lr):
    for param in optimizer.param_groups:
        param['lr'] = lr

gamma = 1e-4

def train_model(model,
                optimizer,
                sm_root: str,
                opt: str,
                ep: int,
                ds: str,
                net_name: str,
                n_class: int,
                history_ID: str,
                doLinesearch: bool,
                dts_percentage: float,
                batch_size: int,
                dts_train: torch.utils.data.DataLoader,
                dts_test: torch.utils.data.DataLoader,
                verbose_train: bool) -> dict:
    
    print('\n ------- Begin training process ------- \n')

    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # Model
    # model = get_pretrained_net(net_name, num_classes=n_class, dataset_name=ds).to(device)
    print('\n The model has: {} trainable parameters'.format(count_parameters(model)))
    # Loss
    criterion = torch.nn.CrossEntropyLoss()
    # Optimizer
    # optimizer = set_optimizer(opt, model)

    #Initial lr
    init_lr = 0.01
    curr_lr = 0.01


    # Initial Setup
    min_acc = 0
    t1 = time.time()
    fw0 = closure(dts_train, model, criterion, device)
    t2 = time.time()

    time_compute_fw0 = t2 - t1  # To be added to the elapsed time in case we are using CMA Light (information used)
    initial_val_loss = closure(dts_test, model, criterion, device)
    train_accuracy = accuracy(dts_train, model, device)
    val_acc = accuracy(dts_test, model, device)
    f_tilde = fw0

    if opt == 'cmal':
        optimizer.set_f_tilde(f_tilde)
        optimizer.set_phi(f_tilde)
        optimizer.set_fw0(fw0)

    history = {'train_loss': [fw0], 'val_loss': [initial_val_loss], 'train_acc': [train_accuracy],
               'val_acc': [val_acc], 'step_size':[],
               'time_4_epoch': [0.0], 'nfev': 1, 'accepted': [], 'Exit': [], 'comments': [],
               'elapsed_time_noVAL': [0.0], 'f_tilde': []}
    
    # Train
    # print("Strating training loop")
    for epoch in range(ep):
        start_time = time.time()
        model.train()
        f_tilde = 0
        if opt == 'cmal' or opt == 'sgd':
            w_before = get_w(model)
        
        with tqdm(dts_train, unit="step", position=0, leave=True) as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{ep} - Training")
                x, y = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                f_tilde += loss.item() * (len(x) / len(dts_train.dataset))
                if verbose_train:
                    print('f_tilde: ',f_tilde)
                    print('loss: ',loss)
                loss.backward()
                optimizer.step()

        history['f_tilde'].append(f_tilde)

        # CMAL support functions
        if opt == 'cmal':
            optimizer.set_f_tilde(f_tilde)
            model, history, f_after, exit = optimizer.control_step(model, w_before, closure, dts_train, device, criterion, history, epoch)
            optimizer.set_phi(min(f_tilde, f_after))
        else:
            f_after = f_tilde
            
        elapsed_time_4_epoch_noVAL = time.time() - start_time

        # Validation
        model.eval()
        val_loss = closure(dts_test, model, criterion, device)
        val_acc = accuracy(dts_test, model, device)
        train_accuracy = accuracy(dts_train, model, device)

        elapsed_time_4_epoch = time.time() - start_time

        if doLinesearch == True and opt == 'sgd':
            model.eval()

            if epoch == 0:
                f_prev = fw0
            else:
                f_prev = history['f_tilde'][epoch - 1]

            if f_tilde >= f_prev:
                print(f"linesearch f_tilde not decreased, f_tilde: {f_tilde:.5f}, f_prev: {f_prev:.5f}")
                


                if epoch == 0:
                    f_start = fw0
                else:
                    # f_start = history['f_tilde'][epoch - 1]
                    sample_model = copy.deepcopy(model)
                    with torch.no_grad():
                        set_w(sample_model, w_before)
                        f_start = closure(dts_train, sample_model, criterion, device)


                # Create a DataLoader for the entire dataset

                # Generate random indices for the subset
                random_indices = np.random.choice(len(dts_train.dataset), size=int(len(dts_train.dataset)*dts_percentage), replace=False)

                # Create a SubsetRandomSampler using these indices
                subset_sampler = SubsetRandomSampler(random_indices)

                # Create a DataLoader for the subset
                subset_loader = DataLoader(dts_train.dataset, batch_size=batch_size, sampler=subset_sampler)


                w_after = get_w(model)
                direction = ((w_after - w_before) / curr_lr)

                gradient_dir = - torch.dot(direction, direction)

                if curr_lr * 2 > init_lr:
                    alfa = init_lr
                elif curr_lr * 2 > 1:
                    alfa = 1
                else:
                    alfa = curr_lr * 2

                
                alfa, f_alfa, n_func_eval = armijoDecreasingZeta(f_start, criterion, w_before, gamma, direction, gradient_dir, model, subset_loader, closure, alfa=alfa, device=device)
                print(f"linesearch ended, f_start: {f_start}, f_end: {f_alfa}")
                
                if f_alfa < f_start:
                    curr_lr = alfa
                    set_w(model, w_before)
                    set_lr(optimizer, alfa)
                print(f"alfa: {alfa}, n_func_eval: {n_func_eval}")


        history['train_loss'].append(f_after)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_acc)
        history['time_4_epoch'].append(history['time_4_epoch'][-1] + elapsed_time_4_epoch)
        history['elapsed_time_noVAL'].append(history['elapsed_time_noVAL'][-1] + elapsed_time_4_epoch_noVAL)
        if epoch == 0 and opt == 'cmal':
            history['time_4_epoch'][-1] += time_compute_fw0
            history['elapsed_time_noVAL'][-1] += time_compute_fw0

        # Save data during training
        if min_acc < val_acc:
            torch.save(model, sm_root + 'train_' + opt + '_' + ds + '_' + net_name + '_model_best.pth')
            min_acc = val_acc
            print('\n - New best Val-ACC: {:.3f} at epoch {} - \n'.format(min_acc, epoch + 1))

        torch.save(history, sm_root + 'history_' + opt + '_' + ds + '_' + net_name + '_' + history_ID + '.txt')

    print('\n - Finished Training - \n')
    torch.save(history, sm_root + 'history_' + opt + '_' + ds + '_' + net_name + '_' + history_ID + '.txt')
    return history


if __name__ == '__main__':

    # Setup in cmd line
    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=float, default=12345)
    # parser.add_argument('--network', type=str, required=True)
    # parser.add_argument('--opt', type=str, required=True)
    parser.add_argument('--network', type=str, default='resnet18')
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--dts', type=str, default='cifar10')
    parser.add_argument('--dts_root', type=str, default='/work/datasets/')
    parser.add_argument('--sm_root', type=str, default='/work/results/classification')
    parser.add_argument('--bs', type=str, default=16)

    parser.add_argument('--trial', type=str, default='classification_10')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
 
    hardware_check()

    dts_root = args.dts_root
    bs = args.bs

    if args.dts == 'cifar10': # Classification
        transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.RandomRotation(10),
                                                torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=dts_root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=dts_root, train=False, download=True, transform=transform)
        num_classes = len(trainset.classes)
    
    elif args.dts == 'cifar100': # Classification
        transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.RandomRotation(10),
                                                torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR100(root=dts_root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=dts_root, train=False, download=True, transform=transform)
        num_classes = len(trainset.classes)
    

    trainset = Subset(trainset, range(bs*4))
    testset = Subset(testset, range(bs*4))
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True) # Togliere random reshuffle --> shuffle=False
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ==================== resnet18 ========================

    model = get_pretrained_net('resnet18', num_classes=num_classes, dataset_name=args.dts).to(device)
    optimizer = set_optimizer('sgd', model)
    args.opt='sgd'

    
    history = train_model(model=model,
                          optimizer=optimizer,
                          sm_root=args.sm_root, 
                          opt=args.opt, 
                          ep=args.ep, 
                          ds=args.dts, 
                          net_name=args.network, 
                          n_class=num_classes, 
                          history_ID=args.trial,
                          doLinesearch=True,
                          dts_percentage=0.5,
                          batch_size=bs,
                          dts_train=trainloader, 
                          dts_test=testloader,
                          verbose_train=False)
    
    # ==================== resnet50 ========================

    model = get_pretrained_net('resnet50', num_classes=num_classes, dataset_name=args.dts).to(device)
    optimizer = set_optimizer('sgd', model)
    args.opt='sgd'

    
    history = train_model(model=model,
                          optimizer=optimizer,
                          sm_root=args.sm_root, 
                          opt=args.opt, 
                          ep=args.ep, 
                          ds=args.dts, 
                          net_name=args.network, 
                          n_class=num_classes, 
                          history_ID=args.trial,
                          doLinesearch=True,
                          dts_percentage=0.5,
                          batch_size=bs,
                          dts_train=trainloader, 
                          dts_test=testloader,
                          verbose_train=False)
    
    # ==================== resnet152 ========================


    model = get_pretrained_net('resnet152', num_classes=num_classes, dataset_name=args.dts).to(device)
    optimizer = set_optimizer('sgd', model)
    args.opt='sgd'

    
    history = train_model(model=model,
                          optimizer=optimizer,
                          sm_root=args.sm_root, 
                          opt=args.opt, 
                          ep=args.ep, 
                          ds=args.dts, 
                          net_name=args.network, 
                          n_class=num_classes, 
                          history_ID=args.trial,
                          doLinesearch=True,
                          dts_percentage=0.5,
                          batch_size=bs,
                          dts_train=trainloader, 
                          dts_test=testloader,
                          verbose_train=False)
    
    # # ==================== mobilenet_v2 ========================



    # model = get_pretrained_net('mobilenet_v2', num_classes=num_classes, dataset_name=args.dts).to(device)
    # optimizer = set_optimizer('sgd', model)
    # args.opt='sgd'

    
    # history = train_model(model=model,
    #                       optimizer=optimizer,
    #                       sm_root=args.sm_root, 
    #                       opt=args.opt, 
    #                       ep=args.ep, 
    #                       ds=args.dts, 
    #                       net_name=args.network, 
    #                       n_class=num_classes, 
    #                       history_ID=args.trial,
    #                       doLinesearch=True,
    #                       dts_percentage=0.5,
    #                       batch_size=bs,
    #                       dts_train=trainloader, 
    #                       dts_test=testloader,
    #                       verbose_train=False)
    
    # # ==================== wide_resnet50 ========================


    # model = get_pretrained_net('wide_resnet50', num_classes=num_classes, dataset_name=args.dts).to(device)
    # optimizer = set_optimizer('sgd', model)
    # args.opt='sgd'

    
    # history = train_model(model=model,
    #                       optimizer=optimizer,
    #                       sm_root=args.sm_root, 
    #                       opt=args.opt, 
    #                       ep=args.ep, 
    #                       ds=args.dts, 
    #                       net_name=args.network, 
    #                       n_class=num_classes, 
    #                       history_ID=args.trial,
    #                       doLinesearch=True,
    #                       dts_percentage=0.5,
    #                       batch_size=bs,
    #                       dts_train=trainloader, 
    #                       dts_test=testloader,
    #                       verbose_train=False)
    
    # # ==================== efficientnet_v2_l ========================


    # model = get_pretrained_net('efficientnet_v2_l', num_classes=num_classes, dataset_name=args.dts).to(device)
    # optimizer = set_optimizer('sgd', model)
    # args.opt='sgd'

    
    # history = train_model(model=model,
    #                       optimizer=optimizer,
    #                       sm_root=args.sm_root, 
    #                       opt=args.opt, 
    #                       ep=args.ep, 
    #                       ds=args.dts, 
    #                       net_name=args.network, 
    #                       n_class=num_classes, 
    #                       history_ID=args.trial,
    #                       doLinesearch=True,
    #                       dts_percentage=0.5,
    #                       batch_size=bs,
    #                       dts_train=trainloader, 
    #                       dts_test=testloader,
    #                       verbose_train=False)


    # # ==================== swin_t ========================


    # model = get_pretrained_net('swin_t', num_classes=num_classes, dataset_name=args.dts).to(device)
    # optimizer = set_optimizer('sgd', model)
    # args.opt='sgd'

    
    # history = train_model(model=model,
    #                       optimizer=optimizer,
    #                       sm_root=args.sm_root, 
    #                       opt=args.opt, 
    #                       ep=args.ep, 
    #                       ds=args.dts, 
    #                       net_name=args.network, 
    #                       n_class=num_classes, 
    #                       history_ID=args.trial,
    #                       doLinesearch=True,
    #                       dts_percentage=0.5,
    #                       batch_size=bs,
    #                       dts_train=trainloader, 
    #                       dts_test=testloader,
    #                       verbose_train=False)


    # # ==================== swin_v2_t ========================


    # model = get_pretrained_net('swin_v2_t', num_classes=num_classes, dataset_name=args.dts).to(device)
    # optimizer = set_optimizer('sgd', model)
    # args.opt='sgd'

    
    # history = train_model(model=model,
    #                       optimizer=optimizer,
    #                       sm_root=args.sm_root, 
    #                       opt=args.opt, 
    #                       ep=args.ep, 
    #                       ds=args.dts, 
    #                       net_name=args.network, 
    #                       n_class=num_classes, 
    #                       history_ID=args.trial,
    #                       doLinesearch=True,
    #                       dts_percentage=0.5,
    #                       batch_size=bs,
    #                       dts_train=trainloader, 
    #                       dts_test=testloader,
    #                       verbose_train=False)


    # # ==================== swin_b ========================


    # model = get_pretrained_net('swin_b', num_classes=num_classes, dataset_name=args.dts).to(device)
    # optimizer = set_optimizer('sgd', model)
    # args.opt='sgd'

    
    # history = train_model(model=model,
    #                       optimizer=optimizer,
    #                       sm_root=args.sm_root, 
    #                       opt=args.opt, 
    #                       ep=args.ep, 
    #                       ds=args.dts, 
    #                       net_name=args.network, 
    #                       n_class=num_classes, 
    #                       history_ID=args.trial,
    #                       doLinesearch=True,
    #                       dts_percentage=0.5,
    #                       batch_size=bs,
    #                       dts_train=trainloader, 
    #                       dts_test=testloader,
    #                       verbose_train=False)
    
    # # ==================== swin_v2_b ========================


    # model = get_pretrained_net('swin_v2_b', num_classes=num_classes, dataset_name=args.dts).to(device)
    # optimizer = set_optimizer('sgd', model)
    # args.opt='sgd'

    
    # history = train_model(model=model,
    #                       optimizer=optimizer,
    #                       sm_root=args.sm_root, 
    #                       opt=args.opt, 
    #                       ep=args.ep, 
    #                       ds=args.dts, 
    #                       net_name=args.network, 
    #                       n_class=num_classes, 
    #                       history_ID=args.trial,
    #                       doLinesearch=True,
    #                       dts_percentage=0.5,
    #                       batch_size=bs,
    #                       dts_train=trainloader, 
    #                       dts_test=testloader,
    #                       verbose_train=False)


    