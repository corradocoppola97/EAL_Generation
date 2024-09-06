import time, os , argparse
import json
import torch
import torchvision
from torch.utils.data import Subset, TensorDataset, SubsetRandomSampler, DataLoader
from warnings import filterwarnings
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import copy


from utils.utils import closure, closure_diffusion, count_parameters, set_optimizer, set_scheduler, accuracy, accuracy_diffusion, hardware_check
from optimizers.cmalight import get_w, set_w
from networks.network import get_pretrained_net, get_diffusion_model
from utils.utils_diffusion_model import T
from networks.diffusionModel import forward_diffusion_sample
from testLinesearch.linesearchSGD import armijoDecreasingZeta, armijo, armijoBase

filterwarnings('ignore')

def set_lr(optimizer, lr):
    for param in optimizer.param_groups:
        param['lr'] = lr

gamma = 1e-4

def train_model(sm_root: str,
                opt: str,
                slr: str,
                ep: int,
                ds: str,
                net_name: str,
                n_class: int,
                history_ID: str,
                doLinesearch: bool,
                dts_train: torch.utils.data.DataLoader,
                dts_test: torch.utils.data.DataLoader,
                verbose_train: bool,
                checkpoint: str = None,
                ) -> dict:
    
    print('\n ------- Begin training process ------- \n')

    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # Model
    model = get_diffusion_model(num_classes=n_class, checkpoint=checkpoint).to(device)
    print('\n The model has: {} trainable parameters'.format(count_parameters(model)))
    # Loss
    criterion = F.mse_loss
    # criterion = F.l1_loss
    # Optimizer
    optimizer = set_optimizer(opt, model)
    #scheduler
    if slr != None:
        print("Setting scheduler")
        scheduler = set_scheduler(slr, optimizer)
    
    #Initial lr
    init_lr = 0.01
    curr_lr = 0.01

    # Initial Setup
    min_acc = 0
    t1 = time.time()
    fw0 = closure_diffusion(dts_train, model, criterion, device)
    t2 = time.time()

    print(f"Time elapsed: {t2 - t1}")

    time_compute_fw0 = t2 - t1  # To be added to the elapsed time in case we are using CMA Light (information used)
    initial_val_loss = closure_diffusion(dts_test, model, criterion, device)
    train_accuracy = accuracy_diffusion(dts_train, model, device)
    val_acc = accuracy_diffusion(dts_test, model, device)
    f_tilde = fw0
    t3 = time.time()

    print(f"Time elapsed 2: {t3 - t2}")

    if opt == 'cmal':
        optimizer.set_f_tilde(f_tilde)
        optimizer.set_phi(f_tilde)
        optimizer.set_fw0(fw0)

    history = {'train_loss': [fw0], 'val_loss': [initial_val_loss], 'train_acc': [train_accuracy],
               'val_acc': [val_acc], 'step_size':[],
               'time_4_epoch': [0.0], 'nfev': 1, 'accepted': [], 'Exit': [], 'comments': [],
               'elapsed_time_noVAL': [0.0], 'f_tilde': []}
    
    # Train
    for epoch in range(ep):
        start_time = time.time()
        model.train()
        f_tilde = 0
        if opt == 'cmal':
            w_before = get_w(model)

        if opt == 'sgd':
            w_before = get_w(model)

        
        with tqdm(dts_train, unit="step", position=0, leave=True) as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{ep} - Training")
                x = batch[0].to(device)
                batch_size = x.shape[0]
                
                optimizer.zero_grad()

                t = torch.randint(0, T, (batch_size,), device=device).long()

                x_noisy, noise = forward_diffusion_sample(x, t, device)
                noise_pred = model(x_noisy, t)

                loss = criterion(noise_pred, noise)
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
            model, history, f_after, exit = optimizer.control_step(model, w_before, closure_diffusion, dts_train, device, criterion, history, epoch)
            optimizer.set_phi(min(f_tilde, f_after))
        else:
            f_after = f_tilde
        

        
        elapsed_time_4_epoch_noVAL = time.time() - start_time

        # Validation
        model.eval()
        val_loss = closure_diffusion(dts_test, model, criterion, device)
        val_acc = accuracy_diffusion(dts_test, model, device)
        train_accuracy = accuracy_diffusion(dts_train, model, device)

        elapsed_time_4_epoch = time.time() - start_time

        #variant with train_acc and not val_acc
        # max_train_acc = max(history['train_acc'])
        # if doLinesearch == True and opt == 'sgd' and train_accuracy - max_train_acc < 0.01:
        # if doLinesearch == True and opt == 'sgd' and val_acc - min_acc < 0.01:
        if doLinesearch == True and opt == 'sgd':
            model.eval()

            if epoch == 0:
                f_prev = fw0
            else:
                f_prev = history['f_tilde'][epoch - 1]

            if f_tilde >= f_prev:
                print(f"f_tilde decreased, f_tilde: {f_tilde:.5f}, f_prev: {f_prev:.5f}")
                

                print("linesearch")

                if epoch == 0:
                    f_start = fw0
                else:
                    # f_start = history['f_tilde'][epoch - 1]
                    sample_model = copy.deepcopy(model)
                    with torch.no_grad():
                        set_w(sample_model, w_before)
                        f_start = closure_diffusion(dts_train, sample_model, criterion, device)


                # Create a DataLoader for the entire dataset

                # Generate random indices for the subset
                random_indices = np.random.choice(len(dts_train.dataset), size=int(len(dts_train.dataset)*0.5), replace=False)

                # Create a SubsetRandomSampler using these indices
                subset_sampler = SubsetRandomSampler(random_indices)

                # Create a DataLoader for the subset
                subset_loader = DataLoader(dts_train.dataset, batch_size=16, sampler=subset_sampler)


                w_after = get_w(model)
                direction = ((w_after - w_before) / curr_lr)

                gradient_dir = - torch.dot(direction, direction)

                if curr_lr * 2 > init_lr:
                    alfa = init_lr
                elif curr_lr * 2 > 1:
                    alfa = 1
                else:
                    alfa = curr_lr * 2

                
                alfa, f_alfa, n_func_eval = armijoDecreasingZeta(f_start, criterion, w_before, gamma, direction, gradient_dir, model, dts_train, closure_diffusion, alfa=alfa, device=device)
                print(f"linesearch ended, f_start: {f_start}, f_end: {f_alfa}")
                
                if f_alfa < f_start:
                    curr_lr = alfa
                    set_w(model, w_before)
                    set_lr(optimizer, alfa)
                print(f"alfa: {alfa}, n_func_eval: {n_func_eval}")

        # scheduler step
        if slr != None:
            if slr == "ReduceLROnPlateau":
                #TODO: Adapt to get both accuracy and loss based on mode 'min', 'max'
                scheduler.step(val_loss)
            else:
                scheduler.step()

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
            torch.save(model, sm_root + 'train_' + opt + '_' + ds + '_' + net_name + '_' + history_ID + '_model_best.pth')
            min_acc = val_acc
            print('\n - New best Val-ACC: {:.3f} at epoch {} - \n'.format(min_acc, epoch + 1))

        torch.save(history, sm_root + 'history_' + opt + '_' + ds + '_' + net_name + '_' + history_ID + '.txt')

    print('\n - Finished Training - \n')
    torch.save(history, sm_root + 'history_' + opt + '_' + ds + '_' + net_name + '_' + history_ID + '.txt')
    return history


if __name__ == '__main__':

    # Setup in cmd line
    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', type=int, default=13)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=float, default=12345)

    # parser.add_argument('--network', type=str, required=True)
    # parser.add_argument('--opt', type=str, required=True)
    parser.add_argument('--network', type=str, default='unet')
    parser.add_argument('--opt', type=str, default='sgd')

    # parser.add_argument('--scheduler', type=str, default='StepLR')
    parser.add_argument('--scheduler', type=str, default=None)

    parser.add_argument('--dts', type=str, default='mnist')
    parser.add_argument('--dts_root', type=str, default='/work/datasets/')
    parser.add_argument('--trial', type=str, default='30_dts20per_ls_nomnes_pt2')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
 
    hardware_check()

    dts_root = args.dts_root
    # dts_root = '/work/datasets/'
    bs=16
    nw=8

    if args.dts == 'cifar10': # Classification
        transform = torchvision.transforms.Compose([
                                                # torchvision.transforms.RandomHorizontalFlip(),
                                                # torchvision.transforms.RandomRotation(10),
                                                # torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                                # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=dts_root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=dts_root, train=False, download=True, transform=transform)
        num_classes = len(trainset.classes)
    
    elif args.dts == 'cifar100': # Classification
        transform = torchvision.transforms.Compose([
                                                # torchvision.transforms.RandomHorizontalFlip(),
                                                # torchvision.transforms.RandomRotation(10),
                                                # torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                                # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR100(root=dts_root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=dts_root, train=False, download=True, transform=transform)
        num_classes = len(trainset.classes)
    
    elif args.dts == 'mnist': 
        class GrayscaleToRGB(object):
            def __call__(self, img):
                return img.repeat(3, 1, 1)

        # Definisci le trasformazioni per i dati
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            # GrayscaleToRGB(), # transform in 3 channels
            # transforms.Normalize((0.5,), (0.5,))
        ])

        trainset = torchvision.datasets.MNIST(root=dts_root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=dts_root, train=False, download=True, transform=transform)
        num_classes = len(trainset.classes)



    # ==================== DATASET PRINTS ====================
    print(len(trainset))
    print(len(testset))
    
    trainset = Subset(trainset, range(int(len(trainset)*0.30)))
    testset = Subset(testset, range(int(len(trainset)*0.30)))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True) # Togliere random reshuffle --> shuffle=False
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)

    sample = next(iter(trainloader))
    print(len(sample))
    print(sample[0].shape)
    print(sample[1].shape)



    # ==================== DATALOADER AND TRAINING ========================

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True) # Togliere random reshuffle --> shuffle=False
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)
    history = train_model(sm_root='/work/results/models_nonpretrained/', 
                          opt=args.opt,
                          slr=args.scheduler, 
                          ep=args.ep, 
                          ds=args.dts, 
                          net_name=args.network, 
                          n_class=num_classes, 
                          history_ID=args.trial, 
                          doLinesearch=True,
                          dts_train=trainloader, 
                          dts_test=testloader,
                          verbose_train=False,
                          checkpoint='/work/results/models_nonpretrained/train_sgd_mnist_unet_ep_30_dts20per_ls_mnes_model_best.pth'
                        #   checkpoint=None
                          )





    # ==================== PRINT HISTORY =========================

    # test_histroy = torch.load(r"/work/results/models_nonpretrained/history_adam_cifar10_unet_l2loss.txt")
    # print(test_histroy)
    # with open('adam.json', 'w') as f:
    #     json.dump(test_histroy, f)

    # test_histroy = torch.load(r"/work/results/models_nonpretrained/history_cmal_cifar10_unet_l2loss.txt")
    # print(test_histroy)
    # with open('cmal.json', 'w') as f:
    #     json.dump(test_histroy, f)
    
    # test_histroy = torch.load(r"/work/results/models_nonpretrained/history_sgd_cifar10_unet_l2loss.txt")
    # print(test_histroy)
    # with open('sgd.json', 'w') as f:
    #     json.dump(test_histroy, f)

    # test_histroy = torch.load(r"/work/results/models_nonpretrained/history_adam_cifar10_unet_prova.txt")
    # print(test_histroy)
    # with open('adam_l1loss.json', 'w') as f:
    #     json.dump(test_histroy, f)

    # test_histroy = torch.load(r"/work/results/models_nonpretrained/history_cmal_cifar10_unet_prova.txt")
    # print(test_histroy)
    # with open('cmal_l1loss.json', 'w') as f:
        # json.dump(test_histroy, f)


    # test_histroy = torch.load(r"/work/results/models_nonpretrained/history_adam_cifar10_unet_l2loss_epoch_30.txt")
    # print(test_histroy)
    # with open('adam.json', 'w') as f:
    #     json.dump(test_histroy, f)

    # test_histroy = torch.load(r"/work/results/models_nonpretrained/history_cmal_cifar10_unet_l2loss_epoch_30.txt")
    # print(test_histroy)
    # with open('cmal.json', 'w') as f:
    #     json.dump(test_histroy, f)

    # test_histroy = torch.load(r"/work/results/models_nonpretrained/history_sgd_cifar10_unet_l2loss_epoch_30_v2.txt")
    # print(test_histroy)
    # with open('sgd.json', 'w') as f:
    #     json.dump(test_histroy, f)

    # test_histroy = torch.load(r"/work/results/models_nonpretrained/history_adadelta_cifar10_unet_l2loss_epoch_30.txt")
    # print(test_histroy)
    # with open('adadelta.json', 'w') as f:
    #     json.dump(test_histroy, f)

    # test_histroy = torch.load(r"/work/results/models_nonpretrained/history_adamax_cifar10_unet_l2loss_epoch_30.txt")
    # print(test_histroy)
    # with open('adamax.json', 'w') as f:
    #     json.dump(test_histroy, f)

    

    # test_histroy = torch.load(r"/work/results/models_nonpretrained/history_sgd_mnist_unet_ep_50_ls_full_mom_nes_tot_f_dec.txt")
    # print(test_histroy)
    # with open('50_ls_full_mom_nes_tot_f_dec.json', 'w') as f:
    #     json.dump(test_histroy, f)

    # test_histroy = torch.load(r"/work/results/models_nonpretrained/history_sgd_mnist_unet_30_dts20per_nols_mnes_pt2.txt")
    # print(test_histroy)
    # with open('30_dts20per_nols_mnes_pt2.json', 'w') as f:
    #     json.dump(test_histroy, f)

