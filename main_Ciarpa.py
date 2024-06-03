import time, os , argparse
from utils import closure, closure_diffusion, count_parameters, set_optimizer, accuracy, accuracy_diffusion, hardware_check
from cmalight import get_w
from network import get_pretrained_net, get_diffusion_model
import torch
import torchvision
from torch.utils.data import Subset
from warnings import filterwarnings
from tqdm import tqdm
import torch.nn.functional as F
from utils_diffusion_model import T
from diffusionModel import forward_diffusion_sample


filterwarnings('ignore')


def train_model(sm_root: str,
                opt: str,
                ep: int,
                ds: str,
                net_name: str,
                n_class: int,
                history_ID: str,
                dts_train: torch.utils.data.DataLoader,
                dts_test: torch.utils.data.DataLoader,
                verbose_train: bool) -> dict:
    
    print('\n ------- Begin training process ------- \n')

    # Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # Model
    model = get_diffusion_model(num_classes=n_class).to(device)
    print('\n The model has: {} trainable parameters'.format(count_parameters(model)))
    # Loss
    criterion = F.mse_loss
    # Optimizer
    optimizer = set_optimizer(opt, model)
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
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=float, default=12345)
    # parser.add_argument('--network', type=str, required=True)
    # parser.add_argument('--opt', type=str, required=True)
    parser.add_argument('--network', type=str, default='unet')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--dts', type=str, default='cifar10')
    parser.add_argument('--trial', type=str, default='l2loss')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
 
    hardware_check()

    dts_root = '/work/datasets/'
    bs=8
    nw=8

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
    
    # print(len(trainset))
    # print(len(testset))
    
    # trainset = Subset(trainset, range(bs*4))
    # testset = Subset(testset, range(bs*4))

    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, pin_memory=True, num_workers=nw) # Togliere random reshuffle --> shuffle=False
    # testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, pin_memory=True, num_workers=nw)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True) # Togliere random reshuffle --> shuffle=False
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)
    history = train_model(sm_root='/work/results/models_nonpretrained/', 
                          opt=args.opt, 
                          ep=args.ep, 
                          ds=args.dts, 
                          net_name=args.network, 
                          n_class=num_classes, 
                          history_ID=args.trial, 
                          dts_train=trainloader, 
                          dts_test=testloader,
                          verbose_train=False)

    # sample = next(iter(trainloader))
    # print(len(sample))
    # print(sample[0].shape)
    # print(sample[1].shape)
    test_histroy = torch.load(r"/work/results/models_nonpretrained/history_adam_cifar10_unet_l2loss.txt")
    print(test_histroy)
    test_histroy = torch.load(r"/work/results/models_nonpretrained/history_cmal_cifar10_unet_l2loss.txt")
    print(test_histroy)

    # test_histroy = torch.load(r"/work/results/models_nonpretrained/train_adam_cifar10_resnet18_model_best.pth")
    