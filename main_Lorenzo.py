import os, time
import argparse	
from tqdm import tqdm

from utils import *
from networks.network import *
from cmalight.cmalight import *
import torch
import torchvision


def collect_data(bs, nw, dts_root):
    transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
    						torchvision.transforms.RandomRotation(10),
    						torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    						torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    						torchvision.transforms.ToTensor(), 
    						torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Normalize into range -1/1

    trainset = torchvision.datasets.CIFAR10(root=dts_root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=nw)
    testset = torchvision.datasets.CIFAR10(root=dts_root, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs,shuffle=False, num_workers=nw)

    print('\nINFO: There are {} training and {} testing samples\n'.format(trainset.__len__(), testset.__len__()))

    return trainloader, testloader, trainset, testset


def train_model(sm_root: str,
                opt: str,
                ep: int,
                ds: str,
                net_name: str,
                history_ID: str,
                dts_train: torch.utils.data.DataLoader,
                dts_test: torch.utils.data.DataLoader,
                verbose_train: bool,
                *args,
                **kwargs):
    if verbose_train: print('\n ------- Begin training process ------- \n')

    # Hardware
    device = hardware_check()
    torch.cuda.empty_cache()

    min_acc = 0

    # Model
    model = get_pretrained_net(net_name).to(device)
    if verbose_train: print('\n The model has: {} trainable parameters'.format(count_parameters(model)))

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = set_optimizer(opt,model,*args,**kwargs)


    #Initial Setup
    min_acc = 0
    t1 = time.time()
    fw0 = closure(dts_train,model, criterion, device)
    t2 = time.time()
    time_compute_fw0 = t2-t1 #To be added to the elapsed time in case we are using CMA Light (information used)
    initial_val_loss = closure(dts_test,model, criterion,device)
    train_accuracy = accuracy(dts_train,model,device)
    test_accuracy = accuracy(dts_test,model,device)
    f_tilde = fw0
    if opt == 'cmal':
        optimizer.set_f_tilde(f_tilde)
        optimizer.set_phi(f_tilde)
    history = {'train_loss': [fw0], 'val_loss': [initial_val_loss], 'train_acc': [train_accuracy], 'val_acc': [test_accuracy],
               'time_4_epoch': [0.0], 'nfev': 1, 'accepted':[], 'Exit':[], 'comments':[], 'elapsed_time_noVAL':[0.0]}

    #Train
    for epoch in range(ep):
        start_time = time.time()
        if verbose_train: print(f'Epoch {epoch+1} di {ep}')
        model.train()
        f_tilde = 0
        if opt == 'cmal':
            w_before = get_w(model)

        for x,y in dts_train:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            f_tilde += loss.item() * (len(x) / len(dts_train))
            loss.backward()
            optimizer.step()

        
        # CMAL support functions
        if opt == 'cmal':
            model, history, f_after, exit = optimizer.control_step(model,w_before,closure,
            dts_train, device, criterion, history, epoch)
            optimizer.set_phi(min(f_tilde,f_after))
        else:
            f_after = f_tilde
        elapsed_time_4_epoch_noVAL = time.time() - start_time

        # Validation
        model.eval()
        val_loss = closure(dts_test, model, criterion, device)
        test_accuracy = accuracy(dts_test, model, device)
        elapsed_time_4_epoch = time.time() - start_time

        history['train_loss'].append(f_after)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(test_accuracy)
        history['val_acc'].append(val_acc)
        history['time_4_epoch'].append(history['time_4_epoch'][-1]+elapsed_time_4_epoch)
        history['elapsed_time_noVAL'].append(history['elapsed_time_noVAL'][-1]+elapsed_time_4_epoch_noVAL)
        if epoch == 0 and opt == 'cmal':
            history['time_4_epoch'][-1] += time_compute_fw0
            history['elapsed_time_noVAL'][-1] += time_compute_fw0

        # Save data during training
        if min_acc < val_acc:
            torch.save(model, sm_root + 'model_best.pth')
            min_acc = val_acc
            if verbose_train: print('\n - New best Val-ACC: {:.3f} at epoch {} - \n'.format(min_acc, ep + 1))

        torch.save(history,'history_'+opt+'_'+ds+'_'+net_name+'_'+history_ID+'.txt')

    if verbose_train: print('\n - Finished Training - \n')
    torch.save(history, 'history_' + opt + '_' + ds + '_' + net_name + '_' + history_ID + '.txt')
    return history
    
    
    
    

def main(optim):
    # Setup in cmd line
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=float, default=1699806)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
 
    print("GPU IN USO: ", args.gpu)

    TEST_NAME = str(args.exp_name)
    
    # Path
    env_path = '/work/project/'
    dataset_root = '/work/datasets/'
    if not os.path.exists('/work/results/' + TEST_NAME + '/'+ optim + '/'):
        os.makedirs('/work/results/' + TEST_NAME + '/'+ optim + '/')
    save_path = '/work/results/' + TEST_NAME + '/' + optim + '/'
    
    print(f'\n - {save_path} - \n')
    
    # Globals
    RGB_SHAPE = (3, 32, 32)
    BATCH_SIZE = 128
    N_WORK = 8
    EPOCHS = 250
    
    # Extract data
    trainloader, testloader, _, _ = collect_data(bs=BATCH_SIZE, nw=N_WORK, dts_root=dataset_root)
    
    # Train model
    train_model(net_name=str(args.network), in_shape=RGB_SHAPE,
                sm_root=save_path, opt=optim, ep=EPOCHS,
                dts_train=trainloader, dts_test=testloader, ds='CIFAR10',
                history_ID='prova', verbose_train=True)


if __name__ == '__main__':
    main(optim='cmal')
    # main(optim='adam')
    # main(optim='adamw')
    # main(optim='adagrad')
    # main(optim='adamax')
    # main(optim='asgd')
    # main(optim='nadam')
    # main(optim='radam')
    # main(optim='rmsprop')
    # main(optim='rprop')
    # main(optim='sgd')
    
