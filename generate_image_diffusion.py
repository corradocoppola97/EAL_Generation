import time, os , argparse
import torch
import torchvision
from torch.utils.data import Subset
from warnings import filterwarnings
from tqdm import tqdm
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import numpy as np


from utils.utils import closure, closure_diffusion, count_parameters, set_optimizer, set_scheduler, accuracy, accuracy_diffusion, hardware_check
from utils.utils_diffusion_model import T, sample_plot_image
from optimizers.cmalight import get_w
from networks.network import get_pretrained_net, get_diffusion_model
from networks.diffusionModel import forward_diffusion_sample


filterwarnings('ignore')

if __name__ == '__main__':
    # Setup in cmd line
    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', type=int, default=30)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=float, default=12345)

    # parser.add_argument('--network', type=str, required=True)
    # parser.add_argument('--opt', type=str, required=True)
    parser.add_argument('--network', type=str, default='unet')
    parser.add_argument('--opt', type=str, default='adamax')

    # parser.add_argument('--scheduler', type=str, default='StepLR')
    parser.add_argument('--scheduler', type=str, default=None)

    parser.add_argument('--dts', type=str, default='cifar10')
    parser.add_argument('--trial', type=str, default='l2loss_epoch_30')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    dts_root = '/work/datasets/'
    bs=4
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True) # Togliere random reshuffle --> shuffle=False
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)
    
    class arg:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        IMG_SIZE = 32


    model = torch.load('/work/results/models_nonpretrained/train_adamax_cifar10_unet_l2loss_epoch_30_model_best.pth')
    model.eval()
    

    # sample_plot_image(model, arg)



    def imshow(img, path=None):
        img = img / 2 + 0.5  # denormalizza
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if path != None:
            plt.savefig(path)
        else:
            plt.show()

    # # Ottieni alcune immagini di esempio
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # # Mostra le immagini
    imshow(torchvision.utils.make_grid(images))

    t = torch.randint(T-1, T, (bs,), device=arg.device).long()
    # print(t)

    x_noisy, noise = forward_diffusion_sample(images, t, arg.device)

    print(x_noisy.shape)
    imshow(torchvision.utils.make_grid(x_noisy.detach().cpu()))


    sample_plot_image(model, arg, x_noisy = x_noisy[1].unsqueeze(0))
