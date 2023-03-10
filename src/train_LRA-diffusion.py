import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils.ema import EMA
import numpy as np
import random
from utils.clip_wrapper import clip_img_wrap
import torch
import torchvision
import torchvision.transforms as transforms
from utils.data_utils import Custom_dataset
from utils.model_SimCLR import SimCLR_encoder
import torch.optim as optim
from utils.learning import cast_label_to_one_hot_and_prototype, adjust_learning_rate
from model_diffusion import Diffusion
from utils.knn_utils import sample_knn_labels
import argparse
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    maxk = min(max(topk), output.size()[1])

    output = torch.softmax(-(output - 1)**2,  dim=-1)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def prepare_fp_x(fp_encoder, dataset, feature_dim=10):
    fp_encoder.eval()
    data_loader = data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
    prior = torch.zeros([len(dataset), feature_dim]).to(device)
    with tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Computing feature fp(x)', ncols=100) as pbar:
        for i, data_batch in pbar:
            [x_batch, _, data_indecies] = data_batch[:3]
            temp = fp_encoder(x_batch.to(device)).detach()
            prior[data_indecies, :] = temp
    return prior


def train(diffusion_model, train_dataset, val_dataset, test_dataset, model_save_dir,
          n_epochs=1000, batch_size=256, k=10, real_fp=True, warmup_epochs=20):
    device = diffusion_model.device
    n_class = diffusion_model.n_class

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

    fp_embd = prepare_fp_x(diffusion_model.fp_encoder, train_dataset, feature_dim=diffusion_model.fp_dim).to(device)

    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')

    # acc_diff = test(diffusion_model, test_loader)

    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model.model)

    max_accuracy = 0.0
    print('Diffusion training start')
    for epoch in range(n_epochs):
        diffusion_model.model.train()

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, data_batch in pbar:
                [x_batch, y_batch, data_indices] = data_batch[:3]

                if real_fp:
                    # compute embeddings for augmented images for better performance
                    y_prior = diffusion_model.fp_encoder(x_batch.to(device))
                else:
                    # use pre-compute embedding for efficiency
                    y_prior = fp_embd[data_indices, :]

                # sample a knn labels and compute weight for the sample
                y_labels_batch, sample_weight = sample_knn_labels(y_prior, y_batch.to(device), fp_embd,
                                                                  torch.tensor(train_dataset.targets).to(device),
                                                                  k=k, n_class=n_class, weighted=True)

                # convert label to one-hot vector
                y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch.to(torch.int64),
                                                                                      n_class=n_class)
                y_0_batch = y_one_hot_batch.to(device)

                # adjust_learning_rate
                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=1000, lr_input=0.001)
                n = x_batch.size(0)

                # sampling t
                t = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(n // 2 + 1,)).to(device)
                t = torch.cat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

                # train with and without prior
                output, e = diffusion_model.forward_t(y_0_batch, x_batch, t, y_prior)

                # compute loss
                mse_loss = diffusion_loss(e, output)
                weighted_mse_loss = torch.matmul(sample_weight, mse_loss)
                loss = torch.mean(weighted_mse_loss)
                pbar.set_postfix({'loss': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(diffusion_model.model)

        if epoch % 5 == 0 and epoch >= warmup_epochs:
            val_acc = test(diffusion_model, val_loader)
            print(f"epoch: {epoch}, validation accuracy: {val_acc:.2f}%")
            if val_acc > max_accuracy:
                # save diffusion model
                print('Improved! evaluate on testing set...')
                test_acc = test(diffusion_model, test_loader)
                # states = [diffusion_model.model.state_dict(), diffusion_model.fp_encoder.state_dict()]
                # torch.save(states, model_save_dir)
                print(f"Model saved, update best accuracy at Epoch {epoch}, val acc: {val_acc}, test acc: {test_acc}")
                max_accuracy = max(max_accuracy, val_acc)


def test(diffusion_model, test_loader):

    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        acc_avg = 0.
        for test_batch_idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Doing DDIM...', ncols=100):
            [images, target, _] = data_batch[:3]
            target = target.to(device)
            with torch.no_grad():
                label_t_0 = diffusion_model.reverse_ddim(images, stochastic=False, fq_x=None).detach().cpu()
                acc_temp = accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
                acc_avg += acc_temp

        acc_avg /= len(test_loader)

    return acc_avg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_type', default='cifar10-1-0.35', help='noise label file', type=str)
    parser.add_argument("--nepoch", default=1000, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=200, help="batch_size", type=int)
    parser.add_argument("--device", default='cpu', help="which GPU to use", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=20, help="warmup_epochs", type=int)
    parser.add_argument("--feature_dim", default=512, help="feature_dim", type=int)
    parser.add_argument("--k", default=10, help="k neighbors for knn", type=int)
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--fp_encoder", default='SimCLR', help="which encoder for fp (SimCLR or CLIP)", type=str)
    parser.add_argument("--diff_encoder", default='resnet34', help="which encoder for diffusion (linear, resnet18, 34, 50...)", type=str)
    args = parser.parse_args()

    # print device
    device = args.device
    print('Using device:', device)

    dataset = args.noise_type.split('-')[0]
    # load datasets
    if dataset == 'cifar10':
        n_class = 10
        train_dataset_cifar = torchvision.datasets.CIFAR10(root='../', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR10(root='../', train=False, download=True)
    elif dataset == 'cifar100':
        n_class = 100
        train_dataset_cifar = torchvision.datasets.CIFAR100(root='../', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR100(root='../', train=False, download=True)
    else:
        raise Exception("Date should be cifar10 or cifar100")

    # load fp encoder
    if args.fp_encoder == 'SimCLR':
        fp_dim = 2048
        real_fp = True
        state_dict = torch.load(f'../model/SimCLR_128_{dataset}.pt', map_location=torch.device(args.device))
        fp_encoder = SimCLR_encoder(feature_dim=128).to(args.device)
        fp_encoder.load_state_dict(state_dict, strict=False)
    elif args.fp_encoder == 'CLIP':
        real_fp = False
        fp_encoder = clip_img_wrap('ViT-L/14', args.device)
        fp_dim = fp_encoder.dim
    else:
        raise Exception("fp_encoder should be SimCLR or CLIP")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = Custom_dataset(train_dataset_cifar.data[:45000], train_dataset_cifar.targets[:45000],
                                   transform=transform_train)
    val_dataset = Custom_dataset(train_dataset_cifar.data[45000:], train_dataset_cifar.targets[45000:])
    test_dataset = Custom_dataset(test_dataset_cifar.data, test_dataset_cifar.targets)

    # load noisy label
    noise_label = np.load('../noise_label/' + args.noise_type + '.npy')
    train_dataset.update_label(noise_label)
    print('Training on noise label:', args.noise_type)

    # initialize diffusion model
    model_path = f'../model/LRA-diffusion_{args.fp_encoder}_{args.noise_type}.pt'
    diffusion_model = Diffusion(fp_encoder=fp_encoder, n_class=n_class, fp_dim=fp_dim, feature_dim=args.feature_dim,
                                device=device, encoder_type=args.diff_encoder, ddim_num_steps=args.ddim_n_step)
    # state_dict = torch.load(model_path, map_location=torch.device(device))
    # diffusion_model.load_diffusion_net(state_dict)
    diffusion_model.fp_encoder.eval()

    # train the diffusion model
    print(f'training LRA-diffusion using fp encoder: {args.fp_encoder} on: {args.noise_type}.')
    print(f'model saving dir: {model_path}')
    train(diffusion_model, train_dataset, val_dataset, test_dataset, model_path, n_epochs=args.nepoch,
          batch_size=args.batch_size, k=args.k, real_fp=real_fp, warmup_epochs=args.warmup_epochs)



