import math, torch, os
import torch.nn as nn
import matplotlib.pyplot as plt
# import pytorch_ssim as pytorch_ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure
import cv2
import numpy as np

def normalize(yn, X, bs):
    maxVal, _ = torch.max(torch.abs(yn.reshape(bs, -1)), dim=1)
    # maxVal[maxVal < 0.1] = 1
    if len(X.shape) == 3:
        return maxVal, yn / maxVal[:, None, None, None], X / maxVal[:, None, None]
    return maxVal, yn / maxVal[:, None, None, None], X / maxVal[:, None, None, None]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_psnr(X, Y):
    criteria = nn.MSELoss()
    return 20 * math.log10(1 / math.sqrt(criteria(X, Y)))


def PSNR1chan(Xk, X):  # ONLY the REAL Part
    bs, C, W, H = X.shape
    Xk = Xk[:, 0, :, :]
    X = X[:, 0, :, :]
    mse = torch.sum(((Xk - X) ** 2).reshape(bs, -1), dim=1) / (W * H)
    return 20 * torch.log10(torch.max(torch.max(X, dim=1)[0], dim=1)[0] / torch.sqrt(mse))


def compute_metrics1chan(Xk, X, X0):
    init_psnr, recon_psnr = PSNR1chan(X0, X), PSNR1chan(Xk, X)
    bs = X.shape[0]
    avg_init_psnr = torch.sum(init_psnr) / bs
    avg_recon_psnr = torch.sum(recon_psnr) / bs
    avg_delta_psnr = torch.sum(recon_psnr - init_psnr) / bs

    # Xk = torch.clamp(torch.abs(torch.view_as_complex(Xk.permute(0, 2, 3, 1).contiguous())), min=0, max=1)
    # X = X[:, 0:1, :, :]
    # avg_ssim = pytorch_ssim.SSIM(Xk, X)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    avg_ssim = ssim(Xk.cpu(), X.cpu())
    return avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim


def PSNR3chan(Xk, X):
    bs, C, W, H = X.shape
    mse = torch.sum(((Xk - X) ** 2).reshape(bs, -1), dim=1) / (C * W * H)
    return 20 * torch.log10(1 / torch.sqrt(mse))


def compute_metrics3chan(Xk, X, X0):
    init_psnr, recon_psnr = PSNR3chan(X0, X), PSNR3chan(Xk, X)
    bs = X.shape[0]
    avg_init_psnr = torch.sum(init_psnr) / bs
    avg_recon_psnr = torch.sum(recon_psnr) / bs
    avg_delta_psnr = torch.sum(recon_psnr - init_psnr) / bs
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    avg_ssim = ssim(Xk.cpu(), X.cpu())
    return avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim


def getUCIQE(img):
    # RGB in range 0~1
    img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_LAB = np.array(img_LAB, dtype=np.float64)
    # Trained coefficients are c1=0.4680, c2=0.2745, c3=0.2576 according to paper.
    coe_Metric = [0.4680, 0.2745, 0.2576]

    img_lum = img_LAB[:, :, 0] / 255.0
    img_a = img_LAB[:, :, 1] / 255.0
    img_b = img_LAB[:, :, 2] / 255.0

    # item-1
    chroma = np.sqrt(np.square(img_a) + np.square(img_b))
    sigma_c = np.std(chroma)

    # item-2
    img_lum = img_lum.flatten()
    sorted_index = np.argsort(img_lum)
    top_index = sorted_index[-1] # sorted_index[int(len(img_lum) * 0.99)]
    bottom_index = sorted_index[0] # sorted_index[int(len(img_lum) * 0.01)]
    con_lum = img_lum[top_index] - img_lum[bottom_index]

    # item-3
    chroma = chroma.flatten()
    sat = np.divide(chroma, img_lum, out=np.zeros_like(chroma, dtype=np.float64), where=img_lum != 0)
    avg_sat = np.mean(sat)

    uciqe = sigma_c * coe_Metric[0] + con_lum * coe_Metric[1] + avg_sat * coe_Metric[2]
    return uciqe * 2


def plot_deq_residual(deq, args, epoch=0):
    """ evaluate intermediate deq results """
    plt.figure(figsize=(7, 3))
    plt.semilogy(deq.forward_res)
    plt.xlabel('DEQ iterations')
    plt.ylabel('log(residual)')
    plt.title('DEQ residual plot')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, f'deq_res_{epoch}.png'))
    plt.close()


def plot_deq_mse(deq, A0, Z0, y, X0, X, args, epoch=0):
    """ evaluate intermediate deq results """
    deq.invBlock.init_setup(A0, Z0)
    # xk, xk_res_list = deq.solver(lambda xk: deq.invBlock(xk, y), torch.zeros_like(X0), **deq.kwargs)
    criteria = nn.MSELoss()
    MSE_list = []

    xk = torch.clone(X0)
    # xk = torch.zeros_like(X0)
    for i in range(50):
        xk = deq.invBlock(xk, y) + X0
        mse = criteria(xk, X).item()
        MSE_list.append(mse)

    # for i in range(len(xk_res_list)):
    #     mse = criteria(xk_res_list[i] + X0, X).item()
    #     MSE_list.append(mse)

    plt.figure(figsize=(7, 3))
    plt.semilogy(MSE_list)
    # plt.xlabel('DEQ iterations')
    # plt.ylabel('log(residual)')
    # plt.title('DEQ residual plot')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, f'deq_res_{epoch}.png'))
    plt.close()


def plot_reconstruction(X, X0, Xk, args, epoch, algo="A-Adaptive_DEQ"):
    plt.figure(figsize=(6, 6))
    plt.suptitle('Underwater Image ' + algo)
    Xk = torch.clamp(Xk, 0, 1).detach().cpu()
    X = torch.clamp(X, 0, 1).detach().cpu()
    X0 = torch.clamp(X0, 0, 1).detach().cpu()
    index = 0
    nsubplots = min(3, Xk.shape[0])
    for i in range(nsubplots):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X0[index + i].permute(1, 2, 0))
        plt.title('$x_0$')
        plt.axis('off')
        plt.subplot(3, 3, i + 4)
        plt.imshow(Xk[index + i].permute(1, 2, 0))
        plt.title('$\hat{x}$')
        plt.axis('off')
        plt.subplot(3, 3, i + 7)
        plt.imshow(X[index + i].permute(1, 2, 0))
        plt.title('$x$')
        plt.axis('off')
    plt.tight_layout()

    if args.train:
        plt.savefig(os.path.join(args.save_path, f'{epoch}_result.png'))
    else:
        plt.savefig(os.path.join(args.save_path, f'test_{epoch}_result.png'))
    plt.close()


def add_detail_to_x(X, args):
    if args.dataset == 'MRI':
        Xg = torch.clone(X)
        row, col = [150, 150]
        length = 20
        patch = Xg[:, :, row:row + 1, col:col + length]
        g = torch.ones_like(patch) * 0.7
        Xg[:, :, row:row + 1, col:col + length] = torch.max(g, patch)
        return Xg
    if args.dataset == 'seis':
        Xg = torch.clone(X)
        row, col = [175, 150]
        length = 10
        patch = Xg[:, :, row:row + 1, col:col + length]
        g = - torch.ones_like(patch) * 0.3  # magnitude of 0.1 and 0.3
        Xg[:, :, row:row + 1, col:col + length] = g  # torch.max(g, patch)
        return Xg
    else:
        Xg = torch.clone(X)
        # row, col = np.random.randint(0, 24, [2])
        row, col = [5, 5]
        patch = Xg[:, :, row:row + 1, col:col + 3]
        g = torch.ones_like(patch) * 0.7
        # g *= args.epsilon / mode.norms_3D(g)
        Xg[:, :, row:row + 1, col:col + 3] = torch.max(g, patch)
        # torch.max(patch + torch.ones_like(patch), torch.ones_like(patch))
        return Xg



def plot_1_channel(X, X0, Xk, args, epoch, algo="A-Adaptive_DEQ"):
    plt.figure(figsize=(6, 6))
    plt.suptitle('MNIST' + algo)
    Xk = torch.clamp(Xk, 0, 1).detach().cpu().squeeze()
    X = torch.clamp(X, 0, 1).detach().cpu().squeeze()
    X0 = torch.clamp(X0, 0, 1).detach().cpu().squeeze()
    index = 0
    for i in range(min(3, len(X))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X0[index + i])
        plt.title('$x_0$')
        plt.axis('off')
        plt.subplot(3, 3, i + 4)
        plt.imshow(Xk[index + i])
        plt.title('$\hat{x}$')
        plt.axis('off')
        plt.subplot(3, 3, i + 7)
        plt.imshow(X[index + i])
        plt.title('$x$')
        plt.axis('off')
    plt.tight_layout()

    if args.train:
        plt.savefig(os.path.join(args.save_path, f'{epoch}_result.png'))
    else:
        plt.savefig(os.path.join(args.save_path, f'test_{epoch}_result.png'))
    plt.close()

def plot_latent_repr(z, args):
    plt.figure(figsize=(5, 7))
    plt.suptitle('latent representation')
    index = 0
    nsubplots = min(3, z.shape[0])
    for i in range(nsubplots):
        plt.subplot(3, 1, i + 1)
        plt.plot(z[index + i].view(-1).detach().cpu())
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, f'latent.png'))
    plt.close()


def plot_origsize(X, X0, Xk, args, epoch, algo="A-Adaptive_DEQ"):
    plt.figure(figsize=(8, 3))
    plt.suptitle('Underwater Image ' + algo)
    Xk = torch.clamp(back2origsize(Xk), 0, 1).detach().cpu()
    X = torch.clamp(back2origsize(X), 0, 1).detach().cpu()
    X0 = torch.clamp(back2origsize(X0), 0, 1).detach().cpu()

    plt.subplot(1, 3, 1)
    plt.imshow(X0.permute(1, 2, 0))
    plt.title('$x_0$')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(Xk.permute(1, 2, 0))
    plt.title('$\hat{x}$')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(X.permute(1, 2, 0))
    plt.title('$x$')
    plt.axis('off')
    plt.tight_layout()

    if args.train:
        plt.savefig(os.path.join(args.save_path, f'origsize_{epoch}_result.png'))
    else:
        plt.savefig(os.path.join(args.save_path, f'origsize_ts_{epoch}_result.png'))
    plt.close()

def back2origsize(X):
    bs, C, H, W = X.shape
    if bs != 16:
        print('Invalid number of patches')
        return None
    img = torch.zeros(C, H * 4, W * 4)
    for idx in range(16):
        i, j = idx // 4, idx % 4
        img[:, H * i: H * (i+1), W * j: W * (j+1)] = X[idx]
    return img