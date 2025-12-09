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

def plot_reconstruction(X, X0, Xk, args, epoch, algo="PhyNN"):
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