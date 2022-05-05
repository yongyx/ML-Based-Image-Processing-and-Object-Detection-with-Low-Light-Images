import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torchvision.transforms.functional import equalize, gaussian_blur, rgb_to_grayscale
from torchvision.transforms import GaussianBlur
from torch import nn


def grad(x):
    '''
    Gradient operator for x as inidicated in code for https://arxiv.org/abs/2103.00832.

    :param x: tensor input.
    :type x: torch.Tensor of size (N, C, H, W).

    :returns: gradient of x.
    '''
    C = x.shape[1]
    if C != 1:
        x = rgb_to_grayscale(x).type(torch.float32)

    kernel1 = np.transpose(np.array([[0, 0], [-1, 1]], dtype=np.float32).reshape((2, 2, 1, 1)), [3, 2, 0, 1])
    kernel2 = np.transpose(np.array([[0, -1], [0, 1]], dtype=np.float32).reshape((2, 2, 1, 1)), [3, 2, 0, 1])

    conv1 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding='same', bias=False)
    conv1.weight = torch.nn.Parameter(torch.from_numpy(kernel1))
    conv2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding='same', bias=False)
    conv2.weight = torch.nn.Parameter(torch.from_numpy(kernel2))

    out1 = conv1(x)
    out2 = conv2(x)
    return out1 + out2


def ICE_loss(S, R, I):
    '''
      Calculates ICE-Net loss as indicated in https://arxiv.org/abs/2103.00832.

      :param S: input low-light image.
      :type S: torch.Tensor of size (N, 3, H, W).
      :param R: reflectance output of ICE-Net - this is the noisy brightened image.
      :type R: torch.Tensor of size (N, 3, H, W).
      :param I: illumination  output of ICE-Net.
      :type I: torch.Tensor of size (N, 1, H, W).

      :returns: loss for ICE-Net.
    '''

    # Lambda hyperparam valuees as indicated in paper.
    l1, l2, l3 = 0.1, 0.01, 0.1

    # Extend 1-channel I to RGB image.
    I_full = I.repeat(1, 3, 1, 1)

    R_max, _ = torch.max(R, 1, keepdims=True)  # Get max channel of image.
    S_max, _ = torch.max(S, 1, keepdims=True)
    S_hist = equalize(S_max)  # Implements histogram equalization.
    R_grad = grad(R)

    I_grad = grad(I)

    L_rcon = torch.mean(torch.abs(S - R * I_full))  # L1 norm = (1/N) * sum(abs(x))
    L_r = l1 * torch.mean(torch.abs(R_max - S_hist)) + l2 * torch.mean(torch.abs(R_grad))
    L_i = l3 * torch.mean(torch.abs(I_grad * torch.exp(-10 * torch.abs(R_grad))))

    iceloss = L_rcon + L_r + L_i
    return iceloss


def RED_loss(S, R, I, R_old, ln1=3, ln2=5, lambdas=None):
    '''
      Calculates RED-Net loss as indicated in https://arxiv.org/abs/2103.00832.

      :param S: input low-light image.
      :type S: torch.Tensor of size (N, 3, H, W).
      :param R: reflectance output of RED-Net - this is the de-noised brightened image.
      :type R: torch.Tensor of size (N, 3, H, W).
      :param I: illumination  output of RED-Net.
      :type I: torch.Tensor of size (N, 1, H, W).
      :param R_old: reflectance output of ICE-Net - this is the noisy brightened image.
      :type R_old: torch.Tensor of size (N, 3, H, W).

      :returns: loss for RED-Net.
    '''
    # lambda hyperparam valuees as indicated in paper.
    l1, l2, l3, l4 = 0.005, 0.01, 0.05, 0.1 if lambdas is None else lambdas

    def local_norm(x):
        '''
          Computes local normalization on x as indicated in http://bigwww.epfl.ch/sage/soft/localnormalization/.

          :param x: tensor input.
          :type x: torch.Tensor of size (N, C, H, W).

          :returns: gradient of x.
        '''
        u = gaussian_blur(x, 3)
        var = gaussian_blur(torch.square(x), 5)
        std = torch.sqrt(var)
        err = torch.tensor(1e-6)
        return torch.div(x - u, std + err)

    def W_(R):
        smooth = gaussian_blur(R, 3)
        smooth_grad = grad(smooth)
        return local_norm(torch.abs(smooth_grad))

    def W_R_(R):
        R_grad = grad(R)
        smooth = gaussian_blur(R_grad, 3)
        return local_norm(torch.abs(smooth))

    def M_(R_grad, S_grad, W):
        return W * local_norm(torch.abs(R_grad)) - W * local_norm(torch.abs(S_grad)) * torch.log(
            torch.abs(W * local_norm(torch.abs(R_grad))))  # Added torch.abs() to gauruntee positive input.

    I_full = I.repeat(1, 3, 1, 1)
    R_grad = grad(R)
    S_grad = grad(S)
    W = W_(R)
    M = M_(R_grad, S_grad, W)

    R_max, _ = torch.max(R, 1, keepdims=True)
    R_old_max, _ = torch.max(R_old, 1, keepdims=True)
    term_r1 = R_max - R_old_max * torch.log(R_max)
    term_r2 = W * local_norm(torch.abs(R_grad)) * torch.exp(-10 * W * local_norm(torch.abs(R_grad)))

    I_grad = grad(I)
    W_R = W_R_(R)
    term_i = local_norm(torch.abs(I_grad)) * torch.exp(-10 * local_norm(torch.abs(I_grad))) * torch.exp(
        -10 * W_R * local_norm(torch.abs(R_grad)))

    L_rcon = torch.mean(torch.abs(R * I_full - S * torch.log(torch.abs(R * I_full)))) + l1 * torch.mean(torch.abs(M))
    L_r = l2 * torch.mean(torch.abs(term_r1)) + l3 * torch.mean(torch.abs(term_r2))
    L_i = l4 * torch.mean(torch.abs(term_i))

    red_loss = L_rcon + L_r + L_i
    return red_loss
