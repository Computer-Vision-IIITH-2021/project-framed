import torch
import torch.nn as nn
import numpy as np

def get_harr_wav(in_channels, pool=True):
    
    """wavelet decomposition using conv2d
    Defining the 4 harr wavelet kernels which when 
    used for pooling results in 4 channels. Dim of the
    four kerns are al 2x2 kernel matrices.
    
    """
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0][0] = - harr_wav_H[0][0]

    harr_wav_LL = (harr_wav_L).T * harr_wav_L
    harr_wav_LH = (harr_wav_L).T * harr_wav_H
    harr_wav_HL = (harr_wav_H).T * harr_wav_L
    harr_wav_HH = (harr_wav_H).T * harr_wav_H

    #prepare the layers for the convolution
    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    #if it is pooling using wavelet transform - use convolution
    #if it is unpooling using wavelet transform - use component wise transposed convolution
    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    #define the layers such that each channel is convolved individually for each of the 4 kernels    
    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH