import os
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import time

def whiten_and_color(cF, sF):
    cF_size = cF.size()
    c_mean = torch.mean(cF, 1) 
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean # center by subtracting mean vector
    # transform linearly to obtain feature maps that are uncorrelated
    content_conv = torch.mm(cF, cF.t()).div(cF_size[1] - 1) + torch.eye(cF_size[0]).double()
    c_u, c_s, c_v = torch.svd(content_conv,some=False)

    sF_size = sF.size()
    s_mean = torch.mean(sF, 1)
    s_mean2 = s_mean.unsqueeze(1).expand_as(sF)
    sF = sF - s_mean2 # center by subtracting mean vector
    # transform linearly to obtain feature maps that are uncorrelated
    style_conv = torch.mm(sF, sF.t()).div(sF_size[1] - 1)
    s_u, s_s, s_v = torch.svd(style_conv,some=False)
        
    k_c = cF_size[0]
    for i in range(cF_size[0]):
        if c_s[i] < 0.00001:
            k_c = i
            break

    k_s = sF_size[0]
    for i in range(sF_size[0]):
        if s_s[i] < 0.00001:
            k_s = i
            break

    c_d = (c_s[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:,0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:,0:k_c].t()))
    whiten_cF = torch.mm(step2, cF)

    s_d = (s_s[0:k_s]).pow(0.5)
    target_feature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s], torch.diag(s_d)), (s_v[:,0:k_s].t())), whiten_cF)
    t_mean = s_mean.unsqueeze(1).expand_as(target_feature) # mean vector of style
    target_feature = target_feature + t_mean # re-center with the mean vector of style
    return target_feature

def transform(cF, sF, csF, alpha):
    cF = cF.double()
    sF = sF.double()

    cF_view = cF.view(cF.size(0), -1)
    sF_view = sF.view(cF.size(0), -1)

    target_feature = whiten_and_color(cF_view, sF_view)
    target_feature = target_feature.view_as(cF)
    # after WCT, blend with the content feature before feeding it to the decoder 
    ccsF = alpha * target_feature + (1.0 - alpha) * cF # alpha serves as the style weight for users to control the transfer effect
    ccsF = ccsF.float().unsqueeze(0)
    with torch.no_grad():
        csF.resize_(ccsF.size()).copy_(ccsF)
    return csF


