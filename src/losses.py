import torch
import torch.nn as nn
import torch.nn.functional as F

def weighted_feature_matching_loss(fake_samples, image_input, real_samples, D, inner_weight, sample_weight):
    # 通过判别器获取特征
    y_fake = D(image_input,fake_samples)[1:]
    y_real = D(image_input,real_samples)[1:]

    # 计算特征匹配损失
    fm_loss = 0
    for i in range(len(y_fake)):
        if i < 3:
            fm_loss += inner_weight * torch.mean(torch.abs(y_fake[i] - y_real[i]))
        else:
            fm_loss += (1 - inner_weight) * torch.mean(torch.abs(y_fake[i] - y_real[i]))

    fm_loss *= sample_weight
    return fm_loss
