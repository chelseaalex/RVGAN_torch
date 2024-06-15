import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from numpy import load
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
def load_real_data(filename):
    data = torch.load(filename)
    X1, X2, X3 = data['src_images'], data['mask_images'], data['label_images']


    # normalize from [0, 255] to [-1, 1]
    #print(X1)
    #X1 = (X1 - 127.5) / 127.5
    #X2 = (X2 - 127.5) / 127.5
    #X3 = (X3 - 127.5) / 127.5
    
    return [X1, X2, X3]

def generate_real_data(data, batch_id, batch_size, patch_shape):
    trainA, trainB, trainC = data

    start = batch_id * batch_size
    end = start + batch_size
    X1, X2, X3 = trainA[start:end], trainB[start:end], trainC[start:end]

    y1 = -np.ones((batch_size, patch_shape[0], patch_shape[0], 1))
    y2 = -np.ones((batch_size, patch_shape[1], patch_shape[1], 1))
    
    return [X1, X2, X3], [y1, y2]

def generate_real_data_random(data, random_samples, patch_shape):
    trainA, trainB, trainC = data

    id = np.random.randint(0, trainA.shape[0], random_samples)
    X1, X2, X3 = trainA[id], trainB[id], trainC[id]

    y1 = -np.ones((random_samples, patch_shape[0], patch_shape[0], 1))
    y2 = -np.ones((random_samples, patch_shape[1], patch_shape[1], 1))
    
    return [X1, X2, X3], [y1, y2]

def generate_fake_data_fine(g_model, batch_data, batch_mask, x_global, patch_shape):
    # 使用 g_model 生成假数据
    g_model=g_model.to(device)
    batch_data=batch_data.to(device)
    batch_mask=batch_mask.to(device)
    with torch.no_grad():
        X = g_model(batch_data, batch_mask, x_global).cpu().numpy()
        #X = g_model(batch_data, batch_mask, x_global).numpy()
    y1 = np.ones((len(X), patch_shape[0], patch_shape[0], 1))

    return X, y1

def generate_fake_data_coarse(g_model, batch_data, batch_mask, patch_shape):
    # 使用 g_model 生成假数据
    g_model=g_model.to(device)
    batch_data=batch_data.to(device)
    batch_mask=batch_mask.to(device)
    with torch.no_grad():
        #print(batch_data.shape, batch_mask.shape)
        X, X_global = g_model(batch_data, batch_mask)
        X, X_global = X.cpu().numpy(), X_global.cpu().numpy()
    #print(patch_shape)
    y1 = np.ones((len(X), patch_shape[1], patch_shape[1], 1))

    return X, X_global, y1

def resize(X_realA, X_realB, X_realC, out_shape):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(out_shape, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor()
    ])

    X_realA = torch.stack([transform(x) for x in X_realA])
    X_realB = torch.stack([transform(x) for x in X_realB])
    X_realC = torch.stack([transform(x) for x in X_realC])

    return [X_realA, X_realB, X_realC]
