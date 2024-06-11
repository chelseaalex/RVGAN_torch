import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
import numpy as np
import argparse
import time
import os
from numpy import load
import gc
from src.model import CoarseGenerator,FineGenerator,RVGAN,DiscriminatorAE
from src.visualization import summarize_performance, summarize_performance_global, plot_history, to_csv
from src.dataloader import resize, generate_fake_data_coarse, generate_fake_data_fine, generate_real_data, generate_real_data_random, load_real_data

def train(d_model1, d_model2, g_global_model, g_local_model, gan_model, dataset, n_epochs=20, n_batch=1, n_patch=[64, 32], savedir='RVGAN'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainA, trainB, trainC = dataset
    train_loader = DataLoader(TensorDataset(torch.tensor(trainA), torch.tensor(trainB), torch.tensor(trainC)), batch_size=n_batch, shuffle=True)
    optimizer_d1 = optim.Adam(d_model1.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d2 = optim.Adam(d_model2.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_g_global = optim.Adam(g_global_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_g_local = optim.Adam(g_local_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_gan = optim.Adam(gan_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.MSELoss()
    hinge_loss = nn.HingeEmbeddingLoss()
    start_time = time.time()
    
    d1_hist, d2_hist, d3_hist, d4_hist =  list(),list(), list(), list()
    fm1_hist,fm2_hist = list(),list()
    g_global_hist, g_local_hist, gan_hist =  list(), list(), list()
    g_global_recon_hist, g_local_recon_hist =list(),list()
    
    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):
            X_realA, X_realB, X_realC = data

            X_realA = X_realA.float().to(device)
            X_realB = X_realB.float().to(device)
            X_realC = X_realC.float().to(device)
            y1 = torch.tensor(-np.ones((n_batch,1, n_patch[0], n_patch[0])),dtype=torch.float32).to(device)
            y2 = torch.tensor(-np.ones((n_batch,1, n_patch[1], n_patch[1])),dtype=torch.float32).to(device)
            # Generate fake data for coarse generator
            out_shape = (X_realA.size(2) // 2, X_realA.size(3) // 2)
            X_realA_half, X_realB_half, X_realC_half = resize(X_realA, X_realB, X_realC, out_shape)
            """
            X_realA_half
            X_realB_half
            X_realC_half
            out_shape=
            """
            g_global_model=g_global_model.to(device)
            X_fakeC_half, x_global ,y1_coarse = generate_fake_data_coarse(g_global_model, X_realA_half, X_realB_half, n_patch)
            #print(type(X_fakeC_half))
            X_fakeC_half = torch.tensor(X_fakeC_half,dtype=torch.float32).to(device)
            x_global = torch.tensor(x_global,dtype=torch.float32).to(device)
            y1_coarse = torch.tensor(y1_coarse,dtype=torch.float32).to(device)


            # Generate fake data for fine generator
            X_fakeC, y1_fine = generate_fake_data_fine(g_local_model, X_realA, X_realB, x_global, n_patch)
            X_fakeC = torch.tensor(X_fakeC,dtype=torch.float32).to(device)
            y1_fine = torch.tensor(y1_fine,dtype=torch.float32).to(device)
            # Update fine discriminator
            d_model1.train()
            optimizer_d1.zero_grad()
            #y1 = torch.ones(n_batch, 1).to(device)
            #y1_coarse = torch.ones(n_batch, 1).to(device)
            
            #print(y1.shape)
            #print(d_model1(X_realA, X_realC).shape)
            #print(type(d_model1(X_realA, X_realC)))
            d_loss1_real = criterion(d_model1(X_realA, X_realC), y1)
            d_loss1_fake = criterion(d_model1(X_realA, X_fakeC.detach()), y1_fine)
            d_loss1 = (d_loss1_real + d_loss1_fake) / 2
            d_loss1.backward()
            optimizer_d1.step()

            # Update coarse discriminator
            d_model2.train()
            optimizer_d2.zero_grad()
            X_realA_half=X_realA_half.to(device)
            X_realB_half=X_realB_half.to(device)
            X_realC_half=X_realC_half.to(device)
            d_loss2_real = criterion(d_model2(X_realA_half, X_realC_half), y2)
            d_loss2_fake = criterion(d_model2(X_realA_half, X_fakeC_half.detach()), y1_coarse)
            d_loss2 = (d_loss2_real + d_loss2_fake) / 2
            d_loss2.backward()
            optimizer_d2.step()

            # Update global generator
            g_global_model.train()
            optimizer_g_global.zero_grad()
            g_glo_out,_ = g_global_model(X_realA_half, X_realB_half)
            g_global_loss = criterion(g_glo_out, X_realC_half)
            g_global_loss.backward()
            optimizer_g_global.step()

            # Update local generator
            g_local_model.train()
            optimizer_g_local.zero_grad()
            g_local_loss = criterion(g_local_model(X_realA, X_realB, x_global), X_realC)
            g_local_loss.backward()
            optimizer_g_local.step()


            # Update GAN model
            gan_model.train()
            optimizer_gan.zero_grad()
            d_out_1_f,d_out_2_f,gen_out_fine,gen_out_coarse,_, _,_, _, fm1, fm2=gan_model(
            X_realA, X_realA_half, x_global, X_realB, X_realB_half, y1, y2, sample_weight=1)
            
            #in_fine, in_coarse, in_x_coarse, in_fine_mask, in_coarse_mask, label_fine, label_coarse, sample_weight)


            """
            gan_loss, _, _, fm1_loss, fm2_loss, _, _, g_global_recon_loss, g_local_recon_loss = gan_model(
                X_realA, X_realA_half, x_global, X_realB, X_realB_half, X_realC, X_realC_half, y1, y1_coarse, criterion
            )
            total_gan_loss = gan_loss + fm1_loss + fm2_loss + g_global_recon_loss + g_local_recon_loss
            """
            total_gan_loss = criterion(d_out_1_f, y1) + criterion(d_out_2_f, y2) + hinge_loss(gen_out_fine, y1) + hinge_loss(gen_out_coarse, y2) + fm1 + fm2
            total_gan_loss.backward()
            optimizer_gan.step()

            d1_hist.append(d_loss1_real.item())
            d2_hist.append(d_loss1_fake.item())
            d3_hist.append(d_loss2_real.item())
            d4_hist.append(d_loss2_fake.item())
            fm1_hist.append(fm1.item())
            fm2_hist.append(fm2.item())
            g_global_hist.append(g_global_loss.item())
            g_local_hist.append(g_local_loss.item())
            g_global_recon_hist.append(hinge_loss(gen_out_coarse, y2).item())
            g_local_recon_hist.append(hinge_loss(gen_out_fine, y1).item())
            gan_hist.append(total_gan_loss.item())
            
            if i%200==0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'd_loss1: {d_loss1.item():.4f}, d_loss2: {d_loss2.item():.4f}, '
                      f'g_global_loss: {g_global_loss.item():.4f}, g_local_loss: {g_local_loss.item():.4f}, '
                      f'gan_loss: {total_gan_loss.item():.4f}, fm1_loss: {fm1.item():.4f}, fm2_loss: {fm2.item():.4f}')

            # Summarize performance
        summarize_performance_global(epoch, g_global_model, dataset, n_samples=3, savedir=savedir)
        summarize_performance(epoch, g_global_model, g_local_model, dataset, n_samples=3, savedir=savedir)

    plot_history(d1_hist, d2_hist, fm1_hist, fm2_hist, g_global_hist, g_local_hist, g_global_recon_hist, g_local_recon_hist, gan_hist, savedir=savedir)
    to_csv(d1_hist, d2_hist, fm1_hist, fm2_hist, g_global_hist, g_local_hist, g_global_recon_hist, g_local_recon_hist, gan_hist, savedir=savedir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--npz_file', type=str, default='DRIVE.npz', help='path/to/npz/file')
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--savedir', type=str, required=False, help='path/to/save_directory', default='RVGAN')
    parser.add_argument('--resume_training', type=str, required=False, default='no', choices=['yes', 'no'])
    parser.add_argument('--weight_name_global', type=str, help='path/to/global/weight/.h5 file', required=False)
    parser.add_argument('--weight_name_local', type=str, help='path/to/local/weight/.h5 file', required=False)
    parser.add_argument('--inner_weight', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_real_data(args.npz_file)
    print('Loaded', dataset[0].shape, dataset[1].shape)
    n_batch=args.batch_size
    in_size = args.input_dim
    
    image_shape_coarse = (3, in_size // 2, in_size // 2)
    mask_shape_coarse = (1, in_size // 2, in_size // 2)
    label_shape_coarse = (1, in_size // 2, in_size // 2)

    image_shape_fine = (3, in_size, in_size)
    mask_shape_fine = (1, in_size, in_size)
    label_shape_fine = (1, in_size, in_size)

    image_shape_xglobal = (64, in_size // 2, in_size // 2)
    ndf = 32
    ncf = 64
    nff = 64

    d_model1 = DiscriminatorAE(image_shape_fine, label_shape_fine, ndf).to(device)
    d_model2 = DiscriminatorAE(image_shape_coarse, label_shape_coarse, ndf).to(device)

    g_model_fine = FineGenerator(x_coarse_shape=image_shape_xglobal, input_shape=image_shape_fine, mask_shape=mask_shape_fine, nff=nff, n_blocks=3).to(device)
    g_model_coarse = CoarseGenerator(img_shape=image_shape_coarse, mask_shape=mask_shape_coarse, n_downsampling=2, n_blocks=9, ncf=ncf, n_channels=1).to(device)

    if args.resume_training == 'yes':
        g_model_coarse.load_state_dict(torch.load(args.weight_name_global))
        g_model_fine.load_state_dict(torch.load(args.weight_name_local))

    rvgan_model = RVGAN(g_model_fine, g_model_coarse, d_model1, d_model2, args.inner_weight).to(device)

    train(d_model1, d_model2, g_model_coarse, g_model_fine, rvgan_model, dataset, n_epochs=args.epochs, n_batch=args.batch_size, n_patch=[128,64], savedir=args.savedir)
