import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

# Load all images in a directory into memory
def load_images(imgpath, maskpath, labelpath, n_crops, size=(128, 128)):
    src_list, mask_list, label_list = [], [], []
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    for i in range(20, 40):
        for j in range(n_crops):  # Number of crops
            # Load and resize the image
            filename = f"{i+1}_{j+1}.png"
            mask_name = f"{i+1}_mask_{j+1}.png"
            label_name = f"{i+1}_label_{j+1}.png"
            
            img = Image.open(os.path.join(imgpath, filename)).convert('RGB')
            fundus_img = transform(img)
            
            mask = Image.open(os.path.join(maskpath, mask_name)).convert('L')
            mask_img = transform(mask)
            
            label = Image.open(os.path.join(labelpath, label_name)).convert('L')
            label_img = transform(label)
            
            src_list.append(fundus_img)
            mask_list.append(mask_img)
            label_list.append(label_img)
            
    return torch.stack(src_list), torch.stack(mask_list), torch.stack(label_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, nargs=2, default=(128, 128))
    parser.add_argument('--n_crops', type=int, default=210)
    parser.add_argument('--outfile_name', type=str, default='DRIVE')
    args = parser.parse_args()

    # Dataset path
    imgpath = 'Drive_crop/Images/'
    maskpath = 'Drive_crop/Masks/'
    labelpath = 'Drive_crop/labels/'
    
    # Load dataset
    src_images, mask_images, label_images = load_images(imgpath, maskpath, labelpath, args.n_crops, args.input_dim)
    print('Loaded:', src_images.shape, mask_images.shape, label_images.shape)
    
    # Save as compressed tensor file
    filename = f"{args.outfile_name}.npz"
    torch.save({'src_images': src_images, 'mask_images': mask_images, 'label_images': label_images}, filename)
    print('Saved dataset:', filename)
