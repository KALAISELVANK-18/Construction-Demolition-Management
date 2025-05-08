import os
from glob import glob
import natsort
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
from lib.utils import rgb2mask
import matplotlib.pyplot as plt


class CoreDataset(Dataset):
    
    def __init__(self, path, transform=None):
        self.path_images = natsort.natsorted(glob(os.path.join(path, 'images', '*.png')))
        self.path_masks = natsort.natsorted(glob(os.path.join(path, 'masks', '*.png')))
        self.transform = transform
        
    def __len__(self):
        return len(self.path_images)
    
    def __getitem__(self, idx):
        
        image = Image.open(self.path_images[idx])
        mask = Image.open(self.path_masks[idx])
        
        sample = {'image':image, 'mask':mask}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
# add image normalization transform at some point
   
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image, mask = sample['image'], sample['mask']  
        # standard scaling would be probably better then dividing by 255 (subtract mean and divide by std of the dataset)
        image = np.array(image)/255
        # convert colors to "flat" labels

        # plt.imshow(mask, cmap='jet')  # You can adjust the colormap if needed
        # plt.axis('off')  # Turn off the axis for a cleaner view
        # plt.show()


        mask = rgb2mask(np.array(mask))

        # Display the image
        # ig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
        #
        # # Display the first image
        # axes[0].imshow(mask)
        # axes[0].axis('off')  # Hide axes
        # axes[0].set_title('Image 1')  # Title for first image
        #
        # # Display the second image
        # axes[1].imshow(image)
        # axes[1].axis('off')  # Hide axes
        # axes[1].set_title('Image 2')  # Title for second image
        #
        # # Adjust layout
        # plt.tight_layout()
        # plt.show()

        sample = {

                  'image': torch.from_numpy(image).permute(2,0,1).float(),

                  'mask': torch.from_numpy(mask).long(),

                 }
        
        return sample
    
def make_datasets(path, val_ratio):
    dataset = CoreDataset(path, transform = transforms.Compose([ToTensor()]))
    val_len = int(val_ratio*len(dataset))
    lengths = [len(dataset)-val_len, val_len]
    train_dataset, val_dataset = random_split(dataset, lengths)
    
    return train_dataset, val_dataset


def make_dataloaders(path, val_ratio, params):
    train_dataset, val_dataset = make_datasets(path, val_ratio)
    train_loader = DataLoader(train_dataset, drop_last=True, **params)
    val_loader = DataLoader(val_dataset, drop_last=True, **params)
    
    return train_loader, val_loader