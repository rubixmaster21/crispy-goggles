# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:44:12 2024

@author: uyy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os

torch.backends.cudnn.enabled = True
print(torch.backends.cudnn.enabled)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

################################################
# Convolution layer for 2D image process
################################################
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (batch_size, 1, 32, 32) -> (batch_size, 16, 16, 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (batch_size, 16, 16, 16) -> (batch_size, 32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # (batch_size, 32, 8, 8) -> (batch_size, 64, 2, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # (batch_size, 64, 2, 2) -> (batch_size, 32, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # (batch_size, 32, 8, 8) -> (batch_size, 16, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # (batch_size, 16, 16, 16) -> (batch_size, 1, 32, 32)
            nn.Sigmoid()  # Sigmoid activation to output values in [0, 1]
        )
        
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 4, 3, stride=2, padding=1),  # (batch_size, 1, 32, 32) -> (batch_size, 16, 16, 16)
        #     nn.ReLU(),
        #     nn.Conv2d(4, 8, 3, stride=2, padding=1),  # (batch_size, 16, 16, 16) -> (batch_size, 32, 8, 8)
        #     nn.ReLU(),
        #     nn.Conv2d(8, 16, 7)  # (batch_size, 32, 8, 8) -> (batch_size, 64, 2, 2)
        # )
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(16, 8, 7),  # (batch_size, 64, 2, 2) -> (batch_size, 32, 8, 8)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),  # (batch_size, 32, 8, 8) -> (batch_size, 16, 16, 16)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1),  # (batch_size, 16, 16, 16) -> (batch_size, 1, 32, 32)
        #     nn.Sigmoid()  # Sigmoid activation to output values in [0, 1]
        # )


    def forward(self, x):
        encoded = self.encoder(x).half()
        #print(encoded.shape)
        decoded = self.decoder(encoded.float())
        return decoded



# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample



# Dummy data (100 samples of 28x28 images)
input_data = np.zeros((100000, 36, 36),np.float32)

loaded_array1d = np.zeros((36,36),np.float32)

#from mpl_toolkits.axes_grid1 import ImageGrid


loaded_images = []
for i in range(1000):
    loaded_array1d = np.fromfile('image_2D\\img2d_T500_'+str(i+2243)+'.bin', dtype=np.float64)
    #print("read from file image_2D\\img3d_T500_"+str(i)+'.bin')
    
    loaded_array1d = loaded_array1d.astype(np.float32)
    loaded_array = loaded_array1d.reshape(368,368)
    # plt.imshow(loaded_array)
    # plt.show()
    loaded_images.append(loaded_array)
    
    tile_size=36
    
    for ii in range(10):
        for j in range(10):
            # Calculate the indices for slicing
            start_x = ii * tile_size+4
            end_x = start_x + tile_size
            start_y = j * tile_size+4
            end_y = start_y + tile_size
            
            # Extract the tile
            tile = loaded_array[start_x:end_x, start_y:end_y]
            
            # Append the tile to the list
            input_data[100*i+10*ii+j] = tile
    
data = np.zeros((60000,36,36))
data = input_data[:60000]

# Transform to tensor and normalize further in range [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5], std=[0.5])
])


# Create dataset and dataloader
dataset = CustomDataset(data, transform=transform)
dataloader = DataLoader(dataset, batch_size=600, shuffle=True)

# Instantiate the Autoencoder model
autoencoder = ConvAutoencoder().to(device)
autoencoder.load_state_dict(torch.load('./autoencoder.h'))#.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=8e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)


# Evaluate the trained model (testing phase)
test_data = input_data[70000:70000+20*100]
#np.random.shuffle(test_data)
#test_data = test_data[:10]
test_dataset = CustomDataset(test_data, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

all_val_loss   = []

num_test_img=20
reconstructed_images = [np.zeros((360,360),np.float32) for i in range(num_test_img)]
with torch.no_grad():
    loss_val_total = []
    for idx, data in enumerate(test_dataloader):
        #inputs = data.view(data.size(0), -1).to(device)
        inputs = data.to(device)
        outputs = autoencoder(inputs)
        loss_val = criterion(outputs, inputs)
        loss_val_total.append(loss_val.cpu().item())
        
        # Reshape back to original image shape if necessary
        inputs_img = inputs.view(data.size(0), 36, 36)
        outputs_img = outputs.view(data.size(0), 36, 36)
        
        j=outputs_img.cpu().numpy()
        print(outputs_img.cpu().numpy())
        
        # Here you could visualize or save the input/output images
        # For example, using matplotlib (not shown here)

        # for i in range(inputs_img.size(0)):
        #     fig, axes = plt.subplots(1, 2)
        #     axes[0].imshow(inputs_img[i].cpu().numpy(),  vmin = 0, vmax = 1,cmap='gray')
        #     axes[0].set_title('Input Image')
        #     axes[1].imshow(outputs_img[i].cpu().numpy(),  vmin = 0, vmax = 1,cmap='gray')
        #     axes[1].set_title('Reconstructed Image')
        #     plt.show()
            
        # Print selected images. Remove if-logic below for saving all the images.
        #if ((epoch+1)%10 == 0):
        #plt.savefig(os.path.join('.', f'new_image_{idx}_{i}.png'))
        print(idx)
        if (idx%100==0):
            print(idx//100)
            # print(10*(i%10))
            # print(10*((i//100)//10))
        
        reconstructed_images[idx//100][36*((idx//10)%10):36*((idx//10)%10+1),36*(idx%10):36*(idx%10+1)]=j
        # Put the image back together. The second index goes [10*ones digit,10*tens digit].
        
    #plt.imshow(loaded_images[620])
    #plt.savefig(os.path.join('.', 'result_true.png'))
    #loss_val_mean = np.mean(loss_val_total)          
    #print('Val_loss: %f .' % loss_val_mean)
    #all_val_loss.append(loss_val_mean)
    
    np.array(all_val_loss).tofile('./testing/loss_val.csv', sep=',')
    
    print('\n')
  
    
for i in range(len(reconstructed_images)):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(loaded_images[i],  vmin = 0, vmax = 1,cmap='gray')
    axes[0].set_title('Input Image')
    axes[1].imshow(reconstructed_images[i],  vmin = 0, vmax = 1,cmap='gray')
    axes[1].set_title('Reconstructed Image')
    
#plt.figure(2)
#print('Validation loss') #; print(all_val_loss)
#plt.plot(all_val_loss)
#plt.title("Validation loss")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss/val")
