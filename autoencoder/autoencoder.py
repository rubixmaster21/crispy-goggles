# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:18:59 2024

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
# Autoencoder linear layer good for 1D signal process
################################################
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(36 * 36, 256), #2D 36 pix by 36 pix, output 256 points
            nn.ReLU(),  #Non-linear operation
            nn.Linear(256, 128),  #compress from 256->64
            nn.ReLU(),
            nn.Linear(128,64),   #compress from 64->32, maybe compress too much (info cannot be recovered)
            #nn.Linear(1296,1296)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 36 * 36),
            #nn.Linear(1296,1296),
            nn.Sigmoid()  # Sigmoid activation to output values in [0, 1]
        )
        
        self.neuron_activations = []
        self.encoder.register_forward_hook(self.store_neuron_activations)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def store_neuron_activations(self, module, input, output):
        # Store neuron activations for all layers
        self.neuron_activations.append(output.detach().cpu().numpy())


################################################
# Convolution layer for 2D image process
################################################
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (batch_size, 1, 32, 32) -> (batch_size, 16, 16, 16)
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (batch_size, 16, 16, 16) -> (batch_size, 32, 8, 8)
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 7)  # (batch_size, 32, 8, 8) -> (batch_size, 64, 2, 2)
        # )
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, 7),  # (batch_size, 64, 2, 2) -> (batch_size, 32, 8, 8)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # (batch_size, 32, 8, 8) -> (batch_size, 16, 16, 16)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # (batch_size, 16, 16, 16) -> (batch_size, 1, 32, 32)
        #     nn.Sigmoid()  # Sigmoid activation to output values in [0, 1]
        # )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1),  # (batch_size, 1, 32, 32) -> (batch_size, 16, 16, 16)
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),  # (batch_size, 16, 16, 16) -> (batch_size, 32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(8, 16, 7)  # (batch_size, 32, 8, 8) -> (batch_size, 64, 2, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 7),  # (batch_size, 64, 2, 2) -> (batch_size, 32, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),  # (batch_size, 32, 8, 8) -> (batch_size, 16, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1),  # (batch_size, 16, 16, 16) -> (batch_size, 1, 32, 32)
            nn.Sigmoid()  # Sigmoid activation to output values in [0, 1]
        )


    def forward(self, x):
        encoded = self.encoder(x)
        #print(encoded.shape)
        decoded = self.decoder(encoded)
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
input_data = np.zeros((70000, 36, 36),np.float32)

loaded_array1d = np.zeros((36,36),np.float32)

#from mpl_toolkits.axes_grid1 import ImageGrid


loaded_images = []
for i in range(700):
    loaded_array1d = np.fromfile('image_2D\\img2d_T500_'+str(i+2243)+'.bin', dtype=np.float64)
    #print("read from file image_2D\\img3d_T500_"+str(i)+'.bin')
    
    loaded_array1d = loaded_array1d.astype(np.float32)
    loaded_array = loaded_array1d.reshape(368,368)
    # plt.imshow(loaded_array)
    # plt.show()
    loaded_images.append(loaded_array)
    
    tile_size=36
    
    
    # fig = plt.figure(figsize=(10, 10))
    # grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 # nrows_ncols=(10, 10),  # creates 2x2 grid of Axes
                 # axes_pad=0.05,  # pad between Axes in inch.
                 # )
    #tiles=[np.zeros((36,36),np.float32) for i in range(100)]
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
            # tiles[10*ii+j]=tile
            #print(tile)
    # for ax, im in zip(grid, tiles):
    # # Iterating over the grid returns the Axes.
    #     ax.imshow(im, vmin = 0, vmax = 1)
    
    # plt.show()
        

            
#input_data = np.random.rand(70000,36,36).astype(np.float32)
#input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

# Normalize to range [0, 1]
#input_data = (input_data - np.min(input_data)) / (np.max(input_data)-np.min(input_data))

data = np.zeros((60000,36,36))
data = input_data[:60000]
np.random.shuffle(data)

# Transform to tensor and normalize further in range [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5], std=[0.5])
])


# Create dataset and dataloader
dataset = CustomDataset(data, transform=transform)
dataloader = DataLoader(dataset, batch_size=600, shuffle=True)

# Instantiate the Autoencoder model
#autoencoder = Autoencoder().to(device)
autoencoder = ConvAutoencoder().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=8e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)


# Evaluate the trained model (testing phase)
test_data = input_data[62000:62000+100]
#np.random.shuffle(test_data)
#test_data = test_data[:10]
test_dataset = CustomDataset(test_data, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


all_train_loss = []
all_val_loss   = []


# Training loop
num_epochs = 60
for epoch in range(num_epochs):
    autoencoder.train()
    for data in dataloader:
        #inputs = data.view(data.size(0), -1).to(device)  # Flatten input data
        inputs = data.to(device)
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
        
        
        optimizer.step()
        

    #print('epoch ',epoch, 'lr ',scheduler.get_lr())
    print('epoch ',epoch, 'lr ', optimizer.param_groups[0]['lr'])
    #print('\n')
    
    scheduler.step(loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, ', end='')
    all_train_loss.append(loss.cpu().item())
    # if epoch == 0:
    #     for layer_idx, activations in enumerate(autoencoder.neuron_activations):
    #         #print(f'Layer {layer_idx + 1} neuron activations:')
    #         #print(activations)
    #         #print()
    #         histactivations.append(activations)
                

#print("Training complete!")

    autoencoder.eval()  # Set the model to evaluation mode
    #tesdata=[]
        
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
            
            # Here you could visualize or save the input/output images
            # For example, using matplotlib (not shown here)

            for i in range(inputs_img.size(0)):
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(inputs_img[i].cpu().numpy(),  vmin = 0, vmax = 1,cmap='gray')
                axes[0].set_title('Input Image')
                axes[1].imshow(outputs_img[i].cpu().numpy(),  vmin = 0, vmax = 1,cmap='gray')
                axes[1].set_title('Reconstructed Image')
                
            # Print selected images. Remove if-logic below for saving all the images.
            #if ((epoch+1)%10 == 0):
            plt.savefig(os.path.join('.', f'new_image_{idx}_{i}.png'))

            # Plot pictures in Spyder->Plots
            if ((epoch+1)%10 == 0) & (idx%20 == 0):
                plt.show()
            plt.close(fig)
        
        #plt.imshow(loaded_images[620])
        plt.savefig(os.path.join('.', 'result_true.png'))
        loss_val_mean = np.mean(loss_val_total)          
        print('Val_loss: %f .' % loss_val_mean)
        all_val_loss.append(loss_val_mean)
        
        np.array(all_train_loss).tofile('loss_train.csv', sep=',')
        np.array(all_val_loss).tofile('loss_val.csv', sep=',')
        
        print('\n')
            

plt.figure(1)
plt.plot(all_train_loss)
plt.title("Train loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Train")

plt.figure(2)
#print('Validation loss') #; print(all_val_loss)
plt.plot(all_val_loss)
plt.title("Validation loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss/val")
#print("Evaluation complete!")
