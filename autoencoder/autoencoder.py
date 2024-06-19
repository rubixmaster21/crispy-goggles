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

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(36 * 36, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)  # bottleneck layer
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 36 * 36),
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

# testing= 0

# PROBLEM_INDICES=[]

# for arr in range(70000):
#     if np.isnan(input_data[arr]).any():
#         PROBLEM_INDICES.append(arr)

for i in range(700):
    loaded_array1d = np.fromfile('image_2D\\img2d_T500_'+str(i)+'.bin', dtype=np.float64)
    #print("read from file image_2D\\img3d_T500_"+str(i)+'.bin')
    loaded_array1d = loaded_array1d.astype(np.float32)
    loaded_array = loaded_array1d.reshape(368,368)
    
    tile_size=36
    if i==102:
        e=open('./among.txt','w')
        e.write(str(loaded_array))
        e.close()
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
            
                


# Normalize to range [0, 1]
#input_data = (input_data - np.min(input_data)) / (np.max(input_data)-np.min(input_data))

data = np.zeros((60000,36,36))
data = input_data[:60000]
np.random.shuffle(data)

# Transform to tensor and normalize further in range [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Create dataset and dataloader
dataset = CustomDataset(data, transform=transform)
dataloader = DataLoader(dataset, batch_size=600, shuffle=True)

# Instantiate the Autoencoder model
autoencoder = Autoencoder()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

histactivations = []

# Training loop
num_epochs = 2
#print("J!!!!!! MIXTILINAER! EXCIRCE!!!!")
for epoch in range(num_epochs):
    for data in dataloader:
        inputs = data.view(data.size(0), -1)  # Flatten input data
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
        optimizer.step()
        print(loss.item())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if epoch == 0:
        for layer_idx, activations in enumerate(autoencoder.neuron_activations):
            print(f'Layer {layer_idx + 1} neuron activations:')
            print(activations)
            print()
            histactivations.append(activations)
                

print("Training complete!")

# Evaluate the trained model (testing phase)
test_data = input_data[60001:60001+100]
np.random.shuffle(test_data)
test_dataset = CustomDataset(test_data, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

autoencoder.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for data in test_dataloader:
        inputs = data.view(data.size(0), -1)
        outputs = autoencoder(inputs)
        
        # Reshape back to original image shape if necessary
        inputs_img = inputs.view(data.size(0), 36, 36)
        outputs_img = outputs.view(data.size(0), 36, 36)
        
        # Here you could visualize or save the input/output images
        # For example, using matplotlib (not shown here)
        plt.imshow(inputs_img[0].cpu().numpy(), cmap='gray')
        plt.imshow(outputs_img[0].cpu().numpy(), cmap='gray')

print("Evaluation complete!")
