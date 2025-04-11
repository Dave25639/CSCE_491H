# Based on https://github.com/sonjakatz/methAE_explainableAE_methylation/blob/master/models/autoencoder.py

import torch
from torch import nn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self,
                inputDim,
                latentSize,
                hidden_layer_encoder_topology=[]):
        super(Autoencoder, self).__init__()
        self.inputDim = inputDim
        self.hidden_layer_encoder_topology = hidden_layer_encoder_topology
        self.latentSize = latentSize

        ### Define encoder
        self.encoder_topology = [self.inputDim] + self.hidden_layer_encoder_topology + [self.latentSize]
        self.encoder_layers = []
        for i in range(len(self.encoder_topology)-1):
            layer = nn.Linear(self.encoder_topology[i],self.encoder_topology[i+1])
            torch.nn.init.xavier_normal_(layer.weight)  ## weight initialisation
            self.encoder_layers.append(layer)
            self.encoder_layers.append(nn.PReLU())
            self.encoder_layers.append(nn.BatchNorm1d(self.encoder_topology[i+1])) ## add this for better training?
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        ### Define decoder
        self.decoder_topology = [self.latentSize] + self.hidden_layer_encoder_topology[::-1] + [self.inputDim]
        self.decoder_layers = []
        for i in range(len(self.decoder_topology)-1):
            layer = nn.Linear(self.decoder_topology[i],self.decoder_topology[i+1])
            torch.nn.init.xavier_uniform_(layer.weight)  ### weight initialisation
            self.decoder_layers.append(layer)
            self.decoder_layers.append(nn.PReLU())
        self.decoder_layers[-1] = nn.Sigmoid() ### replace activation of final layer with Sigmoid()
        self.decoder = nn.Sequential(*self.decoder_layers)

    def encode(self, x):
        hidden = self.encoder(x)
        return hidden
    
    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def generate_embedding(self,x):
        z = self.encode(x)
        return z
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

if __name__=='__main__':
    # Load train data
    data_paths = ["filtered_methylation_data/luad_top250kMAD_cpg.parquet", "filtered_methylation_data/read_top250kMAD_cpg.parquet"]

    df_list = []

    for path in data_paths:
        df = pd.read_parquet(path)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)

    data_tensor = torch.tensor(combined_df.values, dtype=torch.float32)

    batch_size = 16
    dataset = TensorDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize and train model
    input_size = df.shape[1]
    latent_dim = 100
    hidden_layers = [256, 128]

    device = torch.device("cuda")
    model = Autoencoder(input_size, latent_dim, hidden_layers).to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in data_loader:
            batch_data = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            loss = loss_function(reconstructed, batch_data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(data_loader):.6f}")

    # Update model metadata
    metadata_file = "models/model_metadata.csv"

    metadata = {
        "model_name": "ae_normalAE",
        "encoder_type": "ae",
        "input_dim": input_size,
        "latent_dim": latent_dim,
        "hidden_layers": hidden_layers,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "loss_function": "MSELoss",
        "epochs": epochs,
        "batch_size": batch_size,
    }

    try:
        log_df = pd.read_csv(metadata_file)
    except FileNotFoundError:
        log_df = pd.DataFrame(columns=metadata.keys())

    metadata_df = pd.DataFrame([metadata])
    log_df = pd.concat([log_df, metadata_df], ignore_index=True)

    log_df.to_csv(metadata_file, index=False)

    model_path = "models/ae_normalAE.pth"
    torch.save(model.state_dict(), model_path)