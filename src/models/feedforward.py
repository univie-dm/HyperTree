import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim=784, learning_rate=1e-3, embedding_dim=10,):
        super().__init__()
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # Decoder layers (mirror of encoder)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Get embeddings
        embedding = self.encoder(x)

        
        # Reconstruct
        reconstruction = self.decoder(embedding)
        
        return embedding, reconstruction

    def training_step(self, batch, batch_idx):
        _, x, _ = batch
        x = x.view(x.size(0), -1)
        
        # Forward pass
        embedding, reconstruction = self(x)
        
        # Calculate reconstruction loss
        loss = F.mse_loss(reconstruction, x)
        self.log('pretraining/loss', loss, prog_bar=True, on_epoch=True)
        self.log('pretraining/train_loss', loss, prog_bar=True, on_epoch=True)
            
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
