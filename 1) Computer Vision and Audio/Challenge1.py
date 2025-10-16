import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Autoencoder copied from class
class Snake(nn.Module):
  def __init__(self, alpha=1.0):
    super().__init__()
    # Make alpha a learnable parameter
    self.alpha = nn.Parameter(torch.tensor(alpha))
    
  def forward(self, x):
    return x + (1.0/self.alpha) * torch.pow(torch.sin(self.alpha * x), 2)
    
class Autoencoder(pl.LightningModule):
  def __init__(self):
    super().__init__()
    # Encoder layers
    self.enc1 = nn.Sequential(
      nn.Conv1d(1, 64, kernel_size=7, stride=3, padding=1),
      Snake()
    )
    self.enc2 = nn.Sequential(
      nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
      nn.LeakyReLU()
    )
    self.enc3 = nn.Sequential(
      nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
      nn.LeakyReLU()
    )
    self.enc4 = nn.Sequential(
      nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=0),
      nn.LeakyReLU()
    )
    # Decoder layers with skip connections
    self.dec1 = nn.Sequential(
      nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=0),
      nn.LeakyReLU()
    )
    self.dec2 = nn.Sequential(
      nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1),
      nn.LeakyReLU()
    )
    self.dec3 = nn.Sequential(
      nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.LeakyReLU()
    )
    self.dec4 = nn.Sequential(
      nn.ConvTranspose1d(128, 1, kernel_size=7, stride=3, padding=1),
      nn.Tanh()
    )
    
  def forward(self, x):
    # Encoder path with stored intermediate outputs
    e1 = self.enc1(x)
    e2 = self.enc2(e1)
    e3 = self.enc3(e2)
    e4 = self.enc4(e3)
    
    # Decoder path with skip connections
    d1 = self.dec1(e4)
    d1 = torch.cat([d1, e3], dim=1)  # Skip connection 1
    d2 = self.dec2(d1)
    d2 = torch.cat([d2, e2], dim=1)  # Skip connection 2
    d3 = self.dec3(d2)
    d3 = torch.cat([d3, e1], dim=1)  # Skip connection 3
    
    x_hat = self.dec4(d3)
    return x_hat
    
  def training_step(self, batch, batch_idx):
    noisy_sample, ground_truth = batch
    
    # Forward pass
    x_hat = self(noisy_sample)
    
    # Pad x_hat to match x if lengths differ
    if x_hat.size(-1) != ground_truth.size(-1):
      padding = ground_truth.size(-1) - x_hat.size(-1)
      x_hat = torch.nn.functional.pad(x_hat, (0, padding))
    
    # Reconstruction loss
    recon_loss = nn.L1Loss()(ground_truth, x_hat) + nn.MSELoss()(ground_truth, x_hat)
    self.log('recon_loss', recon_loss, prog_bar=True)
    return recon_loss
    
  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=0.001)



# Do not change function signature
def init_model():
  model = Autoencoder()
  return model



# Do not change function signature
def train_model(model, train_dataset):
  
  #train_dataset = MyDataset(train_dataset)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  
  trainer = pl.Trainer(
    max_epochs=200, 
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    benchmark=True,
    #num_sanity_val_steps=0,
    precision='16-mixed',
    #accumulate_grad_batches=4,
    log_every_n_steps=10,
    gradient_clip_val=1.0,
    #detect_anomaly=False,
    enable_progress_bar=False)
  trainer.fit(model, train_loader)
  
  return model



