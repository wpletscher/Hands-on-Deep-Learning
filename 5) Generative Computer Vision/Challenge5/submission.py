import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 8000
batch_size = 10
rune_label = 10


class LabeledDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, label):
    self.dataset = dataset
    self.label = label

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item[0] if isinstance(item, tuple) else item
    return image, torch.tensor(self.label, dtype=torch.long)


class LoRaWrappedModel(nn.Module):
  def __init__(self, model, input_dim, rank, output_dim):
    super().__init__()
    self.model = model
    for name,param in model.named_parameters():
      if "class_embedding" in name:
        param.requires_grad = True
      else:
        param.requires_grad = False
    self.A = nn.Linear(input_dim, rank, bias=False).to(device)
    self.B = nn.Linear(rank, output_dim, bias=False).to(device)
    self.B.weight.data.zero_()

  def forward(self, x, t, y):
    base = self.model(x, t, y) 
    class_emb = self.model.class_embedding(y)
    x_flat = x.view(x.shape[0], -1)
    x_lora = torch.cat([x_flat, class_emb], dim=1)
    correction = self.A(x_lora) @ self.B.weight.T
    correction = correction.view_as(x)
    return base + correction


# Do not change function signature
def finetune_model(pretrained_model, train_dataset):
  old_embed = pretrained_model.class_embedding
  new_embed = nn.Embedding(11, old_embed.embedding_dim).to(device)
  new_embed.weight.data[:10] = old_embed.weight.data
  new_embed.weight.data[10] = old_embed.weight.data.mean(dim=0)
  pretrained_model.class_embedding = new_embed

  labeled_dataset = LabeledDataset(train_dataset, label=rune_label)
  dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)

  model = LoRaWrappedModel(pretrained_model, input_dim=800, rank=10, output_dim=784)
  
  #optimization_parameters = list(model.A.parameters()) + list(model.B.parameters())
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()) , lr=1e-2)
  
  for epoch in range(num_epochs):
    for x, y in dataloader:
      x = x.to(device)
      y = y.to(device) 
      t = torch.rand(batch_size, 1, 1, 1).to(device)
      noise = torch.randn_like(x)
      x_t = (1 - t) * x + t * noise
      
      v_pred = model(x_t, t, y)
      v_true = noise - x
      loss = F.mse_loss(v_pred, v_true)
      if(epoch % 10 == 0):
        print(f"Epoch {epoch}, Train loss: {loss:.4f}")
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  
  # visualize(model)
  return model




def generate_images_rectified_flow(model, num_samples):
  x = torch.randn(num_samples, 1, 28, 28).to(device)
  for step in range(1000, 0, -1):
    timestep = torch.full((num_samples, 1, 1, 1), step).to(device)
    class_label_tensor = torch.full((num_samples,), 10, dtype=torch.long).to(device)
    noise = model(x, timestep/1000, class_label_tensor)
    x = x - 1/1000 * noise
  return x


def visualize(model):
  num_samples = 4
  with torch.no_grad():
    samples = generate_images_rectified_flow(model, num_samples)
  plt.figure(figsize=(8, 8), dpi=64)
  for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    ax.imshow(samples[i].cpu().squeeze().clip(0, 1), cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
  plt.tight_layout(pad=0)
  plt.show()







