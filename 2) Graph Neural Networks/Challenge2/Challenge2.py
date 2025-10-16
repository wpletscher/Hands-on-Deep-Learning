import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
import numpy as np
import torch.nn.functional as F
from torch.nn import GRUCell, ModuleList
from sklearn.metrics import f1_score
from copy import deepcopy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class MazeConv(MessagePassing):
  def __init__(self, edge_nn, nn):
    super(MazeConv, self).__init__(aggr='add')
    #initialize custom message passing, store the MLP's for the convolution
    self.edge_nn = edge_nn
    self.nn = nn

  def forward(self, x, edge_index):
    #define own computation, call the round with
    x = self.nn(self.propagate(edge_index, x=x))
    return x

  def message(self, x_j, x_i):
    #define the custom message that gnns exchange, x_i is own state, x_j is neighboring state
    concatted = torch.cat((x_j, x_i), dim=1)
    return self.edge_nn(concatted)




class MazeGNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.dropout = 0.2
    
    dim = 27
    hidden_dim = dim
    
    
    self.edge_network = torch.nn.Sequential(
      torch.nn.Linear(2*dim, dim), 
      torch.nn.ReLU()
    )
    self.network = torch.nn.Sequential(
      torch.nn.Linear(dim, dim), 
      torch.nn.ReLU()
    )
    
    
    self.encoder = self.get_mlp(2,8,hidden_dim)
    self.decoder = self.get_mlp(hidden_dim,32,2, last_relu = False)
    self.edge_network_mlp = self.get_mlp(2*dim, dim, dim)
    self.network_mlp = self.get_mlp(dim, dim, dim)
    self.concat_mlp = self.get_mlp(2 * hidden_dim, hidden_dim, hidden_dim)
    
    self.conv = MazeConv(
      edge_nn = self.edge_network_mlp,
      nn = self.network_mlp
    )
    
    self.gru = GRUCell(dim, dim)


  # get the graph and number of nodes in the graph
  def forward(self, data, num_nodes):
    x, edge_index = data.x, data.edge_index
    
    x = self.encoder(x)                       # Initial embedding
    h = x                                     # Initial hidden state for GRU
    for _ in range(num_nodes):                # e.g., 3 message-passing steps
      concat = torch.cat((x, h), dim=1)
      concat = self.concat_mlp(concat)        # Optional feature fusion
      x_new = self.conv(concat, edge_index)   # Message passing
      h = self.gru(x_new, h)                  # GRU update
    out = self.decoder(h)
    return F.log_softmax(out, dim=1)


  # helper function - generates an MLP w. relu activation with 3 layers
  def get_mlp(self, input_dim, hidden_dim, output_dim, last_relu = True):
    modules = [torch.nn.Linear(input_dim, int(hidden_dim)), torch.nn.ReLU(), torch.nn.Dropout(self.dropout), torch.nn.Linear(int(hidden_dim), output_dim)]
    if last_relu:
      modules.append(torch.nn.ReLU())
    return torch.nn.Sequential(*modules)




# Do not change function signature
def init_model():
  model = MazeGNN().to(device)
  return model




def eval_model(model, dataset, mode=None):
  model.eval()
  acc = 0
  tot_nodes = 0
  tot_graphs = 0
  perf = 0
  gpred = []
  gsol = []

  for step, batch in enumerate(dataset):
    n = len(batch.x)/batch.num_graphs
    with torch.no_grad():
      batch = batch.to(device)
      pred = model(batch, int(n))
    
    if mode == "small":
      if n > 4*4:
          break
    elif mode == "medium":
      if n > 8*8:
          break
    elif mode == "large":
      if n > 16*16:
          break
    
    y_pred = torch.argmax(pred,dim=1)
    tot_nodes += len(batch.x)
    tot_graphs += batch.num_graphs
    
    graph_acc = torch.sum(y_pred == batch.y).item()
    
    acc += graph_acc
    for p in y_pred:
      gpred.append(int(p.item()))
    for p in batch.y:
      gsol.append(int(p.item()))
    if graph_acc == n:
      perf += 1

  gpred = torch.tensor(gpred)
  gsol = torch.tensor(gsol)
  f1score = f1_score(gpred, gsol)

  return f"node accuracy: {acc/tot_nodes:.3f} | node f1 score: {f1score:.3f} | graph accuracy: {perf/tot_graphs:.3}"


  
  
def train_model(model, train_dataset_gen, epochs=22, lr=0.0004):
  dataset = train_dataset_gen(n_samples=850)
  
  criterion = torch.nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)

  val_split = 0.2
  train_size = int(val_split*len(dataset))
  train_loader = DataLoader(dataset[:train_size], batch_size=1, shuffle=True)
  val_set = DataLoader(dataset[train_size:], batch_size = 1)

  model.train()

  worst_loss = -1
  best_model = None

  for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
      optimizer.zero_grad()
      data = data.to(device)

      # could change additional parameters here
      pred = model(data, data.num_nodes)
      loss = criterion(pred, data.y.to(torch.long))
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      running_loss += loss.item()
    ss = eval_model(model, val_set)
    graph_val = float((ss.split(" ")[-1]))
    print(f'Epoch: {epoch + 1} loss: {running_loss / len(train_loader.dataset):.5f} \t {ss}')
    comp = (-graph_val, running_loss)
    if worst_loss == -1 or comp < worst_loss:
      worst_loss = comp
      best_model = deepcopy(model)
      print("store new best model", comp)
    running_loss = 0.0
  
  return best_model