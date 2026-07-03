import torch.nn as nn
from torch_geometric.nn import GINConv,global_add_pool,global_mean_pool
import torch


class GINGraphClf(nn.Module):
    def __init__(self, in_dim,out_dim, hidden_dim=64):
        super().__init__()
        self.conv1=GINConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim),nn.ReLU()))
        self.conv2=GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim),nn.ReLU()))
        
        self.mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim//2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch):
        h1=self.conv1(x, edge_index)
        h2=self.conv2(h1, edge_index)
        # h3=self.conv3(h2, edge_index)

        h1=global_mean_pool(h1,batch)
        h2=global_mean_pool(h2,batch)
        # h3=global_add_pool(h3,batch)    


        x=torch.cat([h1,h2],dim=1)


        return self.mlp(x)


def train_one_epoch(model,loader,optimizer,criterion,device):
    model.train()
    total_loss=0
    for data in loader:
        data=data.to(device)
        optimizer.zero_grad()
        output=model(data.x,data.edge_index,data.batch)
        loss=criterion(output,data.y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*data.num_graphs
    return total_loss/len(loader.dataset)


def evaluate(model , loader,device):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for data in loader:
            data=data.to(device)
            output=model(data.x,data.edge_index,data.batch)
            pred=output.argmax(dim=1)
            correct+=(pred==data.y).sum().item()
            total+=len(data.y)
    return correct/total




def test_evaluation(model, loader, device):
    correct_indices = []
    wrong_indices = []
    global_idx=0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            for i in range(len(pred)):
                if pred[i] == data.y[i]:
                    correct_indices.append({"index":global_idx,"pred":pred[i].item()})
                else:
                    wrong_indices.append({"index":global_idx,"pred":pred[i].item()})
                global_idx += 1
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
    return correct / total,correct_indices,wrong_indices

