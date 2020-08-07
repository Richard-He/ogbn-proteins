import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


class MetaNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MetaNet, self).__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.drop1 = nn.Dropout(p=0.1)

        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(p=0.1)
        
        self.lin = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, edge_index):
        x1 = F.leaky_relu(self.conv1(x, edge_index))
        x1 = self.drop1(x1)

        x2 = F.leaky_relu(self.conv2(x1, edge_index))
        x2 = self.drop2(x2)

        x = torch.cat([x1, x2], dim=-1)
        x = self.lin(x)

        return torch.sigmoid(x)


class Record(nn.Module):
    def __init__(self, num_nodes, num_classes):
        super(Record, self).__init__()
        self.register_buffer('outputs', torch.zeros(num_nodes, num_classes))
        self.register_buffer('train_loss', torch.ones(num_nodes))
        self.register_buffer('val_loss', torch.ones(num_nodes))

        self.alpha = 0.75

    def update_output(self, n_id, outputs):
        self.outputs[n_id] = outputs

    def update_train_loss(self, n_id, train_loss):
        self.train_loss[n_id] = self.train_loss[n_id] * self.alpha + train_loss * (1 - self.alpha)

    def update_val_loss(self, n_id, val_loss):
        self.val_loss[n_id] = self.val_loss[n_id] * self.alpha + val_loss * (1 - self.alpha)

    def get_record(self, n_id):
        train_loss = self.train_loss[n_id].view(-1, 1)
        train_loss = torch.argsort(train_loss, dim=0).float()

        val_loss = self.val_loss[n_id].view(-1, 1)
        val_loss = torch.argsort(val_loss, dim=0).float()

        outputs = self.outputs[n_id]

        record = torch.cat([train_loss, val_loss, outputs], dim=-1)

        return record
        

        