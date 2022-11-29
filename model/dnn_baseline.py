import torch.nn as nn
import torch.nn.functional as F
import torch

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_hid = 2048
        self.input = nn.Flatten()

        self.hidden1 = nn.Sequential(nn.Linear(5140, self.n_hid, bias=True),
                                     nn.LayerNorm(self.n_hid),
                                     nn.PReLU())
        self.hidden2 = nn.Sequential(nn.Linear(self.n_hid, self.n_hid),
                                     nn.LayerNorm(self.n_hid),
                                     nn.PReLU())
        self.hidden3 = nn.Sequential(nn.Linear(self.n_hid, self.n_hid),
                                     nn.LayerNorm(self.n_hid),
                                     nn.PReLU())
        self.hidden4 = nn.Sequential(nn.Linear(self.n_hid, self.n_hid),
                                     nn.LayerNorm(self.n_hid),
                                     nn.PReLU())

        self.output = nn.Sequential(nn.Linear(self.n_hid, 514)) #514->(257) / 322->(160)

    def forward(self, x):

        x1 = self.input(x)
        x2 = self.hidden1(x1)
        x3 = self.hidden2(x2)
        x4 = self.hidden3(x3)
        x5 = self.hidden4(x4)
        x6 = self.output(x5)
        x7 = x6.reshape(x6.shape[0],2,1,257)
        return x7.squeeze()