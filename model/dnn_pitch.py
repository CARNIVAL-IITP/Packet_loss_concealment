import torch.nn as nn

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

        self.pitch_weight1 = nn.Linear(5140, self.n_hid)
        self.pitch_bias1 = nn.Linear(5140, self.n_hid)
        self.pitch_extend = nn.Linear(1,257)

        self.output = nn.Sequential(nn.Linear(self.n_hid, 514))

    def forward(self, x, pitch):

        pitch_mat = self.pitch_extend(pitch)

        xx = self.input(x)
        pit = self.input(pitch_mat)
        pitch_w1 = self.pitch_weight1(pit)
        pitch_b1 = self.pitch_bias1(pit)

        x2 = self.hidden1(xx)
        x2_prime = x2 + (x2 * pitch_w1 + pitch_b1)

        x3 = self.hidden2(x2_prime)
        x4 = self.hidden3(x3)
        x5 = self.hidden3(x4)
        x5_out = self.output(x5)
        x6 = x5_out.reshape(x5_out.shape[0],2,1,257)

        return x6.squeeze()

