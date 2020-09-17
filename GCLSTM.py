import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from modules import FilterLinear
import math
import numpy as np

class GCLSTM(nn.Module):

    def __init__(self, K, A, feature_size, Clamp_A=True):
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            feature_size: the dimension of features
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(GCLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.K = K
        self.A_list = []  # Adjacency Matrix List

        # normalization
        D_inverse = torch.diag(1 / torch.sum(A, 0))
        norm_A = torch.matmul(D_inverse, A)
        A = norm_A

        A = torch.FloatTensor(A)
        A_temp = torch.eye(feature_size, feature_size)
        for i in range(K):
            A_temp = torch.matmul(A_temp, torch.Tensor(A))
            if Clamp_A:
                # confine elements of A
                A_temp = torch.clamp(A_temp, max=1.)
            # self.A_list.append(torch.mul(A_temp, torch.Tensor(FFR)))
            self.A_list.append(A_temp)

        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList(
            [FilterLinear(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])

        hidden_size = self.feature_size
        input_size = self.feature_size * K

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

    def forward(self, input, Hidden_State, Cell_State):

        x = input

        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)

        combined = torch.cat((gc, Hidden_State), 1)
        # print(gc.size(), Hidden_State.size(), combined.size())
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,
                       torch.mv(Variable(self.A_list[-1], requires_grad=False).cuda(), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * F.tanh(Cell_State)

        return Hidden_State, Cell_State, gc

    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a

    def loop(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        for i in range(time_step):
            Hidden_State, Cell_State, gc = self.forward(torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State)
        return Hidden_State, Cell_State

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State

    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            Cell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(Hidden_State_data, requires_grad=True)
            Cell_State = Variable(Cell_State_data, requires_grad=True)
            return Hidden_State, Cell_State