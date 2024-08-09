""" The module contains classes for the Gated Recurrent Unit (GRU).

Classes
-------
GRUCell
    A class for one GRU cell.
GRU
    A class for GRU neural network.
"""


import torch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


import copy
import numpy as np


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')



class GRUCell(nn.Module):
    """ Gated Recurrent Unit cell.
    """


    def __init__(self, input_size, hidden_size, \
                 task_id, num_layer, device, bias=True):
        """ Constructor.
        Parameters
        ----------
        input_size : int
            Number of input neurons.
        hidden_size : int
            Dimension of a hidden state.
        task_id : int
            Current task identifier.
        num_layer : int
            Number of the current GRU cell.
        device: torch.device ('cpu' or 'cuda')
            The device on which PyTorch model and all torch.
            Tensor are or will be allocated.
        bias : bool
            Use bias or not
        """

        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.num_layer = num_layer

        self.task_id = 0
        self.device = device

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()

        self.base_masks = self._create_masks(num_layer)

        self.tasks_masks = []


    def _create_masks(self, num_layer=0):
        """ The method creates the mask for the current cell.

        Parameters
        ----------
        num_layers : int
            Number of the current GRU cells.

        Returns
        -------
        masks : list
            List of masks for the cell.
        """
        masks = {}
        for name, w in list(self.named_parameters()):
            masks[f'rnn_cell_list.{num_layer}.' + name] = torch.ones_like(w)

        return masks


    def add_mask(self):
        """ The method adds a new mask for a new task.
        """

        self.tasks_masks.append(copy.deepcopy(self.base_masks))


    def reset_parameters(self):
        """ The method initializes the random parameters for the GRU.
        """

        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, input, hx=None, mode=None, alpha=0.95):

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x2h_active_weight = self.x2h.weight * \
            self.tasks_masks[self.task_id][\
                f'rnn_cell_list.{self.num_layer}.x2h.weight'].to(self.device)
        x2h_active_bias = self.x2h.bias * \
            self.tasks_masks[self.task_id][\
                f'rnn_cell_list.{self.num_layer}.x2h.bias'].to(self.device)

        x_t = F.linear(input, weight=x2h_active_weight, bias=x2h_active_bias)
        h2h_active_weight = self.h2h.weight * \
            self.tasks_masks[self.task_id][\
                f'rnn_cell_list.{self.num_layer}.h2h.weight'].to(self.device)
        h2h_active_bias = self.h2h.bias * \
            self.tasks_masks[self.task_id][\
                f'rnn_cell_list.{self.num_layer}.h2h.bias'].to(self.device)
        h_t = F.linear(hx, weight=h2h_active_weight, bias=h2h_active_bias)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        if mode == 'prune':
            beta = 1 - alpha
            x2h_is = (x2h_active_weight.abs()*(input.abs().mean(dim=0))).T
            h2h_is = (h2h_active_weight.abs()*(hx.abs().mean(dim=0))).T

            if (x_reset.abs()/(\
             x_reset.abs()+h_reset.abs())).mean(dim=(0, 1)) < beta:
                x2h_is[:self.hidden_size, :] = 0

            if (h_reset.abs()/(\
             x_reset.abs()+h_reset.abs())).mean(dim=(0, 1)) < beta:
                h2h_is[:self.hidden_size, :] = 0

            if (x_upd.abs()/(\
             x_upd.abs()+h_upd.abs())).mean(dim=(0, 1)) < beta:
                x2h_is[self.hidden_size:2*self.hidden_size, :] = 0

            if (h_upd.abs()/(\
             x_upd.abs()+h_upd.abs())).mean(dim=(0, 1)) < beta:
                h2h_is[self.hidden_size:2*self.hidden_size, :] = 0

            if (x_new.abs()/(\
             x_new.abs()+(reset_gate * h_new).abs())).mean(dim=(0, 1)) < beta:
                x2h_is[2*self.hidden_size:, :] = 0

            if ((reset_gate * h_new).abs() / \
               (x_new.abs()+(reset_gate * h_new).abs())).mean( \
                   dim=(0, 1)) < beta:

                x2h_is[:self.hidden_size, :] = 0
                h2h_is[:self.hidden_size, :] = 0
                h2h_is[2*self.hidden_size:, :] = 0

            if (update_gate*hx).abs().mean( \
             dim=(0, 1))((update_gate*hx).abs() + \
                         ((1 - update_gate)*new_gate).abs()).mean(\
                             dim=(0, 1)) < beta:
                x2h_is[self.hidden_size:2*self.hidden_size, :] = 0
                h2h_is[self.hidden_size:2*self.hidden_size, :] = 0

            if ((1 - update_gate)*new_gate).abs().mean(\
                dim=(0, 1)) / ((update_gate*hx).abs() + \
                               ((1 - update_gate)*new_gate).abs()).mean(\
                                   dim=(0, 1)) < beta:
                x2h_is[2*self.hidden_size:, :] = 0
                h2h_is[2*self.hidden_size:, :] = 0

            return hy, x2h_is, h2h_is

        return hy


class GRU(nn.Module):
    """ Gated Recurrent Unit neural network.
    """


    def __init__(self, input_size, seq_len, \
                 hidden_size, num_layers, output_size, \
                 device, bias=True):
        """ Constructor.
        Parameters
        ----------
        input_size : int
            Number of input neurons.
        seq_len : int
            Input data sequence length.
        hidden_size : int
            Dimension of a hidden state.
        num_layers : int
            Number of GRU cells.
        output_size : int
            Number of output neurons.
        device: torch.device ('cpu' or 'cuda')
            The device on which PyTorch model and all torch.
            Tensor are or will be allocated.
        bias : bool
            Use bias or not        .
        """
        super(GRU, self).__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.device = device

        # self.num_heads = num_heads
        self.num_tasks = 0
        self.task_id = 0

        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_masks = {}

        self.rnn_cell_list, \
            self.rnn_cell_masks = self._make_layers(input_size,
                                                    hidden_size,
                                                    num_layers,
                                                    bias)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.rnn_cell_masks['fc.weight'] = torch.ones(\
            self.output_size, self.hidden_size)
        self.rnn_cell_masks['fc.bias'] = torch.ones(\
            self.output_size)


        self.tasks_masks = []
        self.add_mask(task_id=0)

        self.trainable_mask = copy.deepcopy(self.tasks_masks[0])
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])


    def _make_layers(self, input_size, hidden_size, num_layers, bias):
        """ The method creates layers and masks for the GRU.
        Parameters
        ----------
        input_size : int
            Number of input neurons.
        hidden_size : int
            Dimension of a hidden state.
        num_layers : int
            Number of GRU cells.
        bias : bool
            Use bias or not

        Returns
        -------
        rnn_cell_list : list
            List of GRU cells.
        rnn_cell_masks : list
            List of GRUcells masks.
        """

        rnn_cell_list = nn.ModuleList()
        rnn_cell_masks = {}

        gru_cell = GRUCell(input_size=input_size,
                           hidden_size=hidden_size,
                           task_id=self.task_id,
                           num_layer=0,
                           device=device,
                           bias=bias)
        rnn_cell_list.append(gru_cell)
        rnn_cell_masks.update(gru_cell.base_masks)

        for iter_layer in range(1, num_layers):
            gru_cell = GRUCell(input_size=hidden_size,
                               hidden_size=hidden_size,
                               task_id=self.task_id,
                               num_layer=iter_layer,
                               device=device,
                               bias=bias)

            rnn_cell_list.append(gru_cell)
            rnn_cell_masks.update(gru_cell.base_masks)

        return rnn_cell_list, rnn_cell_masks


    def add_mask(self, task_id, overlap=True):
        """ The method adds a new mask for a new task.

        Parameters
        ----------
        task_id : int
            New task identifier.
        overlap : bool
            Overlapping subnetworks or not
        """

        self.num_tasks += 1
        self.tasks_masks.append(copy.deepcopy(self.rnn_cell_masks))

        for cell in self.rnn_cell_list:
            cell.add_mask()
            for name in cell.tasks_masks[task_id]:
                self.tasks_masks[task_id][name] = \
                    cell.tasks_masks[task_id][name]


    def set_task(self, task_id):
        """ The method activates the subnetwork.

        Parameters
        ----------
        task_id : int
            Task identifier.
        """

        self.task_id = task_id

        for cell in self.rnn_cell_list:
            cell.task_id = task_id


    def set_masks_union(self):
        """ The method sets the union of all masks.
        """
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        for task_id in range(1, self.num_tasks):
            for name in self.rnn_cell_masks:
                self.masks_union[name] = copy.deepcopy(\
                    1*torch.logical_or(self.masks_union[name], \
                                       self.tasks_masks[task_id][name]))


    def set_masks_intersection(self):
        """ The method sets the intersection of all masks.
        """
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])
        for task_id in range(1, self.num_tasks):
            for name in self.rnn_cell_masks:
                self.masks_intersection[name] = copy.deepcopy(\
                    1*torch.logical_and(self.masks_intersection[name], \
                                        self.tasks_masks[task_id][name]))


    def set_trainable_masks(self, task_id):
        """ The method sets a mask for trainable parameters
        for the current task.

        Parameters
        ----------
        task_id : int
            Current task identifier.
        """

        if task_id > 0:
            for name in self.trainable_mask:
                self.trainable_mask[name] = copy.deepcopy(\
                    1*((self.tasks_masks[task_id][name] - \
                        self.masks_union[name]) > 0))
        else:
            self.trainable_mask = copy.deepcopy(self.tasks_masks[task_id])


    def forward(self, input, hx=None):
        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(\
                    self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(\
                    self.num_layers, input.size(0), self.hidden_size))

        else:
            h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])


        for t in range(input.size(1)):
            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](\
                        input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](\
                        hidden[layer - 1], hidden[layer])

                hidden[layer] = hidden_l

            outs.append(hidden_l)


        outputs = []
        for t in range(len(outs)):
            active_weight = self.fc.weight * \
                self.tasks_masks[self.task_id]['fc.weight'].to(self.device)
            active_bias = self.fc.bias * \
                self.tasks_masks[self.task_id]['fc.bias'].to(self.device)
            out = F.linear(outs[t], weight=active_weight, bias=active_bias)

            outputs.append(out.unsqueeze(1))

        out = torch.cat(outputs, dim=1)

        return out


    def save_masks(self, file_name='net_masks.pt'):
        """ The method saves all masks.

        Parameters
        ----------
        file_name : str
            File to save.
        """

        masks_database = {}

        for task_id in range(self.num_tasks):
            masks_database[task_id] = {}
            for name in self.rnn_cell_masks:
                masks_database[task_id][name] = self.tasks_masks[task_id][name]

        torch.save(masks_database, file_name)


    def load_masks(self, file_name='net_masks.pt', num_tasks=1):
        """ The method loads all masks.

        Parameters
        ----------
        file_name : str
            File to load.
        num_tasks : int
            Number of loaded tasks.
        """

        masks_database = torch.load(file_name)
        self.num_tasks = 1
        for task_id in range(num_tasks):
            for cell in self.rnn_cell_list:
                for name in cell.tasks_masks[task_id]:
                    cell.tasks_masks[task_id][name] = \
                        masks_database[task_id][name]
                    self.tasks_masks[task_id][name] = \
                        cell.tasks_masks[task_id][name]

            self.tasks_masks[task_id]['fc.weight'] = \
                masks_database[task_id]['fc.weight']
            self.tasks_masks[task_id]['fc.bias'] = \
                masks_database[task_id]['fc.bias']

            if task_id+1 < num_tasks:
                self._add_mask(task_id+1)

        self.set_masks_union()
        self.set_masks_intersection()
