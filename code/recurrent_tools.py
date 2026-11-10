"""Full-state maps and Jacobians for PyTorch recurrent cells.

LSTM's autonomous state is (h, c), concatenated here in that order.
These utilities analyze a specified constant input; they do not erase input
history or justify treating a driven network as an autonomous one.
"""
import torch


def state_map(cell, state, fixed_input):
    if isinstance(cell, torch.nn.LSTMCell):
        h, c = state.chunk(2)
        hn, cn = cell(fixed_input, (h, c))
        return torch.cat((hn, cn))
    return cell(fixed_input, state)


def state_jacobian(cell, state, fixed_input):
    return torch.autograd.functional.jacobian(
        lambda s: state_map(cell, s, fixed_input), state, create_graph=False)
