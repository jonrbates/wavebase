
import math
import torch
from math import pi as PI
from torch import Tensor
SMALL = 0.03125
INF = 1048576


class Specification():
    """Container for LSTM parameter tensors used to initialize custom LSTMs.

    The ``Specification`` class holds per-gate weight and bias tensors which
    can be converted into the keyword arguments expected by the custom
    ``LSTMCell``/``LSTMLayer`` constructors via ``lstm_params()``.
    """
    def __init__(self, *args, **kwargs):
        self.input_size = input_size = kwargs['input_size']
        self.hidden_size = hidden_size = kwargs['hidden_size']

        self.input_gate_ih = torch.zeros(hidden_size, input_size)
        self.input_gate_hh = torch.zeros(hidden_size, hidden_size)
        self.input_gate_bias = torch.zeros(hidden_size)

        self.forget_gate_ih = torch.zeros(hidden_size, input_size)
        self.forget_gate_hh = torch.zeros(hidden_size, hidden_size)
        self.forget_gate_bias = torch.zeros(hidden_size)

        self.cell_ih = torch.zeros(hidden_size, input_size)
        self.cell_hh = torch.zeros(hidden_size, hidden_size)
        self.cell_bias = torch.zeros(hidden_size)

        self.output_gate_ih = torch.zeros(hidden_size, input_size)
        self.output_gate_hh = torch.zeros(hidden_size, hidden_size)
        self.output_gate_bias = torch.zeros(hidden_size)

        self.initial_state = torch.zeros(hidden_size)

    def lstm_params(self):
        """Return a dict of parameters suitable for LSTM construction.

        Returns:
            dict: Keys include ``input_size``, ``hidden_size``, ``weight_ih_init``,
                ``weight_hh_init``, ``bias_ih_init`` and ``bias_hh_init``.
        """
        bias = torch.cat([
            self.input_gate_bias,
            self.forget_gate_bias,
            self.cell_bias,
            self.output_gate_bias
        ])
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'weight_ih_init': torch.cat([
                self.input_gate_ih,
                self.forget_gate_ih,
                self.cell_ih,
                self.output_gate_ih
            ]),
            'weight_hh_init': torch.cat([
                self.input_gate_hh,
                self.forget_gate_hh,
                self.cell_hh,
                self.output_gate_hh
            ]),
            'bias_ih_init': bias,
            'bias_hh_init': torch.zeros_like(bias)
        }


class Random(Specification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        input_size = kwargs['input_size']
        hidden_size = kwargs['hidden_size']

        self.input_gate_ih = torch.randn(hidden_size, input_size)
        self.input_gate_hh = torch.randn(hidden_size, hidden_size)
        self.input_gate_bias = torch.randn(hidden_size)

        self.forget_gate_ih = torch.randn(hidden_size, input_size)
        self.forget_gate_hh = torch.randn(hidden_size, hidden_size)
        self.forget_gate_bias = torch.randn(hidden_size)

        self.cell_ih = torch.randn(hidden_size, input_size)
        self.cell_hh = torch.randn(hidden_size, hidden_size)
        self.cell_bias = torch.randn(hidden_size)

        self.output_gate_ih = torch.randn(hidden_size, input_size)
        self.output_gate_hh = torch.randn(hidden_size, hidden_size)
        self.output_gate_bias = torch.randn(hidden_size)

        self.initial_state = torch.randn(hidden_size)


class SpectralLSTM(Specification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.input_size == 1, "This network is designed for 1-dimensional signals"
        assert self.hidden_size % 7 == 0, "Please use a multiple of 7 for the hidden size"
        assert self.hidden_size > 7, "The smallest acceptable hidden size is 14"
        k = self.hidden_size // 7
        T = k

        # indices
        self.gen_ = gen_ = 0
        self.sum_ = sum_ = 2*k
        self.prod_ = prod_ = 4*k
        self.spec_ = spec_ = 6*k

        # input gate hh
        self.input_gate_hh[sum_:prod_,gen_:sum_] = SMALL * torch.eye(2*k)
        self.input_gate_hh[prod_:spec_,sum_:prod_] = SMALL * torch.eye(2*k)

        # input gate bias
        self.input_gate_bias[gen_:sum_] = INF * torch.ones(2*k)
        self.input_gate_bias[spec_:] = INF * torch.ones(k)

        # forget gate hh
        gamma = math.cos(2 * PI / T)
        self.forget_gate_hh[sum_:prod_, gen_:gen_+1] = INF / (gamma - 1)

        # forget gate bias
        self.forget_gate_bias = -INF * torch.ones(7*k)
        self.forget_gate_bias[sum_:prod_] = - (gamma+1) / (gamma-1) * INF / 2 * torch.ones(2*k)

        # cell ih
        self.cell_ih[sum_:prod_] = SMALL * torch.ones(2*k, 1)

        # cell hh
        alphas = (2 * PI / T) * torch.arange(1, k+1, dtype=torch.float32)
        self.cell_hh[gen_:sum_, gen_:sum_] = torch.block_diag(*(rotation_matrix(alpha) for alpha in alphas))
        self.cell_hh[prod_:spec_, sum_:prod_] = SMALL * torch.eye(2*k)

        p = torch.zeros(2*k**2)
        p[::(2*k+2)] = 1
        p[1::(2*k+2)] = 1
        self.cell_hh[spec_:, prod_:spec_] = p.view(k, 2*k)

        # output gate bias
        self.output_gate_bias = INF * torch.ones(7*k)

        # initial state
        e1 = torch.zeros(2*k,)
        e1[::2] = 1
        self.initial_state[gen_:sum_] = SMALL * e1


def rotation_matrix(x: Tensor):
    """Create a 2x2 rotation matrix for a scalar angle.

    Args:
        x (Tensor): A scalar ``torch.Tensor`` representing the rotation angle.

    Returns:
        Tensor: A ``2x2`` rotation matrix tensor.

    Raises:
        AssertionError: If ``x`` is not a scalar tensor.
    """
    assert x.ndim == 0, "input is expected to be a torch scalar"
    x = x.unsqueeze(0)
    return torch.cat((torch.cos(x), torch.sin(x), -torch.sin(x), torch.cos(x))).view(2, 2)