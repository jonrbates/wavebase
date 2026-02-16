import torch
import torch.jit as jit
from torch import Tensor
from torch.nn import Parameter
from typing import Tuple, Optional, Union


class LSTMCell(jit.ScriptModule):
    """Custom LSTM cell implementation exposed as a ScriptModule.

    This cell mirrors the gate ordering used by PyTorch's LSTM and
    accepts optional initial weight/bias tensors for deterministic
    initialization in tests and experiments.

    Adapted from
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
        weight_ih_init (Tensor, optional): Initial input-to-hidden weights.
        weight_hh_init (Tensor, optional): Initial hidden-to-hidden weights.
        bias_ih_init (Tensor, optional): Initial input biases.
        bias_hh_init (Tensor, optional): Initial hidden biases.
        ingate_act (callable, optional): Activation for input gate.
    """
    def __init__(self,
        input_size,
        hidden_size,
        weight_ih_init=None,
        weight_hh_init=None,
        bias_ih_init=None,
        bias_hh_init=None,
        ingate_act=torch.sigmoid
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if weight_ih_init is None:
            weight_ih_init = torch.randn(4 * hidden_size, input_size)

        if weight_hh_init is None:
            weight_hh_init = torch.randn(4 * hidden_size, hidden_size)

        if bias_ih_init is None:
            bias_ih_init = torch.randn(4 * hidden_size, )

        if bias_hh_init is None:
            bias_hh_init = torch.randn(4 * hidden_size, )

        self.weight_ih = Parameter(weight_ih_init)
        self.weight_hh = Parameter(weight_hh_init)
        self.bias_ih = Parameter(bias_ih_init)
        self.bias_hh = Parameter(bias_hh_init)

        self.ingate_act = ingate_act

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Run one step of the LSTM cell.

        Args:
            input (Tensor): Input tensor for current timestep with shape (batch, input_size).
            state (Tuple[Tensor, Tensor]): Tuple of hidden and cell states (hx, cx) each shaped (batch, hidden_size).

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: Output tensor and new state tuple (hy, (hy, cy)).
        """
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = self.ingate_act(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    """RNN layer that iterates a cell module over time.

    Adapted from
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py

    Args:
        cell: Callable that returns a ``ScriptModule`` LSTM cell when called with the provided kwargs.
        *args: Positional args forwarded to the cell constructor.
        **kwargs: Keyword args forwarded to the cell constructor. Must include ``input_size`` and ``hidden_size``.
    """
    def __init__(self, cell, *args, **kwargs):
        super().__init__()
        self.input_size = kwargs['input_size']
        self.hidden_size = kwargs['hidden_size']
        self.cell = cell(*args, **kwargs)

    @jit.script_method
    def forward(self,
        input: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
        return_outputs: bool = False
    ) -> Union[Tuple[Tensor, Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]:

        """Run the cell across the input sequence.

        Args:
            input (Tensor): Input sequence with shape (batch, seq_length, input_size) or (seq_length, input_size).
            state (optional): Initial state tuple (h0, c0). If ``None`` zeros are used.
            return_outputs (bool): If True, returns the stacked outputs across time and the final state. Otherwise returns only the final state.

        Returns:
            If ``return_outputs`` is True, returns ``(outputs, state)`` where ``outputs`` has shape (batch, seq_length, hidden_size).
            Otherwise returns ``state`` which matches PyTorch LSTM state shapes.
        """
        # implement for batch_first=True
        if input.ndim == 3:
            batch_size, seq_length, in_size = input.shape
        elif input.ndim == 2:
            seq_length, in_size = input.shape
            batch_size = 1
        else:
            raise ValueError("input must be 2 or 3 dimensional tensor")

        assert in_size == self.input_size

        input = input.reshape(batch_size, seq_length, in_size)

        if state is None:
            device = input.device
            zeros = torch.zeros(batch_size, self.hidden_size, device=device)
            state = (zeros, zeros)

        h0, c0 = state
        if h0.ndim == 3:
            h0 = h0.reshape(batch_size, self.hidden_size)

        if c0.ndim == 3:
            c0 = c0.reshape(batch_size, self.hidden_size)

        state = h0, c0

        inputs = input.unbind(1)
        outputs = []
        for input in inputs:
            out, state = self.cell(input, state)
            if return_outputs:
                outputs += [out]

        hx, cx = state
        state = hx.reshape(1, batch_size, self.hidden_size), cx.reshape(1, batch_size, self.hidden_size)

        if return_outputs:
            return torch.stack(outputs, dim=1), state
        else:
            return state
