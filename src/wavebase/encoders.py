import torch
import torch.nn as nn
from torch.nn import Parameter
from math import pi as PI
from wavebase.base import Base
from wavebase.lstm import LSTMCell, LSTMLayer
from wavebase.specifications import Random, SpectralLSTM, rotation_matrix


class StandardLSTMPowerSpectrumEncoder(Base):
    """Standard LSTM power spectrum encoder.

    Wraps a `torch.nn.LSTM` followed by a linear projection to `out_size`.
    Accepts a 1-D waveform (shape (batch, seq_len, input_size)) and
    produces a power spectrum representation (shape (batch, 1, out_size)).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.project = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x):
        """Compute the encoded output for an input sequence.

        Args:
            x (Tensor): Input tensor with shape (batch, seq_len, input_size).

        Returns:
            Tensor: Projected output tensor with shape (batch, 1, out_size).
        """
        _, (last_out, _) = self.lstm(x)
        o = self.project(last_out)
        return o


class ExactComplexPowerSpectrumEncoder(Base):
    """Exact complex-valued power spectrum encoder.

    Computes complex rotations for a set of frequencies and returns
    the magnitude (power) contributions for each frequency.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        freq_max = self.out_size
        alphas = torch.arange(1, freq_max+1, dtype=torch.float32)
        self.alphas = Parameter(alphas)

    def forward(self, x):
        """Compute the exact complex power spectrum for the input.

        Args:
            x (Tensor): Input sequence tensor with shape (batch, seq_len, 1).

        Returns:
            Tensor: Power spectrum tensor with shape (batch, 1, freq_max).
        """
        T = x.size(1)
        ones = torch.ones((len(self.alphas), 1), dtype=torch.complex64, device=self.device)
        ua = torch.diag(torch.complex(torch.cos(self.alphas * 2 * PI / T), torch.sin(self.alphas * 2 * PI / T)))
        v = torch.eye(len(self.alphas), dtype=torch.complex64, device=self.device)
        o = torch.zeros((len(self.alphas), 1), dtype=torch.complex64, device=self.device)
        for i in range(T):
            v = v @ ua
            o += x[0, i, 0] * v @ ones
        o = (o * torch.conj(o)) * 4 / T**2
        return o.abs().reshape(1, 1, -1)


class ExactRealPowerSpectrumEncoder(Base):
    """Exact real-valued power spectrum encoder using rotation matrices.

    Equivalent to `ExactComplexPowerSpectrumEncoder` but implemented
    using real block rotations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        freq_max = self.out_size
        alphas = torch.arange(1, freq_max+1, dtype=torch.float32)
        self.alphas = Parameter(alphas)

    def forward(self, x):
        """Compute the exact real power spectrum for the input.

        Args:
            x (Tensor): Input sequence tensor with shape (batch, seq_len, 1).

        Returns:
            Tensor: Power spectrum tensor with shape (batch, 1, freq_max).
        """
        T = x.size(1)
        f = len(self.alphas)
        e1 = torch.zeros((2*f, 1), device=self.device)
        e1[::2] = 1
        p = torch.zeros((2*f**2), device=self.device)
        p[::(2*f+2)] = 1
        p[1::(2*f+2)] = 1
        p = p.view(f, 2*f)
        ua = torch.block_diag(*(rotation_matrix(alpha) for alpha in self.alphas))
        v = torch.eye(2 * f, device=self.device)
        o = torch.zeros((2*f, 1), device=self.device)
        for i in range(T):
            v = v @ ua
            o += x[0, i, 0] * v @ e1
        o = p @ (o * o) * 4 / T**2
        return o.reshape(1, 1, -1)


class BlockLSTMPowerSpectrumEncoder(Base):
    """Block LSTM power spectrum encoder.

    Builds an LSTM from a `Random` block specification (randomized
    block weights) and projects the output to a power spectrum vector.

    Note:
        This encoder sets the LSTM input-gate activation via the
        specification (`ingate_act`) to ``torch.tanh`` by default. The
        custom `LSTMCell` default is ``torch.sigmoid``, so overriding it
        to ``tanh`` allows negative gate values and changes the update
        dynamics. Set `ingate_act` in the spec to restore sigmoid
        behavior or supply a different activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = Random(input_size=self.input_size, hidden_size=self.hidden_size)
        params = p.lstm_params()
        params['input_size'] = self.input_size
        params['hidden_size'] = self.hidden_size
        params['ingate_act'] = torch.tanh
        self.blstm = LSTMLayer(LSTMCell, **params)
        self.project = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x):
        """Encode the input using the block LSTM and project the output.

        Args:
            x (Tensor): Input tensor with shape (batch, seq_len, input_size).

        Returns:
            Tensor: Projected output tensor with shape (batch, 1, out_size).
        """
        last_out, _ = self.blstm(x)
        o = self.project(last_out)
        return o


class SpectralLSTMPowerSpectrumEncoder(Base):
    """Spectral LSTM power spectrum encoder.

    Constructs the LSTM from a `SpectralLSTM` specification designed
    to extract spectral components from 1-D signals and projects the
    last hidden state to a power spectrum output.

    Note:
        The constructed specification sets `ingate_act` to ``torch.tanh``
        by default. Because `LSTMCell` uses ``torch.sigmoid`` as its
        default input-gate activation, this override permits negative
        gating and alters the LSTM's update behavior. To change this,
        provide a different `ingate_act` in the specification.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = SpectralLSTM(input_size=self.input_size, hidden_size=self.hidden_size)
        params = p.lstm_params()
        params['input_size'] = self.input_size
        params['hidden_size'] = self.hidden_size
        params['ingate_act'] = torch.tanh
        self.blstm = LSTMLayer(LSTMCell, **params)

        for attr in ('gen_', 'sum_', 'prod_', 'spec_'):
            setattr(self.blstm, attr, getattr(p, attr))

    def forward(self, x):
        """Encode the input sequence and return projected spectral output.

        Args:
            x (Tensor): Input tensor with shape (batch, seq_len, 1).

        Returns:
            Tensor: Projected output tensor with shape (batch, 1, out_size).
        """
        last_out, _ = self.blstm(x)
        o = last_out[:, :, self.blstm.spec_:]
        return o


class SignalEncoder():
    """Factory returning a power spectrum encoder instance.
    """

    def __new__(self, *args, **kwargs):
        model = kwargs.pop('model')
        # Standard LSTM (vanilla)
        if model in ("standard_lstm", "standard-lstm", "lstm"):
            return StandardLSTMPowerSpectrumEncoder(*args, **kwargs)
        # Exact complex spectral encoder
        elif model in ("exact_complex", "exact-complex"):
            return ExactComplexPowerSpectrumEncoder(*args, **kwargs)
        # Exact real spectral encoder
        elif model in ("exact_real", "exact-real"):
            return ExactRealPowerSpectrumEncoder(*args, **kwargs)
        # Block / random-initialized LSTM
        elif model in ("block_lstm", "block-lstm"):
            return BlockLSTMPowerSpectrumEncoder(*args, **kwargs)
        # Spectral LSTM built from SpectralLSTM spec
        elif model in ("spectral_lstm", "spectral-lstm", "lstm_power_spectrum", "lstm-power-spectrum"):
            return SpectralLSTMPowerSpectrumEncoder(*args, **kwargs)
        else:
            raise NotImplementedError(f"Unknown encoder model: {model}")
