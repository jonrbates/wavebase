import math
import torch
from torch.utils.data import Dataset


class Sinusoid(Dataset):
    """Dataset that generates random sinusoidal signals.

    The Sinusoid dataset generates signals composed of multiple
    frequency components per beat. Each call to ``signal`` samples
    random amplitudes and phases and returns the time-domain signal and its
    power spectrum.

    Args:
        n_samples_per_beat (int): Number of samples per beat. Default 32.
        n_beats (int): Number of beats stacked in the returned signal. Default 1.
        freq_max (int): Maximum frequency index to include. Default 4.
        rps_max (float): Maximum amplitude for random frequency components. Default 0.5.
        max_iter (int): Length reported by ``__len__`` (i.e., dataset size). Default 50_000.

    """

    def __init__(self, **kwargs):
        # set params
        self.n_samples_per_beat = kwargs.get('n_samples_per_beat', 32)
        self.n_beats = kwargs.get('n_beats', 1)
        self.freq_max = kwargs.get('freq_max', 4)
        self.rps_max = kwargs.get('rps_max', .5)
        self.max_iter = kwargs.get('max_iter', 50_000)

        # precompute sinusoids
        unit_interval = torch.arange(self.n_samples_per_beat) / self.n_samples_per_beat
        frequencies = torch.arange(1, self.freq_max+1)
        # precompute 2*pi * (frequency * time) to avoid repeated multiplies in `signal`
        self._two_pi_freq_time = 2 * math.pi * torch.outer(frequencies, unit_interval)

    def __len__(self):
        """Return the nominal length of the dataset.

        The dataset generates samples indefinitely; ``max_iter`` provides
        a practical upper bound for iteration.
        """
        return self.max_iter

    def __getitem__(self, idx):
        """Return a single generated sample.
        """
        return self.signal()

    def signal(self):
        """Generate a random sinusoidal signal x and its power spectrum.
        """
        root_power_spectrum = self.rps_max * torch.rand(size=(self.n_beats, self.freq_max))
        phase = torch.rand(size=(self.n_beats, self.freq_max))
        two_pi_phase = 2 * math.pi * phase.unsqueeze(-1)
        # sinusoids: (n_beats, freq, t)
        sinusoids = torch.cos(self._two_pi_freq_time.unsqueeze(0) + two_pi_phase)
        # weighted sum over frequencies using batched matmul: (n_beats, 1, freq) @ (n_beats, freq, t)
        stacked_signal = torch.matmul(root_power_spectrum.unsqueeze(1), sinusoids).squeeze(1)
        x = stacked_signal.reshape(-1, 1)
        power_spectrum = root_power_spectrum**2
        return x, power_spectrum
