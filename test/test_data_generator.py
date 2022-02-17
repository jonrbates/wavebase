import torch
from wavebase.data import Sinusoid
from helper import TestCase


class TestSinusoidDataset(TestCase):  
    
    def test_sinusoid_power_spectrum(self):
        """Assert that ps is power spectrum of x.   
        """
        dataset = Sinusoid(
            n_samples_per_beat=32, 
            n_beats=1,
            freq_max=3)
        
        assert dataset.n_beats == 1, "This test is only valid for dataset.n_beats == 1"

        x, ps = dataset.signal()
        x, ps = x.squeeze(), ps.squeeze()
        ps = torch.cat([torch.tensor([0]), ps])

        fourier = torch.fft.fft(x)   # numpy's FFT
        fourier = fourier * (2 / fourier.shape[0])
        s_ = torch.abs(fourier)**2
        s_ = s_[:ps.size(0)]

        self.assertTensorsClose(ps, s_, atol = 1e-6, msg="This is not the power spectrum of the signal.")