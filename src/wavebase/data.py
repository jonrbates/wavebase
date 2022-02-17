import math
import torch
from torch.utils.data import Dataset
    
class Sinusoid(Dataset):    
    
    def __str__(self):
        return """\
        Generator class for random sinusoids with multiple frequencies.

        :param: n_beats 
        :param: freq_max
        :param: n_samples_per_beat
        :param: iter_max 
        
        Note: 
            rps = root power spectrum = frequency amplitudes
            
            A signal can have multiple beats, each beat corresponding
            to a signal of length n_samples_per_beat,
            the signal in each beat being drawn randomly
            from frequencies in range 0 to freq_max.       
       
        """.replace("\\t"," ")
        
    def __init__(self, **kwargs):
        # set params
        self.n_samples_per_beat = kwargs.get('n_samples_per_beat', 32)
        self.n_beats = kwargs.get('n_beats', 1)
        self.freq_max = kwargs.get('freq_max', 4)
        self.rps_max = kwargs.get('rps_max', .5)
        self.max_iter = kwargs.get('max_iter', 1000)
        # precompute sinusoids
        unit_interval = torch.arange(0, 1, step=1/self.n_samples_per_beat)
        frequencies = torch.arange(1, self.freq_max+1, step=1)
        unit_interval_cross_freq = torch.outer(frequencies, unit_interval)  
        self._sinusoids = torch.cos(2*math.pi*unit_interval_cross_freq)
   
    def __len__(self):
        return self.max_iter
            
    def __getitem__(self, idx):
        return self.signal()
    
    def signal(self):
        """Base function for one random sinusoid with multiple frequencies."""        
        rps = self.rps_max * torch.rand(size=(self.n_beats, self.freq_max)) 
        stacked_signal = torch.mm(rps, self._sinusoids)
        x = stacked_signal.reshape(-1, 1)
        ps = rps**2
        return x, ps