import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import pi as PI
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import Parameter
from torch.utils.data import DataLoader
from wavebase.data import Sinusoid
from wavebase.models.lstm import LSTMCell, LSTMLayer
from wavebase.specifications import Random, Spectralstm, rotation_matrix


class Base(LightningModule):
    def __init__(
        self,    
        input_size = 1,
        hidden_size = 5,
        out_size = 5
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.example_input_array = torch.randn(1, 24, input_size)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)

        tb = self.logger.experiment
        for name, weight in self.named_parameters():
            if name == "blstm.cell.weight_hh":
                w = weight.detach().cpu().numpy()
                tb.add_image("weight_hh_input", w[:self.hidden_size,:].reshape(1, self.hidden_size, self.hidden_size), self.global_step)
                tb.add_image("weight_hh_forget", w[self.hidden_size:2*self.hidden_size,:].reshape(1, self.hidden_size, self.hidden_size), self.global_step)
                tb.add_image("weight_hh_cell", w[2*self.hidden_size:3*self.hidden_size,:].reshape(1, self.hidden_size, self.hidden_size), self.global_step)
                tb.add_image("weight_hh_output", w[3*self.hidden_size:,:].reshape(1, self.hidden_size, self.hidden_size), self.global_step)

        return loss
   
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-2)        
        return optimizer


class LSTM(Base):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.project = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x):        
        _, (last_out, _) = self.lstm(x)        
        o = self.project(last_out)
        return o
 

class ExactComplex(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        freq_max = self.out_size
        alphas = torch.arange(1, freq_max+1, dtype=torch.float32)        
        self.alphas = Parameter(alphas)        

    def forward(self, x):
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


class Exact(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        freq_max = self.out_size
        alphas = torch.arange(1, freq_max+1, dtype=torch.float32)
        self.alphas = Parameter(alphas)
 
    def forward(self, x):
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


class RandomBlockLSTM(Base):
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
        last_out, _ = self.blstm(x)        
        o = self.project(last_out)
        return o


class SpectraLSTM(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = Spectralstm(input_size=self.input_size, hidden_size=self.hidden_size)
        params = p.lstm_params()
        params['input_size'] = self.input_size
        params['hidden_size'] = self.hidden_size
        params['ingate_act'] = torch.tanh
        self.blstm = LSTMLayer(LSTMCell, **params)
        self.project = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x):       
        last_out, _ = self.blstm(x)        
        o = self.project(last_out)
        return o


class SignalEncoder():
    def __new__(self, *args, **kwargs):
        model = kwargs.pop('model')
        if model == "lstm":
            return LSTM(*args, **kwargs)        
        elif model == "exact-complex":
            return ExactComplex(*args, **kwargs)
        elif model == "exact":
            return Exact(*args, **kwargs)
        elif model == "random-lstm":
            return RandomBlockLSTM(*args, **kwargs)
        elif model == "spectra-lstm":
            return SpectraLSTM(*args, **kwargs)
        else:
            raise NotImplementedError("I don't know that one.")


for freq_max in [5, 10, 15]:

    for hidden_size_multiple in [7, 14]:

        for model in ["spectra-lstm", "lstm", "exact-complex"]:

            dataset = Sinusoid(
                n_beats=1,
                n_samples_per_beat=16, 
                freq_max=freq_max, 
                rps_max=.1
            )
            train_loader = DataLoader(
                dataset,
                num_workers=8
            )

            logger = TensorBoardLogger("logs", name=model, prefix=f"freq_max={freq_max}; hsm={hidden_size_multiple}")

            trainer = Trainer(
                logger=logger,
                auto_scale_batch_size="power", 
                gpus=-1, 
                max_epochs=1
            )
            
            encoder = SignalEncoder(
                input_size=1, 
                hidden_size=hidden_size_multiple*freq_max, 
                out_size=freq_max, 
                model=model
            )

            trainer.fit(model=encoder, train_dataloaders=train_loader)