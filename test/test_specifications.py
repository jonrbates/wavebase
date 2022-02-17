from wavebase.data import Sinusoid
from wavebase.specifications import Spectralstm
from wavebase.models.lstm import LSTMLayer, LSTMCell
from helper import TestCase


class TestSpectraLSTM(TestCase):   

    def test_spectra_lstm(self):
      
        freq_max = 2

        hidden_size = 14
         
        p = Spectralstm(input_size=1, hidden_size=hidden_size)
        params = p.lstm_params()        
        custom_lstm = LSTMLayer(LSTMCell, **params)

        dataset = Sinusoid(
                n_beats=1,
                n_samples_per_beat=16, 
                freq_max=freq_max, 
                rps_max=.1
            )

        x, ps = dataset.signal()

        # get custom lstm outputs
        out, (hx, cx) = custom_lstm(x, return_outputs=True)

        hx = hx.reshape(hidden_size)
        hx_generator = hx[p.gen_:p.sum_]
        hx_sum = hx[p.sum_:p.prod_]
        hx_prod = hx[p.prod_:p.spec_]
        hx_spec = hx[p.spec_:]

        cx = cx.reshape(hidden_size)
        cx_generator = cx[p.gen_:p.sum_]
        cx_sum = cx[p.sum_:p.prod_]
        cx_prod = cx[p.prod_:p.spec_]
        cx_spec = cx[p.spec_:]

        self.atol = 1e-3
        self.assertTensorsClose(ps, cx_spec, atol=1e-2, msg="This does not compute the power spectrum")
        self.assertTensorsClose(ps, hx_spec, atol=1e-2, msg="This does not compute the power spectrum")