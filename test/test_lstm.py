import torch
from wavebase.specifications import Random
from wavebase.lstm import LSTMLayer, LSTMCell
from helper import TestCase
from torch.nn import LSTM


class TestCustomLSTM(TestCase):

    def test_custom_lstm(self):
        """Assert the LSTMLayer, LSTMCell agree with pytorch native LSTM
        """
        batch = 94
        seq_len = 256
        input_size = 7
        hidden_size = 23

        input = torch.randn(batch, seq_len, input_size)
        h0 = torch.randn(1, batch, hidden_size)
        c0 = torch.randn(1, batch, hidden_size)
        state = (h0, c0)

        # define custom lstm
        custom_lstm = LSTMLayer(LSTMCell, input_size=input_size, hidden_size=hidden_size)

        # get custom lstm outputs
        out, (hx, cx) = custom_lstm(input, state, return_outputs=True)

        # set pytorch native lstm to have same parameters as custom lstm
        lstm = LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        for lstm_param, custom_lstm_param in zip(lstm.all_weights[0], custom_lstm.parameters()):
            assert lstm_param.shape == custom_lstm_param.shape
            with torch.no_grad():
                lstm_param.copy_(custom_lstm_param)

        # get pytorch native lstm outputs
        lstm_out, (lstm_hx, lstm_cx) = lstm(input, state)

        self.assertEqualShape(lstm_out, out, msg="Tensors do not have same shape")
        self.assertTensorsClose(lstm_out, out, atol=1e-3, msg="Outputs are not equal")
        self.assertTensorsClose(lstm_hx, hx, atol=1e-3, msg="Final hidden states are not equal")
        self.assertTensorsClose(lstm_cx, cx, atol=1e-3, msg="Final cell states are not equal")


    def test_lstm_params(self):

        batch = 213
        seq_len = 7
        input_size = 19
        hidden_size = 36

        input = torch.randn(batch, seq_len, input_size)
        h0 = torch.randn(1, batch, hidden_size)
        c0 = torch.randn(1, batch, hidden_size)
        state = (h0, c0)

        p = Random(input_size=input_size, hidden_size=hidden_size)
        params = p.lstm_params()

        # define custom lstm
        custom_lstm = LSTMLayer(LSTMCell, **params)

        # get custom lstm outputs
        out, (hx, cx) = custom_lstm(input, state, return_outputs=True)

        # set pytorch native lstm to have same parameters as custom lstm
        lstm = LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        for lstm_param, custom_lstm_param in zip(lstm.all_weights[0], custom_lstm.parameters()):
            assert lstm_param.shape == custom_lstm_param.shape
            with torch.no_grad():
                lstm_param.copy_(custom_lstm_param)

        # get pytorch native lstm outputs
        lstm_out, (lstm_hx, lstm_cx) = lstm(input, state)

        self.assertEqualShape(lstm_out, out, msg="Tensors do not have same shape")
        self.assertTensorsClose(lstm_out, out, atol=1e-3, msg="Outputs are not equal")
        self.assertTensorsClose(lstm_hx, hx, atol=1e-3, msg="Final hidden states are not equal")
        self.assertTensorsClose(lstm_cx, cx, atol=1e-3, msg="Final cell states are not equal")