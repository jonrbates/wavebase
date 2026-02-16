from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from wavebase.generators import Sinusoid
from wavebase.encoders import SignalEncoder


for freq_max in [5]:

    for hidden_size_multiple in [7]:

        for model in ["spectral-lstm"]:

            dataset = Sinusoid(
                n_beats=1,
                n_samples_per_beat=16,
                freq_max=freq_max,
                rps_max=.1
            )

            train_loader = DataLoader(
                dataset,
                batch_size=64
            )

            logger = TensorBoardLogger(
                "logs",
                name=model,
                prefix=f"freq_max={freq_max}; hsm={hidden_size_multiple}"
            )

            trainer = Trainer(
                logger=logger,
                gpus=1,
                max_epochs=1
            )

            encoder = SignalEncoder(
                input_size=1,
                hidden_size=hidden_size_multiple*freq_max,
                out_size=freq_max,
                model=model
            )

            trainer.fit(model=encoder, train_dataloaders=train_loader)