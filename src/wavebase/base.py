import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter


class Base(LightningModule):
    """Base LightningModule used by encoders.

    Provides common training configuration and logging helpers used
    by all encoder models in this package.

    Args:
        input_size (int): Number of input features per timestep. Default 1.
        hidden_size (int): Hidden dimension size used by models. Default 5.
        out_size (int): Output feature dimension. Default 5.
    """
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
        """Perform a training step.

        Args:
            batch: A tuple ``(x, y)`` containing input and target tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        self._log_weights()
        return loss

    def on_train_start(self) -> None:
        """Hook run before the training loop starts.

        Logs the initial training step value to the logger.
        """
        # Save an attractive pre-training weight visualization to disk.
        try:
            self._log_weights_matplotlib(path="logs/weight_hh_train_start.png")
        except Exception:
            pass
        self._log_weights()

    def configure_optimizers(self):
        """Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: A SGD optimizer over model parameters.
        """
        optimizer = optim.SGD(self.parameters(), lr=1e-2)
        return optimizer

    def _log_weights(self):
        """Internal helper to log weight images to the experiment logger.

        This method looks for a parameter named ``blstm.cell.weight_hh`` and
        writes its gate-wise submatrices to TensorBoard as images.
        """
        tb = self.logger.experiment
        global_step = self.global_step
        for name, weight in self.named_parameters():
            if name == "blstm.cell.weight_hh":
                w = weight.detach().cpu().numpy()
                tb.add_image("weight_hh_input", w[:self.hidden_size,:].reshape(1, self.hidden_size, self.hidden_size), global_step)
                tb.add_image("weight_hh_forget", w[self.hidden_size:2*self.hidden_size,:].reshape(1, self.hidden_size, self.hidden_size), global_step)
                tb.add_image("weight_hh_cell", w[2*self.hidden_size:3*self.hidden_size,:].reshape(1, self.hidden_size, self.hidden_size), global_step)
                tb.add_image("weight_hh_output", w[3*self.hidden_size:,:].reshape(1, self.hidden_size, self.hidden_size), global_step)

    def _log_weights_matplotlib(self, path: str) -> None:
        """Save a prettier visualization of `blstm.cell.weight_hh` to disk.

        The plot shows the four LSTM gate submatrices (input, forget,
        cell, output) using a custom blue-white-orange diverging colormap
        as well as a grayscale row for quick contrast.

        Args:
            path: Optional path to save the image. If not provided, the
                file is written to `logs/weight_hh_pretrain.png` in the
                repository root.
        """

        # prepare block indices from the model's spec attached to blstm
        blstm = getattr(self, 'blstm')
        gen_ = blstm.gen_
        sum_ = blstm.sum_
        prod_ = blstm.prod_
        spec_ = blstm.spec_

        # find the parameter and plot only the 'cell' gate submatrix
        weight = next((w for n, w in self.named_parameters() if n == "blstm.cell.weight_hh"), None)
        w = weight.detach().cpu().numpy()
        h = self.hidden_size
        cell_mat = w[2 * h:3 * h, :]

        # custom diverging colormap: blue -> white -> orange
        colors = ["#1E90FF", "#FFFFFF", "#FF8C00"]
        cmap = LinearSegmentedColormap.from_list("blue_white_orange", colors)

        # create single colored plot for the cell gate with equal scaling
        # render in xkcd/sketch style
        with plt.xkcd():
            # Some xkcd-style fonts do not include the Unicode minus (U+2212).
            # Force matplotlib to use the ASCII hyphen-minus for negative
            # signs so minus symbols remain visible.
            mpl.rcParams['axes.unicode_minus'] = False
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            im = ax.imshow(cell_mat, aspect="equal", cmap=cmap)
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # Force ASCII hyphen-minus for negative ticks (xkcd font may lack U+2212)
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.6g'))
            cbar.update_ticks()
            # Ensure the colorbar tick labels use ASCII hyphen-minus
            ticks = cbar.get_ticks()
            labels = []
            for t in ticks:
                s = ("%.6g" % t)
                s = s.replace('\u2212', '-')
                labels.append(s)
            # Apply the explicit string labels to avoid missing minus glyphs
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(labels)

            ncols = cell_mat.shape[1]
            bounds = [
                (gen_, sum_, 'gen'),
                (sum_, prod_, 'sum'),
                (prod_, spec_, 'prod'),
                (spec_, self.hidden_size, 'spec'),
            ]

            # draw partition lines and place both row and column labels in one pass
            for start, end, label in bounds:
                # place row label to the left of the plot (outside)
                y_center = (start + end) / 2 - 0.5
                ax.text(-1.0, y_center, label, va='center', ha='right', color='black', fontweight='bold', zorder=11, clip_on=False)

                # place column label above the plot (outside)
                x_center = (start + end) / 2 - 0.5
                ax.text(x_center, 0, label, va='bottom', ha='center', color='black', fontweight='bold', zorder=11, clip_on=False)

                if start == 0:
                    continue  # skip lines at the very edge of the plot

                # horizontal line at the start of this block
                ax.hlines(start - 0.5, xmin=-0.5, xmax=ncols - 0.5, color='black', linewidth=2.0, linestyle='--', zorder=10, clip_on=False)
                # vertical line at the start of this block
                ax.vlines(start - 0.5, ymin=-0.5, ymax=self.hidden_size - 0.5, color='black', linewidth=2.0, linestyle='--', zorder=10, clip_on=False)

            # make room for outside labels
            plt.subplots_adjust(top=0.88, left=0.12)

        plt.tight_layout()

        if path is None:
            path = "logs/weight_hh_pretrain.png"
        try:
            fig.savefig(path, dpi=150)
        finally:
            plt.close(fig)
        return