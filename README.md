# wavebase

Wavebase studies algorithmic stability in gated RNNs by constructing an LSTM that exactly computes the power spectrum of a 1-dimensional signal.

The core object is a hand specified LSTM whose recurrent dynamics implement a bank of oscillators, accumulate signal projections, and output power spectrum features with zero error at initialization. The experiment asks whether gradient descent preserves this exact algorithm, deforms it within an equivalence class, or drifts toward a different representation.

The main benchmark lives in:
```text
scripts/benchmark.py
```
This runs several encoders on synthetic sinusoidal data and logs both training loss and internal weight structure.

Originally presented as a research poster at Google (Chelsea), February
2018.

## What to look for

The SpectraLSTM is initialized to an exact solution, so its training loss is flat at zero. Standard LSTMs start with nonzero loss and decrease under optimization.

## Hidden state structure

The constructed LSTM partitions the hidden state into structured blocks implementing oscillation, accumulation, and spectral pooling. The most informative single visualization is the recurrent cell weight matrix at initialization.

<p align="center"> <img src="images/weight_hh_init.png" width="400" alt="weight_hh_cell at initialization"> </p> <p align="center"> Figure 1. Recurrent cell weights (weight_hh_cell) at initialization showing oscillator, accumulator, and spectral blocks. </p>

This matrix serves as a reference algorithmic configuration. Training
experiments track deviation from this structure.

## Target computation

For each window $x_{j:j+T-1}$ of length $T$, the model computes
```math
P_j[k] = | \sum_{t=0}^{T-1} x_{j+t} \, e^{-2 \pi i k t/T} |^2, \; k = 0, \ldots, T-1.
```

The sequence ${P_j[k]}$ forms a spectrogram over non-overlapping windows.

The construction realizes this computation through real-valued recurrent
oscillator dynamics rather than an explicit FFT.

## Exact construction

The LSTM hidden state is partitioned into structured blocks:

-   `gen` — block-diagonal planar rotations generating Fourier basis
    vectors
-   `sum` — windowed accumulation of Fourier coefficients
-   `prod` — elementwise squaring of real and imaginary components
-   `spec` — linear aggregation into $|z_k|^2$

The recurrent weight matrix contains explicit rotation subblocks with
period T, enforcing oscillator dynamics that correspond to complex
exponentials under the identification $\mathbb{C} \cong \mathbb{R}^2$.

The exact parameterization is defined in [test/test_specifications.py](test/test_specifications.py).
Hidden size must be a multiple of 7 due to block partitioning.


## Results
<p align="center"> <img src="images/train_loss.png" width="400" alt="Training loss curves"> </p> <p align="center"> Figure 2. Training loss. SpectraLSTM stays at zero loss, while standard LSTMs decrease from nonzero loss. </p>


## Optimization experiments

The primary experiment is implemented in:
```
scripts/benchmark.py
```
We compare:

1.  Exact spectral initialization
2.  The same model trained with gradient descent
3.  Randomly initialized LSTM baselines

on synthetic sinusoidal signals.

Learning curves measure convergence and structural drift, allowing
inspection of how gradient descent interacts with a known correct
recurrent algorithm.

## Implementation details

-   Input size must be 1 (scalar time series).
-   Hidden size must be divisible by 7.
-   The input gate activation is modified to tanh to permit signed
    gating required by the construction.
-   Hidden indices are exposed as gen_, sum_, prod_, and spec_.

## Installation and quick start

Install dependencies (this project uses PyTorch; see `requirements.txt`):

```bash
pip install -r requirements.txt
```

See ```wavebase/test``` for specifications and correctness tests.

Run the benchmark experiment:

```bash
python scripts/benchmark.py
```

## What this is not

This repository is not intended as a practical alternative to FFT
implementations. It does not claim improved computational complexity,
efficiency, or performance over classical spectral algorithms. The
purpose of this work is to provide a controlled setting in which a
gated recurrent network is initialized at an exact algorithmic solution,
enabling empirical study of optimization dynamics and representational
stability.
