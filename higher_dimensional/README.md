# Higher-dimensional region-of-attraction prediction

Code for the higher-dimensional section of the S3 manuscript: a four-dimensional
cart-pole and a six-dimensional planar quadrotor, where a regular grid of the
state space is infeasible and the region of attraction must be predicted from a
small random sample of tests.

Two self-contained folders:

- `cartpole_4d/`  the four-dimensional cart-pole (single input)
- `quadrotor_6d/` the six-dimensional planar quadrotor (two inputs)

Trained models and the Monte-Carlo ground-truth caches are included in each
`results_*` folder, so the prediction, certificate, and benchmark scripts run in
seconds to minutes without retraining or re-simulating. Delete a `results_*`
folder to regenerate everything from scratch.

## Setup (macOS, Python 3.13 from python.org)

    python3.13 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

All dependencies ship prebuilt macOS wheels, so no compiler is needed.

## Cart-pole (4D)

    cd cartpole_4d
    python systems_hd.py       # verify the dynamics (eigenvalues, controllability)
    python predict_hd.py       # predict the failure region from 15% of tests
    python conformal_hd.py     # certified false-safe rate
    python benchmark_hd.py     # benchmark vs raw-state, SVM, GP, classical Lyapunov
    python drift_hd.py         # transfer under plant drift
    python lyap_hd.py          # the key feature is a finite-time Lyapunov exponent

To retrain the model yourself (a few minutes), delete `results_hd/` first and run
`python train_hd.py` before the rest.

## Planar quadrotor (6D)

    cd quadrotor_6d
    python systems_quad.py     # verify the dynamics
    python predict_quad.py     # predict the 6D failure region from 15% of tests
    python conformal_quad.py   # certified false-safe rate in 6D
    python benchmark_quad.py   # honest benchmark; the SVM budget sweep takes 10 to 15 minutes

To retrain (a few minutes), delete `results_quad/` first and run
`python train_quad.py` before the rest. The first run of `predict_quad.py`
simulates the 40,000-start ground truth once (about one minute) and caches it.

## What each result shows

- `predict_*`  testing 15% of the states predicts the recover or fail outcome of
  the rest: 98% in 4D, 92% in 6D with the model-feature classifier.
- `conformal_*`  a distribution-free certificate on the false-safe rate that holds
  with 90% confidence across many splits.
- `benchmark_*`  an honest comparison. In 6D a radial-basis SVM on the raw state is
  the best predictor and the engineered features add nothing a strong classifier
  cannot extract; the classical quadratic Lyapunov sublevel set is at chance.
- `drift_hd`  the predictor re-adapts to a drifted plant from a few fresh tests.
- `lyap_hd`  the strongest feature is a finite-time Lyapunov exponent of the
  learned closed loop; its sign is a conservative recovery test with zero
  false-safe.
