# Data-Efficient Prediction of the Region of Attraction for Learned Koopman Controllers

Course project for EEE 577 (Intelligent Control). The project builds a full
learning-for-control pipeline on two nonlinear benchmarks (a Duffing double well
oscillator and an inverted pendulum), and adds a prediction step: it estimates
where the learned controller will fail, from features of the learned model
alone, after physically testing only a fraction of the state space.

## Summary

A controller built on the Koopman operator can stabilise a nonlinear system from
measured trajectories, but it does not say where it works. Every stabilising
controller for an unstable target has a finite region of attraction (ROA), the
set of starting states from which the closed loop returns to the target. Mapping
that set by brute force is slow in simulation and risky on hardware. This project
shows that the ROA can be predicted from the learned model. A classifier trained
on 35% of a state grid predicts the recover or fail outcome of the other 65% with
97.3% accuracy on the Duffing system and 98.6% on the pendulum, measured against
fully simulated ground truth and averaged over ten random splits.

## Pipeline

The scripts in `src/` run in order and build on each other.

| stage | script | what it does |
|------|--------|--------------|
| systems | `systems.py` | the two nonlinear plants (Duffing, pendulum) |
| simulate | `sim.py` | open loop and sampled-data closed loop on the true plant |
| linearise | `step3_linearize.py` | Jacobian linearisation and zero order hold discretisation |
| baseline | `step4_lqr.py` | discrete LQR from the exact equations, closed on the true plant |
| ground truth | `step5_where_it_breaks.py` | map where the baseline recovers or fails (the ROA) |
| classic model | `step6_koopman_edmdc.py` | classical Koopman with control (EDMDc), the baseline model |
| neural model | `koopman.py`, `step7_train_koopman.py` | neural Koopman lifting, trained with a multi step loss |
| control | `step8_koopman_control.py` | design a controller from the learned model, close the loop |
| prediction | `step9_predict_failure.py` | predict the failure region from features of the learned model |
| validation | `validate_predictor.py` | the three evaluation protocols and the results table |

## Results

- The data driven controller matches the exact equation baseline. Region of
  attraction: 61% vs 57% (Duffing), 25% vs 25% (pendulum), on an 81x81 grid.
- The neural Koopman model predicts about 2.6 times better than classical EDMDc.
- Failure prediction, 81x81 grid, 35% of states tested: 97.3% (Duffing) and
  98.6% (pendulum) accurate on the untested states, with low false safe rates.

Figures are in `figures/`. The full write up is in `report/`.

## Install and run

```bash
pip install -r requirements.txt
cd src
python step4_lqr.py             --system duffing --umax 0.5
python step5_where_it_breaks.py --system duffing --umax 0.5 --grid 81
python step7_train_koopman.py   --system duffing
python step8_koopman_control.py --system duffing --umax 0.5 --grid 81
python step9_predict_failure.py --system duffing --umax 0.5
```

Use `--system pendulum --umax 5` for the pendulum. The first run of a mapping
step is the slow part, because it simulates the plant from a grid of starts, and
the result is cached. Dependencies: numpy, scipy, matplotlib, control, torch,
scikit-learn.

## Repository layout

```
src/        the Python pipeline, in run order
figures/    the generated figures
report/     the manuscript (LaTeX source; build with pdflatex)
```

## Scope and limitations

The failure predictor is a statistical estimate with a measured error, not a
guarantee. The one hard guarantee is a small Lyapunov safe core, and it holds for
the linearised learned model, not the true nonlinear plant. The method
interpolates between tests spread across the space. It does not extrapolate well
into a region with no nearby tests, where accuracy falls to about 89 to 92%. The
rule in practice is to sample across the whole operating envelope. See `report/`
for the full discussion.
