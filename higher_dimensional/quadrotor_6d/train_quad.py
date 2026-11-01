"""Train the neural Koopman model of the planar quadrotor (6 states, 2 inputs).

Lifts the 6D state to 6 + n_extra coordinates and fits a linear model with
control, z_{k+1} = A z_k + B u_k, by minimising a multi-step prediction loss
with a latent-consistency term. Saves A, B, the normalisation and the weights.
"""
import os
import time

import numpy as np
import torch

from koopman import KoopmanControl, multistep_loss
from data_quad import make_dataset
from systems_quad import get_system

Ts = 0.05
N_EXTRA = 20
HIDDEN = 128
EPOCHS = 300
BATCH = 128
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs("results_quad", exist_ok=True)

plant = get_system("quadrotor")
print("generating training data...")
t0 = time.time()
X, U = make_dataset(plant, n_traj=2500, horizon=10, Ts=Ts, coll_amp=3.0, diff_amp=0.8, hold=2, seed=SEED)
print(f"  data {X.shape}, inputs {U.shape} in {time.time()-t0:.0f}s")

ntr = int(0.9 * len(X))
Xtr, Utr, Xval, Uval = X[:ntr], U[:ntr], X[ntr:], U[ntr:]
flat = Xtr.reshape(-1, 6)
mu = flat.mean(0)
sd = flat.std(0) + 1e-6

model = KoopmanControl(n_state=6, n_extra=N_EXTRA, hidden=HIDDEN, n_input=2, mu=mu, sd=sd)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=120, gamma=0.3)

Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
Utr_t = torch.tensor(Utr, dtype=torch.float32)
Xval_t = torch.tensor(Xval, dtype=torch.float32)
Uval_t = torch.tensor(Uval, dtype=torch.float32)

ntrain = len(Xtr_t)
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(ntrain)
    for i in range(0, ntrain, BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss, _, _ = multistep_loss(model, Xtr_t[idx], Utr_t[idx])
        loss.backward()
        opt.step()
    sched.step()
    if (epoch + 1) % 50 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            xp, _ = model.rollout(Xval_t[:, 0, :], Uval_t)
            rmse = torch.sqrt(((xp - Xval_t) ** 2).mean()).item()
        print(f"  epoch {epoch+1:3d}   val multi-step RMSE {rmse:.4f}")

# validation metrics and a naive baseline (predict the state never moves)
model.eval()
with torch.no_grad():
    xp, _ = model.rollout(Xval_t[:, 0, :], Uval_t)
    rmse = torch.sqrt(((xp - Xval_t) ** 2).mean()).item()
naive = torch.sqrt(((Xval_t[:, :1, :] - Xval_t) ** 2).mean()).item()
print(f"\nfinal val multi-step RMSE {rmse:.4f}   naive(no-motion) {naive:.4f}   ratio {naive/rmse:.1f}x")

A = model.A.weight.detach().numpy()
B = model.B.weight.detach().numpy()
np.savez("results_quad/koopman_quadrotor.npz", A=A, B=B, sd=sd, mu=mu,
         n_state=6, n_extra=N_EXTRA, hidden=HIDDEN, n_input=2)
torch.save(model.state_dict(), "results_quad/koopman_quadrotor.pt")
print("saved results_quad/koopman_quadrotor.npz + .pt")
