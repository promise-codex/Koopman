"""Step 7: train the neural Koopman model and evaluate its prediction accuracy.

Run:  python step7_train_koopman.py --system duffing

The training and validation errors fall over the epochs, and the final plot
shows the model's multi-step prediction tracking the true nonlinear trajectory
over about 1.5 s, from the start state and the input sequence alone. The result
is a linear model that reproduces nonlinear behaviour in lifted coordinates.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import make_dataset
from koopman import KoopmanControl, multistep_loss
from systems import get_system

p = argparse.ArgumentParser()
p.add_argument("--system", default="duffing", choices=["duffing", "pendulum"])
p.add_argument("--Ts", type=float, default=0.05)
p.add_argument("--extra", type=int, default=6, help="number of learned lifting coordinates")
p.add_argument("--epochs", type=int, default=120)
p.add_argument("--horizon", type=int, default=30)
p.add_argument("--ntraj", type=int, default=400)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--no-show", action="store_true")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
plant = get_system(args.system)

# --- data ------------------------------------------------------------------
print("generating trajectories (runs the nonlinear simulator)...")
X, U = make_dataset(plant, n_traj=args.ntraj, horizon=args.horizon, Ts=args.Ts, seed=args.seed)
Xv, Uv = make_dataset(plant, n_traj=100, horizon=args.horizon, Ts=args.Ts, seed=args.seed + 999)

mu = X.reshape(-1, X.shape[-1]).mean(0)
sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
Xt, Ut = torch.tensor(X, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)
Xvt, Uvt = torch.tensor(Xv, dtype=torch.float32), torch.tensor(Uv, dtype=torch.float32)

# --- model -----------------------------------------------------------------
model = KoopmanControl(n_state=X.shape[-1], n_extra=args.extra, mu=mu, sd=sd)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)

print(f"training Koopman model: z_dim={model.z_dim} (={X.shape[-1]} state + {args.extra} learned)")
for ep in range(1, args.epochs + 1):
    model.train()
    opt.zero_grad()
    loss, pe, le = multistep_loss(model, Xt, Ut)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    sched.step()
    if ep % 20 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            vloss, vpe, vle = multistep_loss(model, Xvt, Uvt)
        print(f"epoch {ep:4d}  train pred={pe:.4e}  val pred={vpe:.4e}  val latent={vle:.4e}")

# --- report & save ----------------------------------------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
A = model.A.weight.detach().numpy()
B = model.B.weight.detach().numpy()
eig = np.linalg.eigvals(A)
print(f"\nLearned lifted model: A is {A.shape}, B is {B.shape}")
print(f"max |eig(A)| = {np.abs(eig).max():.3f}  (open-loop, so >1 is expected for an unstable plant)")
np.savez(f"results/koopman_{plant.name}.npz", A=A, B=B, mu=mu, sd=sd,
         n_state=X.shape[-1], n_extra=args.extra, Ts=args.Ts)
torch.save(model.state_dict(), f"results/koopman_{plant.name}.pt")
print(f"saved results/koopman_{plant.name}.npz and .pt")

# --- prediction plot on a held-out trajectory -------------------------------
model.eval()
with torch.no_grad():
    x_pred, _ = model.rollout(Xvt[:1, 0, :], Uvt[:1])
x_pred = x_pred[0].numpy()
x_true = Xv[0]
t = np.arange(args.horizon + 1) * args.Ts

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for d, ax in enumerate(axes):
    ax.plot(t, x_true[:, d], "-", lw=2, label="true (nonlinear sim)")
    ax.plot(t, x_pred[:, d], "--", lw=2, label="Koopman linear prediction")
    ax.set_ylabel(plant.labels[d])
    ax.legend(fontsize=8)
axes[0].set_title(f"{plant.name}: Koopman multi-step prediction vs truth")
axes[1].set_xlabel("time (s)")
fig.tight_layout()
out = f"figures/step7_{plant.name}_koopman_prediction.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")

# average prediction error over the validation set
with torch.no_grad():
    xp, _ = model.rollout(Xvt[:, 0, :], Uvt)
    rmse = torch.sqrt(((xp - Xvt) ** 2).mean()).item()
print(f"\nVALIDATION multi-step RMSE = {rmse:.4f} (in state units; lower is better)")
if not args.no_show:
    plt.show()
