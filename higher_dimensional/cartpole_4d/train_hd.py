"""Train the neural Koopman model for a higher-dimensional system (cart-pole)."""
import argparse
import os
import time

import numpy as np
import torch

from data_hd import make_dataset
from koopman import KoopmanControl, multistep_loss
from systems_hd import get_system

p = argparse.ArgumentParser()
p.add_argument("--system", default="cartpole")
p.add_argument("--Ts", type=float, default=0.05)
p.add_argument("--extra", type=int, default=16)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--ntraj", type=int, default=1500)
p.add_argument("--horizon", type=int, default=8)
p.add_argument("--uamp", type=float, default=5.0)
p.add_argument("--hold", type=int, default=2)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
plant = get_system(args.system)

print("generating data...")
t0 = time.time()
X, U = make_dataset(plant, n_traj=args.ntraj, horizon=args.horizon, Ts=args.Ts,
                    u_amp=args.uamp, hold=args.hold, seed=args.seed)
Xv, Uv = make_dataset(plant, n_traj=300, horizon=args.horizon, Ts=args.Ts,
                      u_amp=args.uamp, hold=args.hold, seed=args.seed + 999)
print(f"data {X.shape} in {time.time() - t0:.1f}s")

mu = X.reshape(-1, X.shape[-1]).mean(0)
sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
Xt, Ut = torch.tensor(X, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)
Xvt, Uvt = torch.tensor(Xv, dtype=torch.float32), torch.tensor(Uv, dtype=torch.float32)

model = KoopmanControl(n_state=X.shape[-1], n_extra=args.extra, hidden=args.hidden, mu=mu, sd=sd)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=60, gamma=0.5)
print(f"training z_dim={model.z_dim} ({X.shape[-1]} state + {args.extra} learned)")
t0 = time.time()
for ep in range(1, args.epochs + 1):
    model.train()
    opt.zero_grad()
    loss, pe, le = multistep_loss(model, Xt, Ut)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if ep % 50 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            _, vpe, vle = multistep_loss(model, Xvt, Uvt)
        print(f"epoch {ep:4d}  train pred={pe:.4e}  val pred={vpe:.4e}  val latent={vle:.4e}")

os.makedirs("results_hd", exist_ok=True)
A = model.A.weight.detach().numpy()
B = model.B.weight.detach().numpy()
np.savez(f"results_hd/koopman_{plant.name}.npz", A=A, B=B, mu=mu, sd=sd,
         n_state=X.shape[-1], n_extra=args.extra, hidden=args.hidden, Ts=args.Ts)
torch.save(model.state_dict(), f"results_hd/koopman_{plant.name}.pt")

model.eval()
with torch.no_grad():
    xp, _ = model.rollout(Xvt[:, 0, :], Uvt)
    rmse = torch.sqrt(((xp - Xvt) ** 2).mean()).item()
    naive = torch.sqrt(((Xvt[:, :1, :] - Xvt) ** 2).mean()).item()
print(f"\ntrained in {time.time() - t0:.1f}s")
print(f"VAL multi-step RMSE = {rmse:.4f}   naive(no-motion) = {naive:.4f}   ({naive / rmse:.1f}x better)")
print(f"max |eig(A)| = {np.abs(np.linalg.eigvals(A)).max():.3f}  (open-loop)")
print(f"saved results_hd/koopman_{plant.name}.npz and .pt")
