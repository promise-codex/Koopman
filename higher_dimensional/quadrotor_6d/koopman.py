"""Neural Koopman model with control.

Learns an encoder phi that maps the 2D state x into a higher-dimensional lifted
state z, together with matrices A, B such that the lifted dynamics are linear:
z_{k+1} = A z_k + B u_k. The normalized state is placed in the first slots of z,
so decoding reads it back directly. This makes reconstruction exact by
construction and keeps training stable, leaving the learning to the encoder and
to A, B.

The learned (A, B) is the data-driven counterpart of the hand-derived
linearization in step 3, valid over a wider region. Later steps design a
controller on it and map where it fails.
"""
import torch
import torch.nn as nn


class KoopmanControl(nn.Module):
    def __init__(self, n_state=2, n_extra=6, hidden=64, n_input=1, mu=None, sd=None):
        super().__init__()
        self.n_state = n_state
        self.n_extra = n_extra
        self.n_input = n_input
        self.z_dim = n_state + n_extra
        # normalization stats (fixed buffers, not trained)
        self.register_buffer("mu", torch.zeros(n_state) if mu is None else torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.ones(n_state) if sd is None else torch.as_tensor(sd, dtype=torch.float32))
        self.encoder = nn.Sequential(
            nn.Linear(n_state, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_extra),
        )
        self.A = nn.Linear(self.z_dim, self.z_dim, bias=False)
        self.B = nn.Linear(n_input, self.z_dim, bias=False)
        # Start A at the identity and B at zero, so the untrained model predicts
        # no motion. This is a stable starting point and prevents the multi-step
        # rollout from diverging early in training.
        nn.init.eye_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def normalize(self, x):
        return (x - self.mu) / self.sd

    def denormalize(self, xs):
        return xs * self.sd + self.mu

    def lift(self, x):
        """State x -> lifted z = [normalized x, encoder(normalized x)]."""
        xs = self.normalize(x)
        g = self.encoder(xs)
        return torch.cat([xs, g], dim=-1)

    def decode(self, z):
        """Lifted z -> state x (read first n_state slots, denormalize)."""
        return self.denormalize(z[..., : self.n_state])

    def step(self, z, u):
        """One linear step in lifted space."""
        return self.A(z) + self.B(u)

    def rollout(self, x0, u_seq):
        """Predict a whole trajectory: lift x0, step the linear model forward
        using the real input sequence, decode every step.

        x0:    (batch, n_state)
        u_seq: (batch, T, 1)
        returns predicted states (batch, T+1, n_state) and lifted zs.
        """
        z = self.lift(x0)
        xs_pred = [self.decode(z)]
        zs = [z]
        T = u_seq.shape[1]
        for k in range(T):
            z = self.step(z, u_seq[:, k, :])
            zs.append(z)
            xs_pred.append(self.decode(z))
        return torch.stack(xs_pred, dim=1), torch.stack(zs, dim=1)


def multistep_loss(model, X, U, w_latent=1.0):
    """Multi-step prediction loss.

    X: (batch, T+1, n_state)   true states
    U: (batch, T, 1)           inputs
    - prediction: the decoded rollout should match the true states at every step
    - latent consistency: the linear step should match the encoder's own lifting
      of the true next state, which keeps A consistent in lifted space.
    """
    x0 = X[:, 0, :]
    x_pred, z_pred = model.rollout(x0, U)
    pred_err = ((x_pred - X) ** 2).mean()

    # latent consistency: lift the true states, compare to rolled-out z
    z_true = model.lift(X)                       # (batch, T+1, z_dim)
    latent_err = ((z_pred - z_true) ** 2).mean()
    return pred_err + w_latent * latent_err, pred_err.item(), latent_err.item()
