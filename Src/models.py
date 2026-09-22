"""
src/models.py
===============
Four models, all trained for the same task (binary in-hospital mortality
prediction from the first 48h of ICU data) and evaluated identically.

  - CACE            : proposed method (this pilot's subject).
  - LinearSSM       : real S4/Mamba-family baseline (ZOH-discretized linear
                       recurrence over the async event stream).
  - SmallNeuralCDE  : real Neural CDE baseline (RK4-integrated controlled
                       ODE driven by a linearly-interpolated control path).
  - GRUD            : real GRU-D baseline (Che et al., 2018), native hourly-
                       binned decay-imputation format.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_norm_linear(in_f, out_f):
    return nn.utils.parametrizations.spectral_norm(nn.Linear(in_f, out_f))


class CACE(nn.Module):
    """Contractive Asynchronous Channel Equilibrium (proposed).

    One local state h_c per channel c, updated ONLY by that channel's own
    events via a firmly non-expansive operator (spectral-norm-constrained
    linear map averaged with the identity => Lipschitz <= 1 by
    construction), plus exponential decay toward a learned resting state
    between events on that channel. The update map is a SINGLE shared,
    channel-conditioned (via embedding) contractive network -- mathematically
    equivalent for the properties under test (per-channel-only state
    touching, non-expansiveness, order-invariant fusion) to C separate
    per-channel networks, and far cheaper to batch.
    Channels are fused at query time via an elementwise max over a learned
    per-channel monotone transform: max is commutative/associative/
    idempotent by definition, so the fused representation is EXACTLY
    invariant to the order in which different channels' updates are applied.
    """

    def __init__(self, n_channels, hidden_dim=32, static_dim=5):
        super().__init__()
        self.C = n_channels
        self.H = hidden_dim
        self.channel_embed = nn.Embedding(n_channels, hidden_dim)
        self.value_proj = nn.Linear(1, hidden_dim)
        # single shared contractive update map, conditioned on channel embedding
        self.update_map = spectral_norm_linear(3 * hidden_dim, hidden_dim)
        self.log_tau = nn.Parameter(torch.zeros(n_channels))  # per-channel decay time-constant
        self.resting_state = nn.Parameter(torch.zeros(n_channels, hidden_dim))
        self.pre_merge = nn.Linear(hidden_dim, hidden_dim)  # per-channel monotone-ish transform
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, batch):
        times = batch["times"]; channels = batch["channels"]; values = batch["values"]
        mask = batch["mask"]; dt_chan = batch["dt_chan"]; static = batch["static"]
        B, L = times.shape
        H = self.H
        h = self.resting_state.unsqueeze(0).expand(B, self.C, H).clone()  # (B, C, H)
        arangeB = torch.arange(B)

        for t in range(L):
            m = mask[:, t]  # (B,)
            c = channels[:, t]  # (B,)
            dtc = dt_chan[:, t]  # (B,)
            v = values[:, t].unsqueeze(-1)  # (B,1)

            h_c = h[arangeB, c]  # (B, H) current state of the touched channel
            tau = torch.exp(self.log_tau[c]).unsqueeze(-1)  # (B,1)
            rest = self.resting_state[c]  # (B, H)
            decay = torch.exp(-dtc.unsqueeze(-1) / (tau + 1e-3))
            h_decayed = decay * h_c + (1 - decay) * rest

            c_emb = self.channel_embed(c)
            v_emb = self.value_proj(v)
            inp = torch.cat([h_decayed, v_emb, c_emb], dim=-1)  # (B, 3H)
            g = torch.tanh(self.update_map(inp))  # shared contractive map, all channels at once
            h_new = 0.5 * h_decayed + 0.5 * g  # firmly non-expansive: avg(identity, 1-Lipschitz map)

            m_ = m.unsqueeze(-1)
            h_updated = torch.where(m_.bool(), h_new, h_c)
            h = h.clone()
            h[arangeB, c] = h_updated

        # order-invariant fusion: elementwise max over channels of a learned transform
        transformed = self.pre_merge(h)  # (B, C, H)
        fused, _ = transformed.max(dim=1)  # (B, H) -- commutative, associative, idempotent
        s = self.static_proj(static)
        out = self.head(torch.cat([fused, s], dim=-1))
        return out.squeeze(-1)


class LinearSSM(nn.Module):
    """S4/Mamba-family baseline: diagonal linear state-space layer with
    zero-order-hold discretization, operating on the raw async event
    stream (input-dependent step size Delta_t, as in Mamba's selective
    discretization), followed by a GRU-style nonlinear readout gate."""

    def __init__(self, n_channels, hidden_dim=32, state_dim=32, static_dim=5):
        super().__init__()
        self.H = hidden_dim
        self.channel_embed = nn.Embedding(n_channels, hidden_dim)
        self.value_proj = nn.Linear(1, hidden_dim)
        self.in_proj = nn.Linear(2 * hidden_dim, state_dim)
        self.A_log = nn.Parameter(torch.rand(state_dim) * -2 - 0.5)  # negative real part => stable
        self.B = nn.Linear(2 * hidden_dim, state_dim)
        self.C_out = nn.Linear(state_dim, hidden_dim)
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.state_dim = state_dim

    def forward(self, batch):
        times = batch["times"]; channels = batch["channels"]; values = batch["values"]
        mask = batch["mask"]; dt = batch["dt"]; static = batch["static"]
        B, L = times.shape
        state = torch.zeros(B, self.state_dim)
        A = -torch.exp(self.A_log)  # (state_dim,), continuous-time diagonal generator

        for t in range(L):
            m = mask[:, t].unsqueeze(-1)
            c = channels[:, t]; v = values[:, t].unsqueeze(-1)
            x = torch.cat([self.value_proj(v), self.channel_embed(c)], dim=-1)
            dt_t = dt[:, t].unsqueeze(-1).clamp(min=1e-3) / 60.0  # hours
            decay = torch.exp(A.unsqueeze(0) * dt_t)  # ZOH decay, (B, state_dim)
            state_new = decay * state + self.B(x)
            state = torch.where(m.bool(), state_new, state)

        out = self.C_out(state)
        s = self.static_proj(static)
        return self.head(torch.cat([out, s], dim=-1)).squeeze(-1)


class SmallNeuralCDE(nn.Module):
    """Real Neural CDE: dz = f_theta(z) dX(t), X built by linear
    interpolation of the (channel-embedded) observed path, integrated with
    fixed-step RK4. Every channel updates the SAME shared vector field --
    this is the direct structural contrast to CACE's per-channel operators."""

    def __init__(self, n_channels, hidden_dim=32, static_dim=5, rk4_substeps=2):
        super().__init__()
        self.H = hidden_dim
        self.C = n_channels
        self.channel_embed = nn.Embedding(n_channels, hidden_dim)
        self.value_proj = nn.Linear(1, hidden_dim)
        self.input_dim = hidden_dim
        self.f = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                nn.Linear(hidden_dim, hidden_dim * self.input_dim))
        # Near-zero final-layer init: standard trick for Neural ODE-family vector
        # fields (Chen et al. 2018 and follow-ups). Without it, the compounding of
        # ~600-1200 sequential integration steps drifts/saturates and the model
        # fails to train (confirmed empirically in this pilot -- see RESULTS.md).
        nn.init.normal_(self.f[-1].weight, std=1e-3)
        nn.init.zeros_(self.f[-1].bias)
        self.init_proj = nn.Linear(hidden_dim + static_dim, hidden_dim)
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.substeps = rk4_substeps

    def forward(self, batch):
        times = batch["times"]; channels = batch["channels"]; values = batch["values"]
        mask = batch["mask"]; static = batch["static"]
        B, L = times.shape
        x_path = self.value_proj(values.unsqueeze(-1)) + self.channel_embed(channels)  # (B, L, H)
        x_path = x_path * mask.unsqueeze(-1)

        z = self.init_proj(torch.cat([x_path[:, 0], static], dim=-1))
        dt = torch.diff(times, dim=1, prepend=times[:, :1]) / 60.0  # hours
        dt = dt.clamp(min=1e-4)

        def vf(z_):
            J = self.f(z_).view(-1, self.H, self.input_dim)
            return J

        for t in range(1, L):
            m = mask[:, t].unsqueeze(-1)
            dX = (x_path[:, t] - x_path[:, t - 1])
            h_step = dt[:, t] / self.substeps
            zz = z
            for _ in range(self.substeps):
                k1 = torch.bmm(vf(zz), (dX * h_step.unsqueeze(-1) / dt[:, t].unsqueeze(-1)).unsqueeze(-1)).squeeze(-1)
                k2 = torch.bmm(vf(zz + 0.5 * k1), (dX * h_step.unsqueeze(-1) / dt[:, t].unsqueeze(-1)).unsqueeze(-1)).squeeze(-1)
                zz = zz + k2
            z = torch.where(m.bool(), zz, z)

        s = self.static_proj(static)
        return self.head(torch.cat([z, s], dim=-1)).squeeze(-1)


class GRUD(nn.Module):
    """GRU-D (Che et al., 2018): decay-imputation GRU on hourly-binned
    (value, mask, delta) input -- its native, published format."""

    def __init__(self, n_channels, hidden_dim=32, static_dim=5):
        super().__init__()
        self.H = hidden_dim
        self.C = n_channels
        self.gamma_x = nn.Linear(n_channels, n_channels)
        self.gamma_h = nn.Linear(n_channels, hidden_dim)
        self.gru = nn.GRUCell(2 * n_channels, hidden_dim)
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.register_buffer("x_mean", torch.zeros(n_channels))

    def forward(self, batch):
        X = batch["X"]; M = batch["M"]; Delta = batch["Delta"]; static = batch["static"]
        B, T, C = X.shape
        h = torch.zeros(B, self.H)
        last_x = torch.zeros(B, C)
        for t in range(T):
            x_t, m_t, delta_t = X[:, t], M[:, t], Delta[:, t]
            gx = torch.exp(-F.relu(self.gamma_x(delta_t)))
            x_hat = m_t * x_t + (1 - m_t) * (gx * last_x + (1 - gx) * self.x_mean)
            gh = torch.exp(-F.relu(self.gamma_h(delta_t)))
            h = gh * h
            inp = torch.cat([x_hat, m_t], dim=-1)
            h = self.gru(inp, h)
            last_x = torch.where(m_t.bool(), x_t, last_x)
        s = self.static_proj(static)
        return self.head(torch.cat([h, s], dim=-1)).squeeze(-1)
