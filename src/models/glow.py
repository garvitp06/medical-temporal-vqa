"""GLOW (Kingma & Dhariwal, 2018) for 2.5D CT patches, with optional additive
context conditioning hooks used by the ContextFlow++-style specialist.

Shapes (default config, input C=3 neighbouring slices, 64x64 patch):
  x: (B,3,64,64)
  level 1: squeeze -> (B,12,32,32), K flow steps, split -> z1 (B,6,32,32), h (B,6,32,32)
  level 2: squeeze -> (B,24,16,16), K steps, split -> z2 (B,12,16,16), h (B,12,16,16)
  level 3: squeeze -> (B,48,8,8),  K steps, z3 = h (top prior)
Every layer is fully convolutional, so inference on full slices (H,W
divisible by 2**L) is valid.

Flow step = ActNorm -> invertible 1x1 conv (LU-parameterised) -> affine coupling.
log p(x) = sum_l log p(z_l) + sum_k log|det J_k|       (change of variables)

Numerical stability: coupling scale s = sigmoid(raw + 2) (bounded in (0,1),
as in the official GLOW code); ActNorm log-scale clamped; LU diagonal kept
as sign * exp(log|s|); log-dets accumulated in float32 per sample.

Conditioning (additive, ContextFlow++ eq. 6 style): each step may receive a
per-sample context dict with tensors added to
  ActNorm  (d_logs, d_bias)     shape (B,C)
  Conv1x1  d_W                  shape (B,C,C)
  Coupling d_st                 shape (B,C)  (added to NN output channels)
When no context is given the model is a plain GLOW.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)


def gaussian_logp(z, mean, logs):
    """Elementwise log N(z; mean, exp(logs)^2)."""
    return -0.5 * (LOG2PI + 2.0 * logs + (z - mean) ** 2 * torch.exp(-2.0 * logs))


def squeeze(x, f=2):
    b, c, h, w = x.shape
    x = x.view(b, c, h // f, f, w // f, f).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * f * f, h // f, w // f)


def unsqueeze(x, f=2):
    b, c, h, w = x.shape
    x = x.view(b, c // (f * f), f, f, h, w).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c // (f * f), h * f, w * f)


class ActNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def _init(self, x):
        m = x.mean(dim=(0, 2, 3), keepdim=True)
        s = x.std(dim=(0, 2, 3), keepdim=True, correction=0).clamp(min=1e-2)
        self.bias.data.copy_(-m)
        self.logs.data.copy_(torch.log(1.0 / (s + 1e-6)))
        self.initialized.fill_(1)

    def _params(self, ctx):
        b, logs = self.bias, self.logs
        if ctx is not None and "an_bias" in ctx:
            b = b + ctx["an_bias"][..., None, None]
            logs = logs + ctx["an_logs"][..., None, None]
        return b, logs.clamp(-8, 8)

    def forward(self, x, ctx=None):
        if not self.initialized and self.training:
            self._init(x)
        b, logs = self._params(ctx)
        y = (x + b) * torch.exp(logs)
        hw = x.shape[2] * x.shape[3]
        ld = logs.sum(dim=(1, 2, 3)) * hw          # (1,) or (B,)
        return y, ld.expand(x.shape[0])

    def reverse(self, y, ctx=None):
        b, logs = self._params(ctx)
        return y * torch.exp(-logs) - b


class InvConv1x1(nn.Module):
    """LU-decomposed invertible 1x1 convolution: W = P L (U + diag(sign*exp(logs)))."""

    def __init__(self, c):
        super().__init__()
        w = torch.linalg.qr(torch.randn(c, c))[0]
        P, L, U = torch.linalg.lu(w)
        s = torch.diag(U)
        self.register_buffer("P", P)
        self.register_buffer("sign_s", torch.sign(s))
        self.register_buffer("lmask", torch.tril(torch.ones(c, c), -1))
        self.register_buffer("eye", torch.eye(c))
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(torch.triu(U, 1))
        self.log_s = nn.Parameter(torch.log(torch.abs(s)))

    def weight(self):
        L = self.L * self.lmask + self.eye
        U = self.U * self.lmask.T + torch.diag(self.sign_s * torch.exp(self.log_s))
        return self.P @ L @ U

    def forward(self, x, ctx=None):
        hw = x.shape[2] * x.shape[3]
        W = self.weight()
        if ctx is not None and "conv_dW" in ctx:
            Wb = W[None] + ctx["conv_dW"]                       # (B,C,C)
            y = torch.einsum("bij,bjhw->bihw", Wb, x)
            ld = torch.linalg.slogdet(Wb)[1] * hw
        else:
            y = F.conv2d(x, W[..., None, None])
            ld = self.log_s.sum().expand(x.shape[0]) * hw
        return y, ld

    def reverse(self, y, ctx=None):
        W = self.weight()
        if ctx is not None and "conv_dW" in ctx:
            Wi = torch.linalg.inv(W[None] + ctx["conv_dW"])
            return torch.einsum("bij,bjhw->bihw", Wi, y)
        return F.conv2d(y, torch.linalg.inv(W)[..., None, None])


class ZeroConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv.weight.data.zero_(); self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, cout, 1, 1))

    def forward(self, x):
        return self.conv(x) * torch.exp(self.scale * 3)


class AffineCoupling(nn.Module):
    def __init__(self, c, hidden, kernel=3):
        super().__init__()
        self.ca = c // 2
        self.nn = nn.Sequential(nn.Conv2d(self.ca, hidden, kernel, padding=kernel // 2), nn.ReLU(inplace=True),
                                nn.Conv2d(hidden, hidden, 1), nn.ReLU(inplace=True),
                                ZeroConv(hidden, 2 * (c - self.ca)))

    def _st(self, xa, ctx):
        h = self.nn(xa)
        if ctx is not None and "cp_st" in ctx:
            h = h + ctx["cp_st"][..., None, None]             # additive conditioning
        raw, t = h[:, 0::2], h[:, 1::2]
        s = torch.sigmoid(raw + 2.0)
        return s, t

    def forward(self, x, ctx=None):
        xa, xb = x[:, :self.ca], x[:, self.ca:]
        s, t = self._st(xa, ctx)
        yb = (xb + t) * s
        logs = torch.log(s)
        self.ld_map = logs.sum(1)                 # (B,H,W) per-position log-det contribution
        ld = logs.flatten(1).sum(1)
        return torch.cat([xa, yb], 1), ld

    def reverse(self, y, ctx=None):
        ya, yb = y[:, :self.ca], y[:, self.ca:]
        s, t = self._st(ya, ctx)
        return torch.cat([ya, yb / s - t], 1)


class FlowStep(nn.Module):
    def __init__(self, c, hidden, kernel=3):
        super().__init__()
        self.actnorm = ActNorm(c)
        self.conv = InvConv1x1(c)
        self.coupling = AffineCoupling(c, hidden, kernel)
        self.c = c

    def forward(self, x, ctx=None):
        x, l1 = self.actnorm(x, ctx)
        x, l2 = self.conv(x, ctx)
        x, l3 = self.coupling(x, ctx)
        return x, l1 + l2 + l3

    def reverse(self, y, ctx=None):
        y = self.coupling.reverse(y, ctx)
        y = self.conv.reverse(y, ctx)
        return self.actnorm.reverse(y, ctx)


class SplitPrior(nn.Module):
    """Split half the channels off as z_l with prior N(mean(h), exp(logs(h)))."""

    def __init__(self, c):
        super().__init__()
        self.prior = ZeroConv(c // 2, c)

    def forward(self, x):
        h, z = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2:]
        p = self.prior(h)
        mean, logs = p[:, 0::2], p[:, 1::2]
        return h, z, gaussian_logp(z, mean, logs)

    def reverse(self, h, z=None, temperature=1.0):
        p = self.prior(h)
        mean, logs = p[:, 0::2], p[:, 1::2]
        if z is None:
            z = mean + torch.exp(logs) * torch.randn_like(mean) * temperature
        return torch.cat([h, z], 1)


class Glow(nn.Module):
    """squeeze_input=False + levels=1 gives a single-scale flow on feature maps
    (FastFlow/CFLOW-style); alt_kernels alternates 3x3 and 1x1 coupling convs."""

    def __init__(self, in_channels=3, levels=3, steps=6, hidden=64, squeeze_input=True, alt_kernels=False):
        super().__init__()
        self.levels, self.steps, self.do_squeeze = levels, steps, squeeze_input
        self.flows = nn.ModuleList()
        self.splits = nn.ModuleList()
        c = in_channels
        self.step_channels = []
        for lvl in range(levels):
            if squeeze_input:
                c *= 4
            ks = [(3 if k % 2 == 0 else 1) if alt_kernels else 3 for k in range(steps)]
            self.flows.append(nn.ModuleList([FlowStep(c, hidden, ks[k]) for k in range(steps)]))
            self.step_channels.append(c)
            if lvl < levels - 1:
                self.splits.append(SplitPrior(c))
                c //= 2
        self.top_c = c
        self.top_mean = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.top_logs = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x, contexts=None, return_maps=False):
        """Returns (log_px per sample [nats], dict with per-level latent log-density maps).

        contexts: optional list (levels) of lists (steps) of ctx dicts."""
        logdet = torch.zeros(x.shape[0], device=x.device)
        logpz = torch.zeros_like(logdet)
        maps, sum_maps, ld_maps = [], [], []
        h = x
        for lvl in range(self.levels):
            h = self._sq(h)
            ldm = 0.0
            for k, step in enumerate(self.flows[lvl]):
                ctx = contexts[lvl][k] if contexts is not None else None
                h, ld = step(h, ctx)
                logdet = logdet + ld
                if return_maps:
                    ldm = ldm + step.coupling.ld_map
            if lvl < self.levels - 1:
                h, z, lp = self.splits[lvl](h)
            else:
                z = h
                lp = gaussian_logp(z, self.top_mean, self.top_logs)
            logpz = logpz + lp.flatten(1).sum(1)
            if return_maps:
                maps.append(lp.mean(1))          # (B,h_l,w_l): mean log-density per channel
                sum_maps.append(lp.sum(1)); ld_maps.append(ldm)
        out = dict(logpz=logpz, logdet=logdet)
        if return_maps:
            out["latent_logp_maps"] = maps
            out["latent_logp_sum_maps"] = sum_maps      # summed over channels
            out["coupling_logdet_maps"] = ld_maps       # per-position coupling log-det
        return logpz + logdet, out

    def _sq(self, h):
        return squeeze(h) if self.do_squeeze else h

    def _unsq(self, h):
        return unsqueeze(h) if self.do_squeeze else h

    @torch.no_grad()
    def reverse(self, zs=None, shape=None, contexts=None, temperature=0.7):
        """Invert latents back to x. ``zs`` is the list returned by encode()."""
        if zs is None:
            b, c, H, W = shape
            f = 2 ** self.levels if self.do_squeeze else 1
            h = self.top_mean + torch.exp(self.top_logs) * temperature * torch.randn(
                b, self.top_c, H // f, W // f, device=self.top_mean.device)
        else:
            h = zs[-1]
        for lvl in reversed(range(self.levels)):
            if lvl < self.levels - 1:
                h = self.splits[lvl].reverse(h, None if zs is None else zs[lvl], temperature)
            for k in reversed(range(self.steps)):
                ctx = contexts[lvl][k] if contexts is not None else None
                h = self.flows[lvl][k].reverse(h, ctx)
            h = self._unsq(h)
        return h

    def encode(self, x, contexts=None):
        zs, h = [], x
        for lvl in range(self.levels):
            h = self._sq(h)
            for k, step in enumerate(self.flows[lvl]):
                h, _ = step(h, contexts[lvl][k] if contexts is not None else None)
            if lvl < self.levels - 1:
                h, z, _ = self.splits[lvl](h)
                zs.append(z)
        zs.append(h)
        return zs


def nll_per_dim(log_px, x):
    """Negative log-likelihood in nats per dimension."""
    return -log_px / x[0].numel()


def latent_anomaly_map(out, size):
    """Spatial anomaly map: negative mean latent log-density at every scale,
    bilinearly upsampled to ``size`` and averaged across scales.

    Documented approximation: the Jacobian log-det is a global (per-sample)
    term and is not localised; only the latent density term is spatial. This is
    the same family of scoring used by flow-based AD methods (DifferNet/CFLOW).
    """
    acc = None
    for m in out["latent_logp_maps"]:
        up = F.interpolate((-m)[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]
        acc = up if acc is None else acc + up
    return acc / len(out["latent_logp_maps"])


def position_nll_map(out):
    """Per-position negative log-likelihood for a single-scale (non-squeezed)
    flow: -(sum_c log N(z_c) + sum_k log s_k) at each position (FastFlow-style).
    Exact for 1x1 couplings; with 3x3 couplings positions interact, so it is
    the standard per-position approximation. ActNorm/1x1-conv log-dets are
    position-independent constants and are omitted."""
    return -(out["latent_logp_sum_maps"][-1] + out["coupling_logdet_maps"][-1])
