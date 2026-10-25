"""ContextFlow++-INSPIRED generalist-specialist flow for CT anomaly detection.

Verified against: Gudovskiy, Okuno, Nakata, "ContextFlow++: Generalist-
Specialist Flow-based Generative Models with Mixed-Variable Context
Encoding", UAI 2024 (PMLR v244; arXiv:2406.00578; code github.com/gudovskiy/contextflow).

What we reproduce from the paper
  * Generalist f_{theta_g}: a GLOW-type flow trained first on p(v) WITHOUT context,
    then FROZEN.
  * Specialist parameters theta_c trained on (v, c) with ADDITIVE conditioning:
      coupling  [s, t] = NN(v_b) + CN(c)
      ActNorm   scale/shift = generalist + CN(c)
      1x1 conv  W_{g,c} = W_g + W_CN(c)   (eq. 6; log-det via slogdet per sample)
  * Mixed-variable context: continuous variables used directly; discrete
    variables mapped by a surjective encoder. We implement the "learned
    embedding (deterministic)" and "uniform dequantization" options.
  * Unsupervised objective = maximum likelihood of the joint flow (eq. 4) with the
    independence assumption of eq. 5 (context density not modelled -> constant).

Differences / adaptations (this is NOT an exact reproduction)
  * The paper evaluates on MNIST-R, CIFAR-10C, ATM and SMAP; CT anomaly
    detection is our adaptation. The anomaly score/map is the latent-density
    map from :func:`glow.latent_anomaly_map` (the paper gives no CT score).
  * We use a single Gaussian base per latent (no FlowGMM mixture of 8
    components / no semi-supervised classifier term).
  * Context used here (CT-specific, our choice): continuous normalised slice
    position within the lung extent, and the discrete original slice-thickness
    category of the acquisition (an acquisition-protocol variable).
  * CN(c) is zero-initialised so training starts exactly at the generalist.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .glow import Glow


class MixedContextEncoder(nn.Module):
    """Map mixed (continuous + discrete) context to a continuous vector."""

    def __init__(self, n_cont: int, discrete_cardinalities: list[int], emb_dim=8,
                 discrete_mode="embedding"):
        super().__init__()
        self.mode = discrete_mode
        self.cards = list(discrete_cardinalities)
        if discrete_mode == "embedding":
            self.embs = nn.ModuleList([nn.Embedding(k, emb_dim) for k in self.cards])
            self.out_dim = n_cont + emb_dim * len(self.cards)
        elif discrete_mode == "dequantize":       # uniform dequantisation to [0,1)
            self.out_dim = n_cont + len(self.cards)
        else:
            raise ValueError(discrete_mode)

    def forward(self, cont: torch.Tensor, disc: torch.Tensor):
        parts = [cont]
        if self.mode == "embedding":
            parts += [e(disc[:, i]) for i, e in enumerate(self.embs)]
        else:
            u = torch.rand_like(disc, dtype=torch.float32) if self.training else torch.full_like(disc, 0.5, dtype=torch.float32)
            parts.append((disc.float() + u) / torch.tensor(self.cards, device=disc.device).float())
        return torch.cat(parts, 1)


class ContextFlowPP(nn.Module):
    def __init__(self, generalist: Glow, n_cont=1, discrete_cardinalities=(3,), emb_dim=8,
                 cn_hidden=64, discrete_mode="embedding", freeze_generalist=True):
        super().__init__()
        self.g = generalist
        if freeze_generalist:
            for p in self.g.parameters():
                p.requires_grad_(False)
        self.ctx_enc = MixedContextEncoder(n_cont, list(discrete_cardinalities), emb_dim, discrete_mode)
        # number of outputs needed by every flow step
        self.layout = []
        total = 0
        for lvl in range(self.g.levels):
            c = self.g.step_channels[lvl]
            cb = c - c // 2
            for _ in range(self.g.steps):
                sizes = dict(an_bias=c, an_logs=c, conv_dW=c * c, cp_st=2 * cb)
                self.layout.append((lvl, c, sizes))
                total += sum(sizes.values())
        self.cn = nn.Sequential(nn.Linear(self.ctx_enc.out_dim, cn_hidden), nn.SiLU(),
                                nn.Linear(cn_hidden, cn_hidden), nn.SiLU(),
                                nn.Linear(cn_hidden, total))
        nn.init.zeros_(self.cn[-1].weight); nn.init.zeros_(self.cn[-1].bias)
        self.dw_scale = 0.1   # keeps W_g + W_c well conditioned early in training

    def contexts(self, cont, disc):
        vec = self.cn(self.ctx_enc(cont, disc))
        out = [[None] * self.g.steps for _ in range(self.g.levels)]
        i, idx = 0, 0
        for lvl, c, sizes in self.layout:
            ctx = {}
            for name, n in sizes.items():
                v = vec[:, i:i + n]; i += n
                if name == "conv_dW":
                    v = v.view(-1, c, c) * self.dw_scale
                ctx[name] = v
            out[lvl][idx % self.g.steps] = ctx
            idx += 1
        return out

    def forward(self, x, cont, disc, return_maps=False):
        return self.g(x, contexts=self.contexts(cont, disc), return_maps=return_maps)

    def train(self, mode=True):
        super().train(mode)
        self.g.eval()           # frozen generalist: ActNorm must not re-initialise
        return self
