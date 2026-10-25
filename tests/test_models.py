import numpy as np
import pytest
import torch

from src.models.contextflow import ContextFlowPP
from src.models.glow import ActNorm, AffineCoupling, Glow, InvConv1x1, latent_anomaly_map, squeeze, unsqueeze
from src.models.vqa import ANSWERS, TASKS, VQAModel, answer_mask, default_tokenizer, keyword_route

torch.manual_seed(0)


def _jac_logdet(fn, x):
    J = torch.autograd.functional.jacobian(lambda v: fn(v.view_as(x)).reshape(-1), x.reshape(-1))
    return torch.linalg.slogdet(J)[1]


@pytest.mark.parametrize("layer", [ActNorm(4), InvConv1x1(4), AffineCoupling(4, 8)])
def test_layer_logdet_matches_bruteforce_jacobian(layer):
    x = torch.randn(1, 4, 3, 3, dtype=torch.float64)
    layer = layer.double()
    if isinstance(layer, AffineCoupling):                 # make the zero-init net non-trivial
        for p in layer.parameters():
            p.data.normal_(0, 0.3)
    layer.train(); layer(x)                               # ActNorm data init
    layer.eval()
    y, ld = layer(x)
    assert ld.item() == pytest.approx(_jac_logdet(lambda v: layer(v)[0], x).item(), abs=1e-6)
    assert torch.allclose(layer.reverse(y), x, atol=1e-8)


def test_squeeze_roundtrip():
    x = torch.randn(2, 3, 8, 8)
    assert squeeze(x).shape == (2, 12, 4, 4) and torch.equal(unsqueeze(squeeze(x)), x)


def test_glow_shapes_invertibility_and_full_logdet():
    g = Glow(in_channels=1, levels=2, steps=2, hidden=8).double()
    for p in g.parameters():
        p.data += 0.05 * torch.randn_like(p)
    g.train(); g(torch.randn(8, 1, 4, 4, dtype=torch.float64)); g.eval()   # ActNorm data init
    x = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    zs = g.encode(x)
    assert [tuple(z.shape) for z in zs] == [(1, 2, 2, 2), (1, 8, 1, 1)]
    assert torch.allclose(g.reverse(zs), x, atol=1e-8)
    # total log-det of the bijection x -> concat(z) equals the brute-force Jacobian
    _, out = g(x)
    brute = _jac_logdet(lambda v: torch.cat([z.reshape(-1) for z in g.encode(v)]), x)
    assert out["logdet"].item() == pytest.approx(brute.item(), abs=1e-6)


def test_glow_fully_convolutional_and_anomaly_map_shape():
    g = Glow(3, levels=3, steps=2, hidden=16)
    g.train(); g(torch.randn(4, 3, 32, 32)); g.eval()
    lp, out = g(torch.randn(2, 3, 48, 64), return_maps=True)
    assert lp.shape == (2,)
    assert latent_anomaly_map(out, (48, 64)).shape == (2, 48, 64)


def test_contextflow_zero_init_equals_generalist_and_is_invertible():
    g = Glow(3, levels=2, steps=2, hidden=16)
    g.train(); g(torch.randn(4, 3, 16, 16)); g.eval()
    cf = ContextFlowPP(g, n_cont=1, discrete_cardinalities=[3])
    x = torch.randn(3, 3, 16, 16); cont = torch.rand(3, 1); disc = torch.tensor([[0], [1], [2]])
    lp_g, _ = g(x)
    lp_c, _ = cf(x, cont, disc)
    assert torch.allclose(lp_g, lp_c, atol=1e-4)
    assert not any(p.requires_grad for p in g.parameters())
    for p in cf.cn.parameters():
        p.data.normal_(0, 0.02)
    ctx = cf.contexts(cont, disc)
    zs = g.encode(x, ctx)
    assert torch.allclose(g.reverse(zs, contexts=ctx), x, atol=1e-3)
    lp1, _ = cf(x, cont, disc); lp2, _ = cf(x, cont, torch.tensor([[2], [0], [1]]))
    assert not torch.allclose(lp1, lp2)                      # context changes the density


def test_vqa_model_shapes_and_answer_mask():
    tok = default_tokenizer()
    for temporal in (False, True):
        m = VQAModel(tok, 3, temporal=temporal)
        x = torch.randn(2, 3, 32, 48, 48)
        logits, region = m(tok.batch(["Where is it?", "Did it grow?"]), torch.tensor([3, 4]), x, x if temporal else None)
        assert logits.shape == (2, len(ANSWERS)) and region.shape == (2, 4 * 6 * 6)
    mask = answer_mask(torch.tensor([TASKS.index("existence")]))
    allowed = [ANSWERS[i] for i in range(len(ANSWERS)) if mask[0, i] == 0]
    assert sorted(allowed) == ["no", "yes"]


def test_keyword_router():
    assert keyword_route("Has the lesion grown?") == "progression"
    assert keyword_route("What is the volume change in mm3?") == "measurement"
    assert keyword_route("What is the patient's heart rate?") == "unsupported"
