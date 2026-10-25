"""Evidence figures: CT slice, anomaly overlay, candidate boxes; side-by-side
earlier(registered)/later for temporal results; slice-score profile."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..data.preprocessing import window_normalize  # noqa: E402


def _panel(ax, hu, amap, lung, cands, z, title, color="#4fe3c9", thr=3.0):
    ax.imshow(window_normalize(hu[z], -600, 1500, "window_0_1"), cmap="gray", vmin=0, vmax=1)
    m = np.ma.masked_where((amap[z] < 0.5 * thr) | (lung[z] == 0), amap[z])
    ax.imshow(m, cmap="inferno", alpha=0.55, vmin=0.5 * thr, vmax=2 * thr)
    for c in cands:
        z0, y0, x0, z1, y1, x1 = c["bbox_voxel"]
        if z0 <= z < z1:
            ax.add_patch(patches.Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.2))
            ax.text(x0, y0 - 2, c["candidate_id"], color=color, fontsize=7)
    ax.set_title(title, fontsize=9); ax.axis("off")


def evidence_figure(later, tctx, path, label="research output — not a diagnosis", thr=3.0):
    cands = later.candidates
    z = cands[0]["key_slice"] if cands else later.processed.normalized.shape[0] // 2
    n = 3 if tctx else 2
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4.3))
    hu, lung = later.processed.hu, later.processed.lung_mask
    if tctx:
        _panel(axs[0], tctx["earlier_on_later"], tctx["warped_earlier_map"], tctx["earlier_lung_on_later"],
               tctx["earlier_candidates_on_later"], z, f"earlier (registered) z={z}", "#9b8cf0", thr)
    _panel(axs[n - 2], hu, later.anomaly.map, lung, cands, z, f"later z={z}" if tctx else f"slice z={z}", thr=thr)
    axs[n - 1].plot(later.anomaly.slice_scores, color="#5fb3a8"); axs[n - 1].axvline(z, color="#e0b04f", ls="--")
    axs[n - 1].set_title("slice-level max anomaly score", fontsize=9); axs[n - 1].set_xlabel("axial slice")
    fig.suptitle(label, fontsize=9, color="#a06000")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path
