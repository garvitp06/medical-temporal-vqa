"""Visual Question Answering (spec §7) — single-exam and temporal.

Components
  * ``Tokenizer``          word-level vocabulary (lower-cased, punctuation stripped)
  * ``QuestionRouter``     learned GRU classifier over task types, with a keyword
                           fallback; routes to the inference procedure.
  * ``VQAModel``           cross-attention fusion of question tokens with the
                           spatial image tokens of 1 or 2 exams. For 2 exams
                           (temporal) the token set is [earlier, later, later-earlier]
                           with a learned time embedding.
       heads: answer classification (masked to the task's answer space)
              region head: a distribution over image tokens (evidence /
              coarse localisation), supervised with the reference lesion cell.
Numbers are NEVER produced by these heads; see temporal/measurements.py.
"""
from __future__ import annotations

import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_encoder import VolumeEncoder

TASKS = ["existence", "laterality", "zone", "where", "progression", "new_finding",
         "measurement", "appearance", "unsupported"]
TEMPORAL_TASKS = {"progression", "new_finding", "measurement"}
ANSWER_SPACE = {
    "existence": ["yes", "no"],
    "laterality": ["left", "right", "both", "none"],
    "zone": ["upper", "middle", "lower", "none"],
    "where": ["region", "none"],
    "progression": ["increased", "decreased", "stable", "new", "resolved", "none"],
    "new_finding": ["yes", "no"],
}
ANSWERS = sorted({a for v in ANSWER_SPACE.values() for a in v})

# Paraphrase bank used to train the router and to augment VQA training. The
# held-out paraphrases (``HELDOUT_PARAPHRASES``) are never used for training and
# measure router generalisation.
PARAPHRASES = {
    "existence": ["Is there a suspicious region in this scan?", "Is any abnormality visible?",
                  "Are there suspicious findings in the lungs?", "Does this CT show a suspicious lesion?",
                  "Is there anything abnormal in this examination?", "Do you see a nodule or mass?"],
    "laterality": ["Which lung contains the suspicious finding?", "Is the finding in the left or right lung?",
                   "On which side is the abnormality?", "Which side of the chest has the lesion?",
                   "Left or right lung, where is the suspicious region?"],
    "zone": ["Which zone of the lung contains the largest suspicious region?",
             "Is the lesion in the upper, middle or lower zone?", "Which lung zone is affected?",
             "In which part of the lung, upper middle or lower, is the finding?"],
    "where": ["Where is the suspicious region?", "Where is the suspected tumor?", "Show me the abnormality.",
              "Which image region supports this answer?", "Locate the suspicious finding.",
              "Which regions require further review?"],
    "progression": ["Has the suspected lesion increased or decreased in size since the previous scan?",
                    "Has the abnormality changed since the previous scan?", "Did the lesion grow?",
                    "Is the nodule larger or smaller than before?", "How has the lesion size changed over time?",
                    "Has the candidate lesion increased or decreased in size?"],
    "new_finding": ["Is there a new suspicious region compared with the previous examination?",
                    "Are there any new lesions since the last scan?", "Has a new nodule appeared?",
                    "Is anything new compared with the prior CT?"],
    "measurement": ["What is the volume change of the lesion?", "By how many cubic millimetres did it change?",
                    "What is the percentage change in lesion volume?", "How large is the lesion now compared with before?",
                    "What is the diameter of the lesion?"],
    "appearance": ["What abnormality is visible in this region?", "What does the lesion look like?",
                   "Describe the suspicious region.", "What type of lesion is this?"],
    "unsupported": ["What is the patient's heart rate?", "Should we start chemotherapy?",
                    "What is the patient's name?", "Is the liver normal?", "What medication should be prescribed?"],
}
HELDOUT_PARAPHRASES = {
    "existence": ["Can you find any suspicious area here?"],
    "laterality": ["Is the lesion located on the right or the left?"],
    "zone": ["Is it in the upper or lower part of the lung?"],
    "where": ["Point out where the abnormal region is."],
    "progression": ["Has the size of the lesion changed compared with the earlier scan?"],
    "new_finding": ["Did any new suspicious region appear since the prior examination?"],
    "measurement": ["What is the change in volume in mm3?"],
    "appearance": ["Can you describe what the finding looks like?"],
    "unsupported": ["What dose of radiation therapy is needed?"],
}

GEN_CATEGORY_TO_TASK = {"existence": "existence", "evidence": "where", "progression": "progression",
                        "comparison": "new_finding"}


def qa_task(qa: dict) -> str:
    if qa["category"] == "localization":
        return "zone" if "zone" in qa["question"].lower() else "laterality"
    return GEN_CATEGORY_TO_TASK[qa["category"]]


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class Tokenizer:
    def __init__(self, texts=None, vocab=None):
        if vocab is None:
            words = sorted({w for t in (texts or []) for w in tokenize(t)})
            vocab = ["<pad>", "<unk>", "<cls>"] + words
        self.vocab = list(vocab)
        self.idx = {w: i for i, w in enumerate(self.vocab)}

    def encode(self, text, max_len=24):
        ids = [2] + [self.idx.get(w, 1) for w in tokenize(text)][: max_len - 1]
        return ids + [0] * (max_len - len(ids))

    def batch(self, texts, max_len=24):
        return torch.tensor([self.encode(t, max_len) for t in texts], dtype=torch.long)


def default_tokenizer():
    texts = [p for v in PARAPHRASES.values() for p in v]
    return Tokenizer(texts)


KEYWORDS = [
    ("measurement", ["volume", "mm3", "cubic", "percentage", "diameter", "how large", "millimetre", "millimeter"]),
    ("new_finding", ["new", "appeared"]),
    ("progression", ["increase", "decrease", "grow", "larger", "smaller", "changed", "change"]),
    ("appearance", ["look like", "describe", "type of", "what abnormality"]),
    ("zone", ["zone", "upper", "lower"]),
    ("laterality", ["left", "right", "which lung", "side"]),
    ("where", ["where", "locate", "show", "region supports", "point", "review"]),
    ("existence", ["is there", "any abnormal", "suspicious", "nodule", "mass", "find"]),
]


def keyword_route(q: str) -> str:
    s = q.lower()
    for task, keys in KEYWORDS:
        if any(k in s for k in keys):
            return task
    return "unsupported"


class QuestionRouter(nn.Module):
    def __init__(self, tok: Tokenizer, d=64):
        super().__init__()
        self.tok = tok
        self.emb = nn.Embedding(len(tok.vocab), d, padding_idx=0)
        self.gru = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * d, len(TASKS))

    def forward(self, ids):
        h, _ = self.gru(self.emb(ids))
        mask = (ids != 0).float()[..., None]
        return self.head((h * mask).sum(1) / mask.sum(1).clamp(min=1))

    @torch.no_grad()
    def route(self, q: str, min_conf=0.6):
        self.eval()
        p = F.softmax(self(self.tok.batch([q])), -1)[0]
        conf, k = p.max(0)
        unk = sum(1 for w in tokenize(q) if w not in self.tok.idx) / max(1, len(tokenize(q)))
        if conf.item() < min_conf or unk > 0.5:
            return keyword_route(q), float(conf), "keyword_fallback"
        return TASKS[int(k)], float(conf), "learned"


def train_router(tok, epochs=150, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    r = QuestionRouter(tok)
    data = [(q, TASKS.index(t)) for t, qs in PARAPHRASES.items() for q in qs]
    opt = torch.optim.Adam(r.parameters(), 3e-3)
    for _ in range(epochs):
        random.shuffle(data)
        ids = tok.batch([d[0] for d in data]); y = torch.tensor([d[1] for d in data])
        loss = F.cross_entropy(r(ids), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return r


class VQAModel(nn.Module):
    def __init__(self, tok: Tokenizer, in_channels=3, d_model=128, n_heads=4, n_layers=2,
                 grid=(32, 48, 48), temporal=False, width=16):
        super().__init__()
        self.tok, self.temporal = tok, temporal
        self.encoder = VolumeEncoder(in_channels, width, d_model, grid)
        self.q_emb = nn.Embedding(len(tok.vocab), d_model, padding_idx=0)
        self.q_pos = nn.Parameter(torch.zeros(1, 24, d_model))
        self.time_emb = nn.Parameter(torch.zeros(3, d_model))          # earlier, later, difference
        self.task_emb = nn.Embedding(len(TASKS), d_model)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout=0.1, batch_first=True)
        self.fusion = nn.TransformerDecoder(layer, n_layers)           # question attends to image tokens
        self.answer_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, len(ANSWERS)))
        self.region_q = nn.Linear(d_model, d_model)
        self.n_tok = self.encoder.token_grid[0] * self.encoder.token_grid[1] * self.encoder.token_grid[2]

    def image_tokens(self, x_later, x_earlier=None):
        t_later = self.encoder(x_later)
        if not self.temporal:
            return t_later + self.time_emb[1]
        t_earlier = self.encoder(x_earlier)
        return torch.cat([t_earlier + self.time_emb[0], t_later + self.time_emb[1],
                          (t_later - t_earlier) + self.time_emb[2]], 1)

    def forward(self, q_ids, task_ids, x_later, x_earlier=None):
        img = self.image_tokens(x_later, x_earlier)
        q = self.q_emb(q_ids) + self.q_pos[:, : q_ids.shape[1]]
        q[:, 0] = q[:, 0] + self.task_emb(task_ids)
        pad = q_ids == 0
        h = self.fusion(q, img, tgt_key_padding_mask=pad)
        cls = h[:, 0]
        logits = self.answer_head(cls)
        # region logits over the LATER exam's spatial tokens
        later = img[:, self.n_tok: 2 * self.n_tok] if self.temporal else img
        region = torch.einsum("bd,bnd->bn", self.region_q(cls), later) / later.shape[-1] ** 0.5
        return logits, region


def answer_mask(task_ids, device=None):
    m = torch.full((len(task_ids), len(ANSWERS)), float("-inf"), device=device)
    for i, t in enumerate(task_ids.tolist()):
        for a in ANSWER_SPACE.get(TASKS[t], []):
            m[i, ANSWERS.index(a)] = 0.0
    return m
