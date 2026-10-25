"""SQLite-backed study/result registry + on-disk exam artefacts.

Only pseudonymous identifiers are stored: the DICOM PatientID is hashed
(salted SHA-256, 12 hex chars) and never written to logs or URLs."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
  study_id TEXT PRIMARY KEY, patient_key TEXT, study_date TEXT, status TEXT, error TEXT,
  synthetic INTEGER, created REAL, summary TEXT);
CREATE TABLE IF NOT EXISTS results (
  result_id TEXT PRIMARY KEY, kind TEXT, study_ids TEXT, created REAL, payload TEXT);
"""


class Registry:
    def __init__(self, root: Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "registry.sqlite"
        self.lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def study_dir(self, sid) -> Path:
        d = self.root / "studies" / sid; d.mkdir(parents=True, exist_ok=True); return d

    def new_study(self, synthetic=False) -> str:
        sid = uuid.uuid4().hex[:16]
        with self.lock, self._conn() as c:
            c.execute("INSERT INTO studies VALUES (?,?,?,?,?,?,?,?)",
                      (sid, None, None, "uploaded", None, int(synthetic), time.time(), None))
        return sid

    def update_study(self, sid, **kw):
        if "summary" in kw and not isinstance(kw["summary"], str):
            kw["summary"] = json.dumps(kw["summary"], default=_json_default)
        sets = ", ".join(f"{k}=?" for k in kw)
        with self.lock, self._conn() as c:
            c.execute(f"UPDATE studies SET {sets} WHERE study_id=?", (*kw.values(), sid))

    def get_study(self, sid):
        with self._conn() as c:
            r = c.execute("SELECT * FROM studies WHERE study_id=?", (sid,)).fetchone()
        if not r:
            return None
        keys = ["study_id", "patient_key", "study_date", "status", "error", "synthetic", "created", "summary"]
        d = dict(zip(keys, r))
        d["summary"] = json.loads(d["summary"]) if d["summary"] else None
        d["synthetic"] = bool(d["synthetic"])
        return d

    def list_studies(self):
        with self._conn() as c:
            ids = [r[0] for r in c.execute("SELECT study_id FROM studies ORDER BY created")]
        out = []
        for i in ids:
            s = self.get_study(i)
            s.pop("summary", None)
            out.append(s)
        return out

    def save_result(self, kind, study_ids, payload) -> str:
        rid = uuid.uuid4().hex[:16]
        payload = dict(payload, result_id=rid)
        with self.lock, self._conn() as c:
            c.execute("INSERT INTO results VALUES (?,?,?,?,?)",
                      (rid, kind, json.dumps(study_ids), time.time(), json.dumps(payload, default=_json_default)))
        return rid

    def get_result(self, rid):
        with self._conn() as c:
            r = c.execute("SELECT kind, study_ids, created, payload FROM results WHERE result_id=?", (rid,)).fetchone()
        if not r:
            return None
        return dict(result_id=rid, kind=r[0], study_ids=json.loads(r[1]), created=r[2], payload=json.loads(r[3]))


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
