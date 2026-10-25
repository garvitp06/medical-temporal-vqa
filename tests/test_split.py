from src.data.dataset import assert_no_patient_leakage, patient_level_split, split_summary


def _manifest(n=30):
    m = []
    for p in range(n):
        for t in (1, 2):
            m.append(dict(exam_id=f"P{p}-T{t}", patient_id=f"P{p}", timepoint=t, abnormal=int(p % 3 != 0)))
    return m


def test_patient_level_split_no_leakage_and_stratified():
    m = _manifest()
    s = patient_level_split(m, (0.6, 0.2, 0.2), seed=1)
    assert assert_no_patient_leakage(s, m)
    summ = split_summary(m, s)
    assert sum(v["patients"] for v in summ.values()) == 30
    for name in ("train", "val", "test"):
        assert summ[name]["examinations"] == 2 * summ[name]["patients"]   # both timepoints stay together
        assert summ[name]["temporal_pairs"] == summ[name]["patients"]
        assert 0 < summ[name]["abnormal_exams"] < summ[name]["examinations"]


def test_split_deterministic_and_held_out_site():
    m = _manifest()
    for r in m:
        r["site"] = "B" if r["patient_id"] in ("P1", "P2") else "A"
    s1 = patient_level_split(m, seed=5, held_out_key="site", held_out_values=["B"])
    s2 = patient_level_split(m, seed=5, held_out_key="site", held_out_values=["B"])
    assert s1 == s2 and s1["P1"] == "held_out" and s1["P2"] == "held_out"
