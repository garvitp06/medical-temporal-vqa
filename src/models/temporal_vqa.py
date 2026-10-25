"""Evidence-grounded answer composition for single-exam and temporal VQA
(spec §7.4, §8.4).

The learned VQA model supplies the categorical answer; the candidate
localisation + measurement modules supply every location and number. The
composer cross-checks the two and reports disagreement as uncertainty. It
returns an explicit insufficient-evidence answer instead of inventing
findings, and it refuses to state growth when registration is unreliable.
"""
from __future__ import annotations

from ..temporal.measurements import categorize_change

DISCLAIMER = ("Research prototype output: suspicious regions are model-flagged areas for clinician "
              "review, not diagnoses.")


def _calib_note(calibrated):
    return ("VQA confidence was temperature-scaled on synthetic validation data only; it is not clinically calibrated"
            if calibrated else "VQA confidence is uncalibrated")


def _fmt(v, unit="", nd=0):
    return "n/a" if v is None else f"{v:,.{nd}f}{unit}"


def candidate_derived_answer(task, candidates):
    """Deterministic answer implied by the localised candidates alone."""
    if task == "existence":
        return "yes" if candidates else "no"
    if not candidates:
        return "none"
    if task == "laterality":
        sides = {c["laterality"] for c in candidates}
        return sides.pop() if len(sides) == 1 else "both"
    if task == "zone":
        return max(candidates, key=lambda c: c["volume_mm3"])["zone"]
    if task == "where":
        return "region"
    return None


def compose_single(question, task, model_answer, model_conf, candidates, exam_ref, calibrated=False):
    cand_ans = candidate_derived_answer(task, candidates)
    unc = []
    if cand_ans is not None and model_answer is not None and cand_ans != model_answer:
        unc.append(f"learned VQA answer '{model_answer}' disagrees with candidate-derived answer '{cand_ans}'")
    unc.append(_calib_note(calibrated))
    top = max(candidates, key=lambda c: c["max_score"]) if candidates else None
    evidence = []
    if top:
        evidence.append(dict(candidate_id=top["candidate_id"], key_slice=top["key_slice"],
                             bbox_voxel=top["bbox_voxel"], bbox_mm=top["bbox_mm"],
                             centroid_mm=top["centroid_mm"], max_score=top["max_score"]))
    if task == "where":
        if not top:
            text = "No suspicious region exceeded the detection threshold; insufficient evidence to localise a finding."
        else:
            c = top["centroid_mm"]
            text = (f"The most suspicious region ({top['candidate_id']}) is in the {top['zone']} zone of the "
                    f"{top['laterality']} lung, centred at ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}) mm (LPS), "
                    f"axial slice {top['key_slice'] + 1} (1-based), candidate-mask volume {_fmt(top['volume_mm3'], ' mm³')}.")
    elif task in ("existence", "laterality", "zone"):
        ans = model_answer
        text = {"existence": f"Suspicious region present: {ans}.",
                "laterality": f"Side of suspicious finding: {ans}.",
                "zone": f"Lung zone of the largest suspicious region: {ans}."}[task]
        if top:
            text += f" Supporting evidence: {top['candidate_id']} on axial slice {top['key_slice'] + 1} (1-based)."
    else:
        text = "This question type is not supported for a single examination."
    return dict(question=question, task=task, exam=exam_ref, answer=model_answer,
                model_confidence=model_conf, confidence_calibrated=calibrated,
                candidate_derived_answer=cand_ans, answer_text=text + " " + DISCLAIMER,
                evidence=evidence, uncertainty=unc,
                status="answered" if (top or task == "existence" or model_answer == "none") else "insufficient_evidence")


def compose_temporal(question, task, model_answer, model_conf, pair, registration, earlier_ref, later_ref,
                     stable_band_pct=15.0, calibrated=False):
    """pair: output of the temporal pipeline's primary-candidate selection:
    dict(match_status, earlier, later, measurements, n_matched, n_new, n_resolved, n_uncertain)."""
    reliable = bool(registration.get("reliable"))
    unc = []
    if not reliable:
        unc.append("registration flagged unreliable: growth/regression is NOT reported")
    unc.append(_calib_note(calibrated))
    meas = pair.get("measurements") if pair else None
    derived = None
    if pair and meas:
        vm = meas["volume_mm3"]
        derived = categorize_change(vm["percent"], vm["case"], stable_band_pct)
    elif pair is None or pair.get("match_status") == "none":
        derived = "none"
    if task == "new_finding":
        derived = "yes" if (pair and pair.get("n_new", 0) > 0) else "no"
    if derived is not None and model_answer is not None and derived != model_answer:
        unc.append(f"learned VQA answer '{model_answer}' disagrees with measurement-derived answer '{derived}'")
    cid = None
    if pair and pair.get("later"):
        cid = pair["later"]["candidate_id"]
    elif pair and pair.get("earlier"):
        cid = pair["earlier"]["candidate_id"]
    if task == "measurement" or task == "progression":
        if not reliable:
            text = ("Registration between the examinations was assessed as unreliable, so no change in "
                    "size is reported.")
            answer = "undetermined"
        elif not pair or not meas:
            text = "No suspicious candidate was found in either examination; insufficient evidence to assess change."
            answer = "none"
        else:
            vm, dm = meas["volume_mm3"], meas["longest_diameter_mm"]
            pct = "undefined (no earlier candidate)" if vm["percent"] is None else f"{vm['percent']:+.1f}%"
            text = (f"Candidate {cid}: candidate-mask volume {_fmt(vm['earlier'], ' mm³')} in the earlier "
                    f"examination and {_fmt(vm['later'], ' mm³')} in the later examination "
                    f"(absolute change {_fmt(vm['absolute'], ' mm³')}, percentage change {pct}); longest "
                    f"diameter {_fmt(dm['earlier'], ' mm', 1)} -> {_fmt(dm['later'], ' mm', 1)}. "
                    f"Registration quality: {registration.get('quality')}. ")
            answer = model_answer if task == "progression" else derived
            if task == "progression":
                text += f"Model assessment of change: {model_answer}; measurement-derived category: {derived}."
    elif task == "new_finding":
        answer = model_answer
        text = (f"New suspicious region compared with the previous examination: {model_answer}. "
                f"Matched candidates: {pair.get('n_matched', 0) if pair else 0}, potentially new: "
                f"{pair.get('n_new', 0) if pair else 0}, potentially resolved: {pair.get('n_resolved', 0) if pair else 0}, "
                f"uncertain: {pair.get('n_uncertain', 0) if pair else 0}.")
        if not reliable:
            answer = "undetermined"
    else:
        answer, text = None, "This question type is not supported for temporal comparison."
    return dict(question=question, task=task, earlier_exam=earlier_ref, later_exam=later_ref,
                matched_candidate_id=cid, answer=answer, model_answer=model_answer,
                model_confidence=model_conf, confidence_calibrated=calibrated,
                measurement_derived_answer=derived,
                measurements=meas, registration_quality=registration,
                coordinates=dict(earlier=pair["earlier"]["centroid_mm"] if pair and pair.get("earlier") else None,
                                 later=pair["later"]["centroid_mm"] if pair and pair.get("later") else None),
                answer_text=text + " The finding requires clinician review. " + DISCLAIMER,
                uncertainty=unc)
