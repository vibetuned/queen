from queen.evaluator import CUE_TO_E2V, cue_direction
from queen.judge import summarize_compare, summarize_scores
from queen.manifest import EMOTIONS


def test_cue_direction():
    assert cue_direction('(adult - american - male - angry - raised voice) "Text here"') == "adult - american - male - angry - raised voice"
    assert cue_direction('"No descriptor"') == ""


def test_cue_map_covers_manifest_emotions():
    assert all(e in CUE_TO_E2V for e in EMOTIONS)
    assert set(filter(None, CUE_TO_E2V.values())) <= {"angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"}


def test_summarize_scores_excludes_unmapped_emotions():
    rows = [
        {"source": "x", "mos": 3.0, "mos_target": 4.0, "e2v_match": True, "e2v_target_match": True, "e2v_sim_target": 0.9},
        {"source": "x", "mos": 4.0, "mos_target": 4.0, "e2v_match": False, "e2v_target_match": True, "e2v_sim_target": 0.7},
        {"source": "x", "mos": 2.0, "mos_target": 4.0, "e2v_match": None, "e2v_target_match": None, "e2v_sim_target": 0.5},
    ]
    s = summarize_scores(rows)
    assert s["x"]["n"] == 3 and s["x"]["n_emotion"] == 2
    assert s["x"]["mos"] == 3.0 and s["x"]["e2v_acc"] == 0.5 and s["x"]["e2v_acc_target"] == 1.0
    assert s["x"]["e2v_sim_target"] == 0.7
    assert s["all"] == s["x"]


def test_summarize_compare_counts():
    rows = [{"source": "x", "consensus": c} for c in ("A", "A", "B", "Tie", "Positional")]
    s = summarize_compare(rows)
    assert s["x"] == {"n": 5, "A": 2, "B": 1, "tie": 1, "positional": 1}
