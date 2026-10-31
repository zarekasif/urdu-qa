import pandas as pd
import pytest

from urduqa.metrics import evaluate, exact_match, f1
from urduqa.normalize import clean_text, normalize_answer
from urduqa.prepare import prepare
from urduqa.split import leakage, split_by_paragraph


# --- normalisation -------------------------------------------------------------------

def test_clean_text_unescapes_html_entities():
    assert clean_text("B&#39;Day &quot;Déjà Vu&quot;") == "B'Day \"Déjà Vu\""


def test_normalize_unifies_arabic_and_urdu_letters():
    # Arabic kaf + Arabic yeh vs Urdu keheh + Urdu yeh
    assert normalize_answer("كي") == normalize_answer("کی")


def test_normalize_maps_urdu_digits_and_strips_punctuation():
    assert normalize_answer("۲۰۰۳۔") == "2003"


# --- metrics -------------------------------------------------------------------------

def test_exact_match_ignores_trailing_urdu_full_stop():
    assert exact_match("ہیوسٹن، ٹیکساس۔", "ہیوسٹن ٹیکساس") == 1.0


def test_f1_partial_overlap():
    # prediction has one extra token: precision 2/3, recall 1 -> F1 0.8
    assert f1("جے زیڈ سے", "جے زیڈ") == pytest.approx(0.8)


def test_f1_no_overlap_is_zero():
    assert f1("فوربس", "ٹائم") == 0.0


def test_evaluate_counts_missing_predictions_as_wrong():
    golds = {"a": ["2003"], "b": ["فوربس"]}
    result = evaluate(golds, {"a": "2003"})
    assert result["exact_match"] == pytest.approx(50.0)
    assert result["missing"] == 1


# --- data preparation and split ------------------------------------------------------

def _toy_frame():
    p1 = "بیونس ہیوسٹن میں پیدا ہوئی۔ اس نے 2003 میں پہلا البم ریلیز کیا۔"
    p2 = "فوربس میگزین نے انہیں 2015 کی سب سے طاقتور موسیقار قرار دیا۔"
    return pd.DataFrame(
        {
            "Paragraphs": [p1, p1, p2, p2, p2],
            "Questions": ["کہاں پیدا ہوئی؟", "پہلا البم کب آیا؟", "کس میگزین نے؟", "کس سال؟", "کس سال؟"],
            "Answers": ["ہیوسٹن", "2003", "فوربس", "2015", "2015"],
        }
    )


def test_prepare_finds_spans_and_drops_duplicates():
    records, dropped = prepare(_toy_frame())
    assert len(records) == 4
    assert dropped["duplicate_question"] == 1
    for r in records:
        start = r["answers"]["answer_start"][0]
        text = r["answers"]["text"][0]
        assert r["context"][start : start + len(text)] == text


def test_prepare_drops_answers_missing_from_context():
    df = _toy_frame()
    df.loc[0, "Answers"] = "لاہور"
    _, dropped = prepare(df)
    assert dropped["answer_not_in_context"] == 1


def test_split_keeps_paragraphs_together():
    records = []
    for p in range(50):
        for q in range(3):
            records.append({"paragraph_id": f"p{p}", "id": f"p{p}-{q}"})
    splits = split_by_paragraph(records, seed=0)
    assert sum(len(v) for v in splits.values()) == len(records)
    assert leakage(splits["train"], splits["dev"]) == 0.0
    assert leakage(splits["train"], splits["test"]) == 0.0


def test_leakage_detects_shared_paragraphs():
    train = [{"paragraph_id": "p1"}, {"paragraph_id": "p2"}]
    dev = [{"paragraph_id": "p1"}, {"paragraph_id": "p3"}]
    assert leakage(train, dev) == pytest.approx(0.5)


def test_load_any_reads_simpletransformers_and_squad_json(tmp_path):
    import json
    from urduqa.prepare import load_any

    para = {"context": "فوربس میگزین 2015", "qas": [{"id": "1", "question": "کون؟", "answers": [{"text": "فوربس", "answer_start": 0}]}]}
    st = tmp_path / "st.json"
    st.write_text(json.dumps([para], ensure_ascii=False), encoding="utf-8")
    sq = tmp_path / "sq.json"
    sq.write_text(json.dumps({"data": [{"title": "t", "paragraphs": [para]}]}, ensure_ascii=False), encoding="utf-8")
    for path in (st, sq):
        df = load_any(path)
        assert list(df.itertuples(index=False, name=None)) == [("فوربس میگزین 2015", "کون؟", "فوربس", 0)]


def test_prepare_prefers_the_datasets_own_answer_offset():
    # The answer occurs twice; the given offset points at the second occurrence.
    ctx = "پہلے 2003 پھر &quot;دوبارہ&quot; 2003"
    df = pd.DataFrame({"Paragraphs": [ctx], "Questions": ["کب؟"], "Answers": ["2003"], "answer_start": [ctx.rindex("2003")]})
    records, _ = prepare(df)
    r = records[0]
    start = r["answers"]["answer_start"][0]
    assert start == r["context"].rindex("2003")
    assert r["context"][start : start + 4] == "2003"
