import json
import math

from chuvashia_rag.evaluation.run_evaluation import generate_recommendations, to_json_safe


def test_to_json_safe_replaces_nan_and_inf_recursively():
    report = {"summary": {"faithfulness": math.nan}, "rows": [1.0, math.inf, "x"]}

    safe = to_json_safe(report)

    assert safe == {"summary": {"faithfulness": None}, "rows": [1.0, None, "x"]}
    json.dumps(safe, allow_nan=False)  # не должно бросать


def test_historical_report_with_nan_becomes_valid_json():
    raw = '{"faithfulness": NaN, "answer_relevancy": 0.5}'

    assert json.dumps(to_json_safe(json.loads(raw)), allow_nan=False) == (
        '{"faithfulness": null, "answer_relevancy": 0.5}'
    )


def test_low_metrics_produce_targeted_recommendations():
    summary = {
        "faithfulness": {"mean": 0.5},
        "context_precision": {"mean": 0.9},
        "context_recall": {"mean": 0.9},
        "answer_relevancy": {"mean": 0.9},
    }

    recs = generate_recommendations(summary, noise_report={})

    assert any(r.startswith("🔴 Faithfulness") for r in recs)
    assert not any(r.startswith("🔴 Context Precision") for r in recs)
