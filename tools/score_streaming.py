#!/usr/bin/env python3
"""Official scorer for AIRIS streaming questions.

Predictions are JSONL records with ``qid`` and ``pred``.  ``pred`` may be an
option key, or a list of option keys.  Missing predictions are scored as
incorrect.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping, Sequence


ACCURACY_TYPES = {"single_choice", "bbox_select", "yes_no"}
MULTI_TYPE = "multi_choice"
REACTION_CAPABILITIES = (
    "reaction.engagement_decision",
    "reaction.action_domain",
    "reaction.subaction",
    "reaction.speak_subtype",
)
SEGMENT_CAPABILITIES = (
    "interaction.d1_completeness",
    "interaction.d1_mutual",
    "interaction.d2_spatial",
    "interaction.d3_physical",
    "interaction.d4_comm",
    "interaction.d5_category",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL lines and report malformed input with a line number."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def normalize_keys(value: Any) -> tuple[str, ...]:
    """Normalize a scalar/list prediction into a de-duplicated tuple of keys."""
    if value is None:
        return ()
    if isinstance(value, str) and any(separator in value for separator in (",", "，", "、")):
        values = re.split(r"[,，、]", value)
    else:
        values = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[str] = []
    for item in values:
        key = str(item).strip()
        if key and key not in result:
            result.append(key)
    return tuple(result)


def question_clip(question: Mapping[str, Any]) -> str:
    if question.get("clip"):
        return str(question["clip"])
    frames = question.get("frames") or []
    if frames and isinstance(frames[0], Mapping):
        return str(frames[0].get("clip_id", "unknown"))
    return "unknown"


def question_layer(question: Mapping[str, Any]) -> str:
    capability = str(question.get("capability", "unknown"))
    return capability.split(".", 1)[0]


def question_subset(question: Mapping[str, Any]) -> str:
    extra = question.get("extra") or {}
    for container in (question, extra):
        for key in ("source_subset", "subset"):
            if container.get(key) is not None:
                return str(container[key])
    return "unknown"


def score_question(question: Mapping[str, Any], prediction: Any) -> dict[str, Any]:
    """Score one question without mutating either input."""
    qtype = str(question.get("qtype", ""))
    if qtype not in ACCURACY_TYPES | {MULTI_TYPE}:
        raise ValueError(f"unsupported qtype {qtype!r} for {question.get('qid')!r}")
    gold = set(normalize_keys(question.get("answer")))
    pred = set(normalize_keys(prediction))
    option_keys = {
        str(option.get("key"))
        for option in question.get("options") or []
        if option.get("key") is not None
    }
    invalid_prediction_keys = sorted(pred - option_keys)
    exact = float(pred == gold)
    result: dict[str, Any] = {
        "qid": question.get("qid"),
        "qtype": qtype,
        "capability": question.get("capability", "unknown"),
        "layer": question_layer(question),
        "clip": question_clip(question),
        "source_subset": question_subset(question),
        "source_record_id": question.get("source_record_id"),
        "gold": sorted(gold),
        "pred": sorted(pred),
        "missing": prediction is None,
        "invalid_prediction_keys": invalid_prediction_keys,
        "exact_match": exact,
        "primary": exact,
    }
    if qtype in ACCURACY_TYPES:
        result["accuracy"] = exact
    else:
        overlap = len(gold & pred)
        precision = overlap / len(pred) if pred else (1.0 if not gold else 0.0)
        recall = overlap / len(gold) if gold else (1.0 if not pred else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.update(precision=precision, recall=recall, f1=f1, primary=f1)
    return result


def metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Macro-average the metrics present in scored question rows."""
    result: dict[str, Any] = {
        "n": len(rows),
        "missing": sum(bool(row.get("missing")) for row in rows),
    }
    if not rows:
        return result
    result["score"] = fmean(float(row["primary"]) for row in rows)
    for metric in ("accuracy", "precision", "recall", "f1", "exact_match"):
        values = [float(row[metric]) for row in rows if metric in row]
        if values:
            result[metric] = fmean(values)
            result[f"n_{metric}"] = len(values)
    return result


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)
    return {key: metric_summary(groups[key]) for key in sorted(groups)}


def _trigger_ms(question: Mapping[str, Any]) -> float | None:
    extra = question.get("extra") or {}
    streaming = extra.get("streaming") or {}
    value = extra.get("trigger_tMs", streaming.get("trigger_tMs"))
    return float(value) if value is not None else None


def _is_undetermined(question: Mapping[str, Any], pred: Any) -> bool:
    keys = set(normalize_keys(pred))
    if not keys:
        return False
    extra = question.get("extra") or {}
    if str(extra.get("gold_key", "")).lower() == "undetermined":
        gold_keys = set(normalize_keys(question.get("answer")))
        if keys == gold_keys:
            return True
    for option in question.get("options") or []:
        if str(option.get("key")) in keys:
            text = str(option.get("text", "")).lower()
            if (
                "尚不能判定" in text
                or "insufficient evidence" in text
                or text in {"undetermined", "cannot determine"}
            ):
                return True
    return any(key.lower() in {"undetermined", "尚不能判定"} for key in keys)


def _semantic_label(question: Mapping[str, Any], value: Any) -> str | None:
    keys = normalize_keys(value)
    if len(keys) != 1:
        return None
    option_by_key = {
        str(option.get("key")): str(option.get("text") or option.get("text_zh"))
        for option in question.get("options") or []
    }
    return option_by_key.get(keys[0])


def _classification_metrics(
    pairs: Sequence[tuple[str, str | None]],
    labels: Sequence[str],
) -> dict[str, Any]:
    if not pairs:
        return {
            "n": 0,
            "accuracy": None,
            "macro_f1": None,
            "balanced_accuracy": None,
        }
    correct = sum(gold == pred for gold, pred in pairs)
    per_label: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    for label in labels:
        tp = sum(gold == label and pred == label for gold, pred in pairs)
        fp = sum(gold != label and pred == label for gold, pred in pairs)
        fn = sum(gold == label and pred != label for gold, pred in pairs)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if support:
            f1_values.append(f1)
            recall_values.append(recall)
    return {
        "n": len(pairs),
        "accuracy": correct / len(pairs),
        "macro_f1": fmean(f1_values) if f1_values else None,
        "balanced_accuracy": fmean(recall_values) if recall_values else None,
        "per_label": per_label,
    }


def _cluster_bootstrap_ci(
    records: Sequence[tuple[str, str, str | None]],
    labels: Sequence[str],
    *,
    iterations: int = 1000,
) -> dict[str, list[float] | int]:
    by_cluster: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for cluster, gold, pred in records:
        by_cluster[cluster].append((gold, pred))
    clusters = sorted(by_cluster)
    if not clusters:
        return {"iterations": iterations, "accuracy": [], "macro_f1": []}
    rng = random.Random(2505)
    accuracy: list[float] = []
    macro_f1: list[float] = []
    for _ in range(iterations):
        sampled: list[tuple[str, str | None]] = []
        for _ in clusters:
            sampled.extend(by_cluster[rng.choice(clusters)])
        metrics = _classification_metrics(sampled, labels)
        accuracy.append(float(metrics["accuracy"]))
        macro_f1.append(float(metrics["macro_f1"]))

    def interval(values: Sequence[float]) -> list[float]:
        ordered = sorted(values)
        low = ordered[int(0.025 * (len(ordered) - 1))]
        high = ordered[int(0.975 * (len(ordered) - 1))]
        return [low, high]

    return {
        "iterations": iterations,
        "cluster": "clip",
        "accuracy": interval(accuracy),
        "macro_f1": interval(macro_f1),
    }


def summarize_intention_classification(
    questions: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    relevant = [
        question
        for question in questions
        if question.get("capability")
        in {"intention.intent_at", "intention.intent_kf"}
    ]
    labels = sorted(
        {
            label
            for question in relevant
            if (label := _semantic_label(question, question.get("answer"))) is not None
        }
    )
    records: list[tuple[str, str, str | None]] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for question in relevant:
        gold = _semantic_label(question, question.get("answer"))
        if gold is None:
            raise ValueError(f"{question.get('qid')}: invalid semantic gold")
        pred = _semantic_label(
            question, predictions.get(str(question.get("qid")))
        )
        records.append((question_clip(question), gold, pred))
        confusion[gold][pred or "__missing_or_invalid__"] += 1
    metrics = _classification_metrics(
        [(gold, pred) for _, gold, pred in records], labels
    )
    accuracy = metrics.get("accuracy")
    chance = 0.25
    return {
        **metrics,
        "labels": labels,
        "confusion_matrix": {
            gold: dict(sorted(counts.items()))
            for gold, counts in sorted(confusion.items())
        },
        "uniform_random_accuracy": chance,
        "chance_normalized_accuracy": (
            (float(accuracy) - chance) / (1.0 - chance)
            if accuracy is not None
            else None
        ),
        "always_undetermined_accuracy": (
            sum(
                gold == "Insufficient evidence to determine the intention"
                for _, gold, _ in records
            )
            / len(records)
            if records
            else None
        ),
        "cluster_bootstrap_95_ci": _cluster_bootstrap_ci(records, labels),
    }


def intent_event_key(question: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the event identity shared by all polls in one intent timeline."""
    extra = question.get("extra") or {}
    explicit = extra.get("event_id") or question.get("event_id")
    if explicit is not None:
        return (question_clip(question), str(explicit))
    return (
        question_clip(question),
        extra.get("t1_ms"),
        extra.get("t2_ms"),
        extra.get("i_star_key"),
    )


def question_event_key(question: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Return the cross-layer event key shared by one T0--T3 exchange."""
    extra = question.get("extra") or {}
    anchors = tuple(extra.get(name) for name in ("t0_ms", "t1_ms", "t2_ms"))
    if any(value is None for value in anchors):
        return None
    return (question_clip(question), *anchors)


def compute_intent_at_ias(
    questions: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any],
    stable_k: int = 2,
) -> dict[str, Any]:
    """Compute event-level IAS for ``intention.intent_at`` timelines.

    Only timelines explicitly marked ``ias_primary_eligible`` enter the primary
    metric. Each must contain at least one pre-T1 poll and ``stable_k`` post-T1
    polls. This prevents no-pre events from being treated as FCR=0 and prevents
    reduced-K timelines from receiving an easier stable-success criterion.
    """
    if stable_k < 1:
        raise ValueError("stable_k must be at least 1")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for question in questions:
        if question.get("capability") == "intention.intent_at":
            grouped[intent_event_key(question)].append(question)

    events: list[dict[str, Any]] = []
    skipped = 0
    excluded = 0
    invalid_primary = 0
    for key, timeline in grouped.items():
        timeline = sorted(timeline, key=lambda q: _trigger_ms(q) or float("-inf"))
        extra = timeline[0].get("extra") or {}
        if extra.get("ias_primary_eligible") is not True:
            excluded += 1
            continue
        t1, t2 = extra.get("t1_ms"), extra.get("t2_ms")
        points = [(q, _trigger_ms(q)) for q in timeline]
        if t1 is None or t2 is None or float(t2) <= float(t1) or any(t is None for _, t in points):
            skipped += 1
            continue
        t1, t2 = float(t1), float(t2)
        pre_t1 = [(q, t) for q, t in points if t < t1]
        post = [(q, t) for q, t in points if t >= t1]
        if not pre_t1 or len(post) < stable_k:
            invalid_primary += 1
            continue
        false_commitments = sum(
            predictions.get(str(q["qid"])) is not None
            and not _is_undetermined(q, predictions.get(str(q["qid"])))
            for q, _ in pre_t1
        )
        fcr = false_commitments / len(pre_t1)
        effective_k = stable_k
        correct = [
            set(normalize_keys(predictions.get(str(q["qid"]))))
            == set(normalize_keys(q.get("answer")))
            for q, _ in post
        ]
        p: float | None = None
        for index in range(len(post) - effective_k + 1):
            if all(correct[index : index + effective_k]) and all(correct[index:]):
                p = float(post[index][1])
                break
        first_observable_poll = float(post[0][1])
        if p is None:
            anticipation = 0.0
        elif t2 <= first_observable_poll:
            anticipation = 1.0
        else:
            anticipation = max(
                0.0,
                min(1.0, (t2 - p) / (t2 - first_observable_poll)),
            )
        events.append(
            {
                "event": "|".join(str(part) for part in key),
                "clip": question_clip(timeline[0]),
                "n_polls": len(timeline),
                "effective_stable_k": effective_k,
                "t1_ms": t1,
                "t2_ms": t2,
                "first_observable_poll_ms": first_observable_poll,
                "p_ms": p,
                "lead_time_ms": None if p is None else max(0.0, t2 - p),
                "stable_success": float(p is not None and p <= t2),
                "false_commitment_rate": fcr,
                "anticipation_margin": anticipation,
                "ias": anticipation * (1.0 - fcr),
            }
        )
    successful_leads = [
        float(event["lead_time_ms"])
        for event in events
        if event["stable_success"] and event["lead_time_ms"] is not None
    ]
    random_stable = 0.25**stable_k
    random_fcr = 0.75
    random_anticipation_values: list[float] = []
    for key, timeline in grouped.items():
        timeline = sorted(timeline, key=lambda q: _trigger_ms(q) or float("-inf"))
        extra = timeline[0].get("extra") or {}
        if extra.get("ias_primary_eligible") is not True:
            continue
        t1, t2 = float(extra["t1_ms"]), float(extra["t2_ms"])
        post_times = [
            float(_trigger_ms(question))
            for question in timeline
            if _trigger_ms(question) is not None
            and float(_trigger_ms(question)) >= t1
        ]
        pre_times = [
            float(_trigger_ms(question))
            for question in timeline
            if _trigger_ms(question) is not None
            and float(_trigger_ms(question)) < t1
        ]
        if not pre_times or len(post_times) < stable_k:
            continue
        first = post_times[0]
        expected_margin = 0.0
        m = len(post_times)
        for trailing_run in range(stable_k, m + 1):
            probability = (
                0.25**m
                if trailing_run == m
                else 0.75 * 0.25**trailing_run
            )
            p_ms = post_times[m - trailing_run]
            margin = (
                1.0
                if t2 <= first
                else max(0.0, min(1.0, (t2 - p_ms) / (t2 - first)))
            )
            expected_margin += probability * margin
        random_anticipation_values.append(expected_margin)
    random_margin = (
        fmean(random_anticipation_values) if random_anticipation_values else None
    )
    return {
        "stable_k": stable_k,
        "n_events": len(events),
        "n_skipped_events": skipped,
        "n_excluded_events": excluded,
        "n_invalid_primary_events": invalid_primary,
        "n_reduced_k_events": 0,
        "n_stable_success_events": len(successful_leads),
        "stable_success_rate": (
            len(successful_leads) / len(events) if events else None
        ),
        "successful_lead_time_ms_mean": (
            fmean(successful_leads) if successful_leads else None
        ),
        "successful_lead_time_ms_median": (
            median(successful_leads) if successful_leads else None
        ),
        "ias": fmean(event["ias"] for event in events) if events else None,
        "anticipation_margin": (
            fmean(event["anticipation_margin"] for event in events) if events else None
        ),
        "false_commitment_rate": (
            fmean(event["false_commitment_rate"] for event in events) if events else None
        ),
        "baselines": {
            "uniform_random": {
                "single_question_accuracy": 0.25,
                "stable_success_rate": random_stable,
                "false_commitment_rate": random_fcr,
                "anticipation_margin": random_margin,
                "ias": (
                    random_margin * (1.0 - random_fcr)
                    if random_margin is not None
                    else None
                ),
            },
            "always_undetermined": {
                "stable_success_rate": 0.0,
                "false_commitment_rate": 0.0,
                "anticipation_margin": 0.0,
                "ias": 0.0,
            },
        },
        "events": events,
    }


def _active_option_key(question: Mapping[str, Any]) -> str:
    """Identify the positive engagement option without assuming key order."""
    for option in question.get("options") or []:
        text = str(option.get("text") or "").strip().lower()
        text_zh = str(option.get("text_zh") or "").strip()
        if text.startswith("active response") or text_zh.startswith("主动响应"):
            return str(option.get("key"))
    return "A"


def summarize_engagement_decision(
    questions: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    """Balanced active/refrain metrics for the imbalanced engagement task."""
    relevant = [
        question
        for question in questions
        if question.get("capability") == "reaction.engagement_decision"
    ]
    tp = fp = fn = tn = 0
    for question in relevant:
        active_key = _active_option_key(question)
        gold_active = active_key in set(normalize_keys(question.get("answer")))
        pred_active = active_key in set(
            normalize_keys(predictions.get(str(question.get("qid"))))
        )
        if gold_active and pred_active:
            tp += 1
        elif not gold_active and pred_active:
            fp += 1
        elif gold_active and not pred_active:
            fn += 1
        else:
            tn += 1

    def f1(true_positive: int, false_positive: int, false_negative: int) -> float:
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        return (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    active_f1 = f1(tp, fp, fn)
    refrain_f1 = f1(tn, fn, fp)
    return {
        "n": len(relevant),
        "n_active_gold": tp + fn,
        "n_refrain_gold": tn + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(relevant) if relevant else None,
        "active_f1": active_f1 if relevant else None,
        "refrain_f1": refrain_f1 if relevant else None,
        "macro_f1": (active_f1 + refrain_f1) / 2 if relevant else None,
        "false_intervention_rate": fp / (fp + tn) if fp + tn else None,
        "missed_intervention_rate": fn / (fn + tp) if fn + tp else None,
        "correct_inaction_rate": tn / (tn + fp) if tn + fp else None,
    }


def summarize_reaction(
    questions: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reaction = [row for row in rows if row["capability"] in REACTION_CAPABILITIES]
    components = {
        capability: metric_summary([r for r in reaction if r["capability"] == capability])
        for capability in REACTION_CAPABILITIES
        if any(r["capability"] == capability for r in reaction)
    }
    component_scores = [
        summary.get("f1", summary.get("accuracy")) for summary in components.values()
    ]
    component_scores = [float(value) for value in component_scores if value is not None]
    rows_by_qid = {str(row["qid"]): row for row in reaction}
    by_decision: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for question in questions:
        if question.get("capability") in REACTION_CAPABILITIES:
            group = str(question.get("source_record_id") or question.get("qid"))
            by_decision[group].append(question)

    decisions: list[dict[str, Any]] = []
    incomplete_decisions = 0
    for group, group_questions in sorted(by_decision.items()):
        by_capability = {
            str(question["capability"]): question for question in group_questions
        }
        engagement = by_capability.get("reaction.engagement_decision")
        if engagement is None:
            incomplete_decisions += 1
            continue
        gold_active = _active_option_key(engagement) in set(
            normalize_keys(engagement.get("answer"))
        )
        required = ["reaction.engagement_decision"]
        if gold_active:
            if not all(
                capability in by_capability
                for capability in ("reaction.action_domain", "reaction.subaction")
            ):
                incomplete_decisions += 1
                continue
            required.extend(("reaction.action_domain", "reaction.subaction"))
            if "reaction.speak_subtype" in by_capability:
                required.append("reaction.speak_subtype")
        required_rows = [
            rows_by_qid[str(by_capability[capability]["qid"])]
            for capability in required
        ]
        decisions.append(
            {
                "source_record": group,
                "clip": required_rows[0]["clip"],
                "gold_active": gold_active,
                "n_required_components": len(required_rows),
                "decision_score": fmean(
                    float(row["primary"]) for row in required_rows
                ),
                "joint_em": float(
                    all(float(row["exact_match"]) == 1.0 for row in required_rows)
                ),
            }
        )
    active_decisions = [
        decision for decision in decisions if decision["gold_active"]
    ]
    refrain_decisions = [
        decision for decision in decisions if not decision["gold_active"]
    ]
    active_joint = (
        fmean(decision["joint_em"] for decision in active_decisions)
        if active_decisions
        else None
    )
    refrain_joint = (
        fmean(decision["joint_em"] for decision in refrain_decisions)
        if refrain_decisions
        else None
    )
    return {
        "components": components,
        "reaction_score": fmean(component_scores) if component_scores else None,
        "engagement": summarize_engagement_decision(questions, predictions),
        "n_complete_decisions": len(decisions),
        "n_incomplete_decisions": incomplete_decisions,
        "n_active_decisions": len(active_decisions),
        "n_refrain_decisions": len(refrain_decisions),
        "decision_score": (
            fmean(decision["decision_score"] for decision in decisions)
            if decisions
            else None
        ),
        "reaction_joint_em": (
            fmean(decision["joint_em"] for decision in decisions)
            if decisions
            else None
        ),
        "active_reaction_joint_em": active_joint,
        "refrain_reaction_joint_em": refrain_joint,
        "macro_reaction_joint_em": (
            (active_joint + refrain_joint) / 2
            if active_joint is not None and refrain_joint is not None
            else None
        ),
        "decisions": decisions,
    }


def summarize_airi_chain(
    questions: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate strict and soft A-I-R-I scores over linked social events."""
    layers = ("attention", "intention", "reaction", "interaction")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    unanchored_questions = 0
    for question in questions:
        key = question_event_key(question)
        if key is None:
            unanchored_questions += 1
            continue
        grouped[key].append(question)

    events: list[dict[str, Any]] = []
    incomplete_groups = 0
    failure_stage_counts = {layer: 0 for layer in layers}
    failure_stage_counts["none"] = 0
    for key, event_questions in sorted(grouped.items()):
        by_layer = {
            layer: [
                question
                for question in event_questions
                if question_layer(question) == layer
            ]
            for layer in layers
        }
        if any(not by_layer[layer] for layer in layers):
            incomplete_groups += 1
            continue
        layer_scores: dict[str, float] = {}
        layer_joint_em: dict[str, float] = {}
        for layer in layers:
            scored = [
                score_question(
                    question, predictions.get(str(question.get("qid")))
                )
                for question in by_layer[layer]
            ]
            layer_scores[layer] = fmean(float(row["primary"]) for row in scored)
            layer_joint_em[layer] = float(
                all(float(row["exact_match"]) == 1.0 for row in scored)
            )
        soft_chain = 1.0
        for layer in layers:
            soft_chain *= layer_scores[layer]
        soft_chain **= 1.0 / len(layers)
        strict_joint = float(all(layer_joint_em.values()))
        first_failure = next(
            (layer for layer in layers if not layer_joint_em[layer]), "none"
        )
        failure_stage_counts[first_failure] += 1
        events.append(
            {
                "event": "|".join(str(part) for part in key),
                "clip": str(key[0]),
                "t0_ms": key[1],
                "t1_ms": key[2],
                "t2_ms": key[3],
                "n_questions": len(event_questions),
                "layer_scores": layer_scores,
                "layer_joint_em": layer_joint_em,
                "soft_chain_score": soft_chain,
                "strict_chain_joint_em": strict_joint,
                "first_failure_stage": first_failure,
            }
        )
    strict_successes = sum(event["strict_chain_joint_em"] for event in events)
    return {
        "event_key": "clip|t0_ms|t1_ms|t2_ms",
        "n_events": len(events),
        "n_incomplete_event_groups": incomplete_groups,
        "n_unanchored_questions": unanchored_questions,
        "soft_chain_score": (
            fmean(event["soft_chain_score"] for event in events)
            if events
            else None
        ),
        "strict_chain_joint_em": (
            strict_successes / len(events) if events else None
        ),
        "n_strict_success_events": int(strict_successes),
        "first_failure_stage_counts": failure_stage_counts,
        "events": events,
    }


def summarize_interaction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute clip-macro Segment Score and complete-segment Joint EM."""
    segment_rows = [row for row in rows if row["capability"] in SEGMENT_CAPABILITIES]
    by_clip: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in segment_rows:
        group = str(row.get("source_record_id") or row.get("clip"))
        by_clip[group].append(row)

    segments: list[dict[str, Any]] = []
    for group, group_rows in sorted(by_clip.items()):
        by_capability = {str(row["capability"]): row for row in group_rows}
        if not all(capability in by_capability for capability in SEGMENT_CAPABILITIES):
            continue
        scores = [
            float(by_capability[capability]["primary"]) for capability in SEGMENT_CAPABILITIES
        ]
        segments.append(
            {
                "source_record": group,
                "clip": group_rows[0]["clip"],
                "segment_score": fmean(scores),
                "joint_em": float(
                    all(by_capability[capability]["exact_match"] for capability in SEGMENT_CAPABILITIES)
                ),
            }
        )
    return {
        "components": {
            capability: metric_summary(
                [row for row in segment_rows if row["capability"] == capability]
            )
            for capability in SEGMENT_CAPABILITIES
            if any(row["capability"] == capability for row in segment_rows)
        },
        "n_complete_segments": len(segments),
        "n_incomplete_segments": len(by_clip) - len(segments),
        "segment_score": (
            fmean(segment["segment_score"] for segment in segments) if segments else None
        ),
        "segment_joint_em": (
            fmean(segment["joint_em"] for segment in segments) if segments else None
        ),
        "segments": segments,
    }


def summarize_violation_detection(
    questions: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        question
        for question in questions
        if question.get("capability") == "interaction.violation_presence_at"
    ]
    tp = fp = fn = tn = 0
    for question in rows:
        gold = set(normalize_keys(question.get("answer")))
        pred = set(normalize_keys(predictions.get(str(question.get("qid")))))
        positive_key = next(
            (
                str(option.get("key"))
                for option in question.get("options") or []
                if str(option.get("text") or "").lower().startswith("yes")
                or str(option.get("text_zh") or option.get("text") or "").startswith("是")
            ),
            "A",
        )
        gold_positive = positive_key in gold
        pred_positive = positive_key in pred
        if gold_positive and pred_positive:
            tp += 1
        elif not gold_positive and pred_positive:
            fp += 1
        elif gold_positive and not pred_positive:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(rows) if rows else None,
        "precision": precision if rows else None,
        "recall": recall if rows else None,
        "f1": f1 if rows else None,
    }


def build_report(
    questions: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    stable_k: int = 2,
) -> dict[str, Any]:
    """Build the complete JSON-serializable report."""
    predictions: dict[str, Any] = {}
    for record in prediction_records:
        if "qid" not in record or "pred" not in record:
            raise ValueError("each prediction record must contain qid and pred")
        qid = str(record["qid"])
        if qid in predictions:
            raise ValueError(f"duplicate prediction qid: {qid}")
        predictions[qid] = record["pred"]
    qids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for question in questions:
        if "qid" not in question:
            raise ValueError("each question must contain qid")
        qid = str(question["qid"])
        if qid in qids:
            raise ValueError(f"duplicate question qid: {qid}")
        qids.add(qid)
        rows.append(score_question(question, predictions.get(qid)))

    by_clip_rows = aggregate_rows(rows, "clip")
    return {
        "schema_version": "airis-streaming-score-v3",
        "n_questions": len(rows),
        "n_predictions": sum(not row["missing"] for row in rows),
        "n_missing_predictions": sum(row["missing"] for row in rows),
        "n_unknown_predictions": len(set(predictions) - qids),
        "n_invalid_prediction_keys": sum(
            len(row["invalid_prediction_keys"]) for row in rows
        ),
        "overall": metric_summary(rows),
        "overall_clip_macro": (
            fmean(float(summary["score"]) for summary in by_clip_rows.values())
            if by_clip_rows
            else None
        ),
        "by_capability": aggregate_rows(rows, "capability"),
        "by_layer": aggregate_rows(rows, "layer"),
        "by_clip": by_clip_rows,
        "by_source_subset": aggregate_rows(rows, "source_subset"),
        "intention_classification": summarize_intention_classification(
            questions, predictions
        ),
        "intent_at": compute_intent_at_ias(questions, predictions, stable_k),
        "reaction": summarize_reaction(questions, predictions, rows),
        "interaction": summarize_interaction(rows),
        "violation_detection": summarize_violation_detection(questions, predictions),
        "airi_chain": summarize_airi_chain(questions, predictions),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="canonical questions.jsonl")
    parser.add_argument("--predictions", required=True, help="predictions.jsonl")
    parser.add_argument("--output", "--out", default="report.json", help="output report.json")
    parser.add_argument("--stable-k", "--k", type=int, default=2, help="IAS stable run length")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        read_jsonl(args.questions), read_jsonl(args.predictions), stable_k=args.stable_k
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}: {report['n_questions']} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
