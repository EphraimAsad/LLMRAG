# training/gold_eval.py
from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple
import pandas as pd

from engine.schema import canonicalize_record
from engine.parser_rules import parse_and_canonicalize as parse_rules_only
from engine.parser_llm import smart_parse

def _compare(expected: Dict[str, str], predicted: Dict[str, str]) -> Tuple[int,int,Dict[str,Tuple[int,int]]]:
    """
    Compare predicted vs expected using ONLY keys present in expected.
    Returns: (correct, total, per_field_stats) where per_field_stats[field]=(correct,total)
    """
    correct = 0
    total = 0
    per_field: Dict[str, Tuple[int,int]] = {}
    for k, exp_v in expected.items():
        total += 1
        got = predicted.get(k, "Unknown")
        ok = (str(got).strip().lower() == str(exp_v).strip().lower())
        if k not in per_field:
            per_field[k] = (0,0)
        c,t = per_field[k]
        per_field[k] = (c + (1 if ok else 0), t + 1)
        if ok:
            correct += 1
    return correct, total, per_field

def run_gold_tests(gold_path: str, use_llm: bool = True, model: str = "deepseek-r1:latest"):
    """
    Load gold tests, parse each text, compare to expected, and return:
    - summary dict (overall accuracy, per-field accuracy)
    - dataframe of per-case results (name, accuracy, num_correct/num_expected)
    """
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold file not found at: {gold_path}")

    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)

    per_field_stats: Dict[str, Tuple[int,int]] = {}
    rows: List[List] = []
    total_correct = 0
    total_expected = 0

    for case in gold:
        name = case.get("name", "Unknown")
        text = case.get("input", "")
        expected = canonicalize_record(case.get("expected", {}))

        predicted = smart_parse(text, model=model) if use_llm else parse_rules_only(text)

        c, t, pf = _compare(expected, predicted)
        total_correct += c
        total_expected += t

        for k, (c1,t1) in pf.items():
            c0,t0 = per_field_stats.get(k, (0,0))
            per_field_stats[k] = (c0 + c1, t0 + t1)

        acc = 0 if t == 0 else round(100 * c / t)
        rows.append([name, acc, c, t])

    overall_acc = 0 if total_expected == 0 else round(100 * total_correct / total_expected)
    per_field_acc = {
        k: (0 if t==0 else round(100 * c / t))
        for k,(c,t) in sorted(per_field_stats.items(), key=lambda x: x[0].lower())
    }

    df_cases = pd.DataFrame(rows, columns=["Name", "Accuracy (%)", "Correct", "Expected Fields"])
    df_fields = pd.DataFrame(
        [(k, v) for k,v in per_field_acc.items()],
        columns=["Field", "Accuracy (%)"]
    ).sort_values("Accuracy (%)", ascending=False)

    summary = {
        "overall_accuracy_percent": overall_acc,
        "cases_count": len(gold),
        "per_field_accuracy": per_field_acc,
    }
    return summary, df_cases, df_fields
