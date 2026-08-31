"""
Evaluators module for stock-underdog-ml.
Includes composite multi-strategy evaluation, factor scoring, and report formatting.
"""
from evaluators.composite_evaluator import CompositeEvaluator, EvaluationReport
from evaluators.formatter import format_value, print_evaluation_report

__all__ = [
    "CompositeEvaluator",
    "EvaluationReport",
    "format_value",
    "print_evaluation_report"
]
