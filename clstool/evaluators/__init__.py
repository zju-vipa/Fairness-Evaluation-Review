from .default import DefaultEvaluator
from .fairness import FairnessEvaluator


def build_evaluator(args):
    evaluator_name = args.evaluator.lower()

    if evaluator_name in ['default']:
        metrics = ['acc', 'recall', 'precision', 'f1']
        return DefaultEvaluator(metrics)
    elif evaluator_name in ['fairness']:
        static_metrics = ['ACC', 'DP', 'EOpp', 'EOdd']
        dynamic_metrics = ['Tol', 'Dev', 'Cou']
        return FairnessEvaluator(static_metrics, dynamic_metrics, args)

    raise ValueError(f"Evaluator '{evaluator_name}' is not found.")
