from .protocol import IEvaluator


class _BaseEvaluator:

    def __init__(self, evaluator: IEvaluator):
        self.evaluator = evaluator