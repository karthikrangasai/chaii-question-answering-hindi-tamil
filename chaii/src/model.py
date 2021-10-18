import torch

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type, Union
from torch.optim.lr_scheduler import _LRScheduler

from flash.text import QuestionAnsweringTask


class ChaiiQuestionAnswering(QuestionAnsweringTask):
    def __init__(
        self,
        backbone: str = "distilbert-base-uncased",
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr_scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        metrics: Union[Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 5e-5,
        enable_ort: bool = False,
    ):
        super().__init__(
            backbone=backbone,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            learning_rate=learning_rate,
            enable_ort=enable_ort,
        )

    @staticmethod
    def jaccard(str1, str2):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    def compute_metrics(self, generated_tokens, batch):
        scores = []
        for example in batch:
            predicted_answer = generated_tokens[example["example_id"]]
            target_answer = (
                example["answer"]["text"][0]
                if len(example["answer"]["text"]) > 0
                else ""
            )
            scores.append(
                ChaiiQuestionAnswering.jaccard(predicted_answer, target_answer)
            )

        result = {"jaccard_score": torch.mean(torch.tensor(scores))}
        return result
