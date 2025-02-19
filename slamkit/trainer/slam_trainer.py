from typing import Union, Optional, Callable, List, Tuple, Dict
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    PreTrainedModel,
    EvalPrediction, 
    PreTrainedTokenizerBase, 
    BaseImageProcessor, 
    FeatureExtractionMixin, 
    ProcessorMixin
)
from transformers.data.data_collator import DataCollator
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
import datasets
from dataclasses import dataclass, field

@dataclass
class SLAMTrainingArguments(TrainingArguments):

    min_token_id_count: Optional[int] = field(default=None, metadata={"help": "Minimum token id to consider for calculating token seen count"})
    max_token_id_count: Optional[int] = field(default=None, metadata={"help": "Maximum token id to consider for calculating token seen count"})

class SLAMTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None, # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def get_num_tokens(self, labels):
        valid_tokens = labels != -100
        if self.args.min_token_id_count is not None:
            valid_tokens = torch.logical_and(valid_tokens, labels >= self.args.min_token_id_count)
        if self.args.max_token_id_count is not None:
            valid_tokens = torch.logical_and(valid_tokens, labels <= self.args.max_token_id_count)
        return valid_tokens.sum()
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        if "labels" in inputs:
            num_tokens_seen = self.get_num_tokens(inputs["labels"])
            self.state.num_input_tokens_seen += self.accelerator.gather(num_tokens_seen).sum().item()
        return super().training_step(model, inputs, num_items_in_batch)
