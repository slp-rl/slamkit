import logging
logger = logging.getLogger(__name__)

from transformers import TrainerCallback
import time
from typing import Union



class RunTimeStopperCallback(TrainerCallback):
    """
    A callback that stops training after a certain amount of time has passed
    """
    def __init__(self, run_time: Union[str,int]):
        """
        Args:
            run_time (str): The time to run the training for in the format "days-hours:minutes:seconds
        """
        if isinstance(run_time, int):
            self.run_time = run_time
            return
        
        days = 0
        if "-" in run_time:
            days,run_time = run_time.split("-")
            days = int(days)
        hours, minutes, seconds = run_time.split(":")
        self.run_time = days*24*60*60 + int(hours)*60*60 + int(minutes)*60 + int(seconds)

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info(f"Training will run for {self.run_time} seconds")
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.start_time > self.run_time:
            control.should_training_stop = True
            control.should_evaluate = True
            control.should_save = True
            logger.info(f"Stopping training as it has run for {self.run_time} seconds")


class MaxTokensStopperCallback(TrainerCallback):
    def __init__(self, train_max_tokens: int):
        self.max_tokens = train_max_tokens

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info(f"Training will run for {self.max_tokens} tokens according to specified range if provided")

    def on_step_end(self, args, state, control, **kwargs):
        if state.num_input_tokens_seen >= self.max_tokens:
            control.should_training_stop = True
            control.should_evaluate = True
            control.should_save = True
            logger.info(f"Stopping training as it has seen {state.num_input_tokens_seen} tokens according to specified range if provided")