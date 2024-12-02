from typing import Literal

from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from eedi.datasets import TrainDataset, hn_mine_hf


class IterativeHNMiningCallback(TrainerCallback):
    def __init__(
        self,
        bs: int,
        top_k_negatives: int,
        token_pool: Literal["first", "last"],
    ):
        """Iterative hard negative mining callback. Somewhat hacky way to change the
        dataset mid training.

        Args:
            bs (int): Batch size for hn mining
            top_k_negatives (int): The size of hard samples
            token_pool (str, {"first", "last"}): Which token to use as sentence emb.
        """
        self.bs = bs
        self.top_k_negatives = top_k_negatives
        self.token_pool: Literal["first", "last"] = token_pool

    def on_epoch_start(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        *,
        train_dataloader: DataLoader,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs,
    ):
        print("Starting hard negative mining...")
        curr_train_state = model.training  # save model .train() or .eval() state
        train_ds: TrainDataset = train_dataloader.dataset  # type: ignore
        device = next(model.parameters()).device  # TODO idk check!!
        model.eval()
        new_hards = hn_mine_hf(
            model=model,
            tokenizer=tokenizer,
            q_texts=train_ds.q_texts,
            q_mis_ids=train_ds.q_mis_ids,
            mis_texts=train_ds.mis_texts,
            mis_ids=train_ds.mis_ids,
            k=self.top_k_negatives,
            bs=self.bs,
            token_pool=self.token_pool,
            device=device,
            tqdm=state.is_local_process_zero,
        )
        train_ds.replace_hards(new_hards)
        model.train(curr_train_state)  # return back the model state
