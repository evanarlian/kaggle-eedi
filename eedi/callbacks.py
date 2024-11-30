from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from eedi.my_datasets import TrainDatasetProxy, hn_mine_sbert


class IterativeHNMiningCallback(TrainerCallback):
    def __init__(self, bs: int, top_k_negatives: int):
        """Iterative hard negative mining callback. Somewhat hacky way to change the
        dataset mid training.

        Args:
            bs (int): Batch size for hn mining
            top_k_negatives (int): The size of hard samples
        """
        self.bs = bs
        self.top_k_negatives = top_k_negatives

    # TODO check if en epoch end enough? is this too often? we can modify this using TrainerState
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        *,
        train_dataloader: DataLoader,
        model: SentenceTransformer,
        **kwargs,
    ):
        print("Starting hard negative mining...")
        curr_train_state = model.training  # save model .train() or .eval() state
        train_ds: TrainDatasetProxy = train_dataloader.dataset  # type: ignore
        model.eval()
        show_tqdm = state.is_local_process_zero
        new_hards = hn_mine_sbert(
            model=model,
            q_texts=train_ds.q_texts,
            q_mis_ids=train_ds.q_mis_ids,
            mis_texts=train_ds.mis_texts,
            mis_ids=train_ds.mis_ids,
            k=self.top_k_negatives,
            bs=self.bs,
            tqdm=show_tqdm,
        )
        train_ds.replace_hards(new_hards)
        model.train(curr_train_state)  # return back the model state
