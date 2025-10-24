import hydra
import torch.distributed as dist
from tqdm import tqdm
from RL2.trainer import Trainer
from RL2.datasets import SFTDataset, get_dataloader
from RL2.workers import initialize_actor
from RL2.utils.communication import initialize_global_process_group


class SFTTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)

        self.actor = initialize_actor(config.actor, True)
        dataset = SFTDataset(
            config.data, self.actor.tokenizer
        )
        self.train_dataloader = get_dataloader(
            dataset, config.data.batch_size
        )
        self.actor.prepare_scheduler(
            self.config.trainer.n_epochs * len(self.train_dataloader)
        )

    def train(self):

        step = self.load_ckpt((self.actor,))

        # Optional: limit the number of batches per epoch for faster iterations
        steps_per_epoch_cfg = getattr(self.config.trainer, "steps_per_epoch", None)

        for epoch in range(
            step // len(self.train_dataloader),
            self.config.trainer.n_epochs
        ):
            total_batches = len(self.train_dataloader)
            steps_per_epoch = (
                min(steps_per_epoch_cfg, total_batches)
                if steps_per_epoch_cfg is not None else total_batches
            )

            batches_processed = 0
            for tensor_dict in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0),
                total=steps_per_epoch,
                initial=step % steps_per_epoch
            ):
                step += 1
                batches_processed += 1
                self.actor.sft_update(tensor_dict, step)
                self.save_ckpt((self.actor,), step)
                if batches_processed >= steps_per_epoch:
                    break
        self.save_model((self.actor,))


@hydra.main(config_path="config", config_name="sft", version_base=None)
def main(config):

    initialize_global_process_group()

    trainer = SFTTrainer(config)
    trainer.train()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
