import datasets
import pytorch_lightning as pl

from mq.data.QuantizeDataset import QuantizeDataset
from mq.data.sampler import RandomBucketSampler
from torch.utils import data


class QuantizeDataModule(pl.LightningDataModule):
    def __init__(self, hp, data_dir='dump/phone_semantic_dataset'):
        super().__init__()
        self.hp = hp
        dataset_all = datasets.load_from_disk(data_dir, keep_in_memory=True)
        self.train_data = QuantizeDataset(hp, dataset_all['train'])
        self.val_data = QuantizeDataset(hp, dataset_all['test'])


    def train_dataloader(self):
        length = self.train_data.lengths
        sampler = RandomBucketSampler(self.hp.train_bucket_size, length, self.hp.batch_size, drop_last=True, distributed=self.hp.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank)
        dataset = data.DataLoader(self.train_data,
                                  num_workers=self.hp.nworkers,
                                  batch_sampler=sampler,
                                  collate_fn=self.train_data.seqCollate)
        return dataset

    def val_dataloader(self):
        dataset = data.DataLoader(self.val_data,
                                  num_workers=self.hp.nworkers,
                                  collate_fn=self.val_data.seqCollate,
                                  shuffle=False)
        return dataset