import torch
import os
from torch.utils.tensorboard import SummaryWriter
#from optim import Adam
from torch.optim import AdamW
from torch import nn
from optim import NoamOpt
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, args, model, tokz, train_dataset, valid_dataset,
                 log_dir, logger, device=torch.device('cuda'), valid_writer=None, distributed=False):
        self.config = args
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.tokz = tokz
        self.rank = torch.distributed.get_rank() if distributed else -1
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
        if valid_writer is None:
            self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid'))
        else:
            self.valid_writer = valid_writer
        self.model = model.to(device, non_blocking=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokz.pad_token_id, reduction='None').to(device)

        base_optimizer = AdamW(self.momdel.parameters(), lr=self.config.lr, weight_decay=0.01)

        if hasattr(self.model, 'config'):
            self.optimizer = NoamOpt(self.model.config.hidden_size, 0.1, self.config.lr_warmup, base_optimizer)
        else:
            self.optimizer = NoamOpt(self.model.module.config.hidden_size, 0.1, self.config.lr_warmup, base_optimizer)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else torch.utils.data.RandomSampler(train_dataset)
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if distributed else None

        self.train_dataloader = DataLoader(
            train_dataset,
            sampler=self.train_sampler,
            batch_size=self.config.bs,
            num_workers=self.config.n_jobs,
            pin_memory=True
        )
