from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from mq.data.data_module import QuantizeDataModule
from mq.trainer import Wav2TTS
from pytorch_lightning.strategies import DDPStrategy
import json


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

args = AttrDict()

arg_dict = json.load(open('configs/s1.json', 'r'))
args.__dict__.update(arg_dict)

fname_prefix = f''

if args.accelerator == 'ddp':
    args.accelerator = DDPStrategy(find_unused_parameters=False)

checkpoint_callback = ModelCheckpoint(
    dirpath=args.saving_path,
    filename=(fname_prefix+'{epoch}-{step}'),
    every_n_train_steps=(None if args.val_check_interval == 1.0 else args.val_check_interval),
    every_n_epochs=(None if args.check_val_every_n_epoch == 1 else args.check_val_every_n_epoch),
    verbose=True,
    save_last=True
)

logger = TensorBoardLogger(args.sampledir, name="VQ-TTS", version=args.version)
lr_monitor = LearningRateMonitor(logging_interval='step')

wrapper = Trainer(
    precision=args.precision,
    callbacks=[checkpoint_callback, lr_monitor],
    val_check_interval=args.val_check_interval,
    num_sanity_val_steps=0,
    max_steps=args.training_step,
    devices=(-1 if args.distributed else 1),
    strategy=(args.accelerator if args.distributed else None),
    use_distributed_sampler=False,
    accumulate_grad_batches=args.accumulate_grad_batches,
    logger=logger,
    check_val_every_n_epoch=None
)

training_wrapper = Wav2TTS(args)
if args.pretrained_path:
    training_wrapper.load()
data_module = QuantizeDataModule(args)
wrapper.fit(training_wrapper, datamodule=data_module, ckpt_path=args.resume_checkpoint)
