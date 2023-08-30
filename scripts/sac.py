
from argparse import ArgumentParser
import pytorch_lightning as pl 

from rl_lightning.algos import SAC

parser = ArgumentParser()
parser = SAC.add_model_specific_args(parser)
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--accelerator", type=str, default="gpu")
args = parser.parse_args()
dict_args = vars(args)
agent = SAC(**dict_args)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = "./checkpoints",
    verbose = True,
    filename = "_".join(["sac", args.env, ".pt"]),
    monitor = "last_episode_score",
    save_top_k = 1,
    save_weights_only = True,
    mode = "max" # only pick max of `score`
)

trainer = pl.Trainer(
    accelerator=args.accelerator, 
    devices=args.devices, 
    callbacks=[checkpoint_callback],
    max_epochs=-1,
    enable_progress_bar=False, 
    limit_val_batches=0)

trainer.fit(agent)
