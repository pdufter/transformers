import sys
sys.path.append("/mounts/work/mzhao/master-pse/pse-bert")

from finetune_ops.dataloaders import PTBDataset
from finetune_ops.trainer import TaggingTrainer
#from finetune_ops.predictors import POSTagger
from predictors import POSTagger
from finetune_ops.helpers import read_ptb
from core.loader import wrap_dataloader
from core import ValidationStateRecorder, GradientNormMonitor
from cons import BERT_MODEL, SEED
from finetune_sst2 import limit_params
from kitz.ios.ops import load_pkl_from

import torch
import numpy as np

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def main():
    num_cards, finetune = 1, True
    max_epoch, eval_every_batch = 5, 1
    task, optmizer = "PTB", "adam"
    log_fn, bs, msl, lr = print, 10, 128, 0 # 5e-5
    
    trn_sents, val_sents, t2i = read_ptb()
    trn_dl = wrap_dataloader(PTBDataset(exp_type="val", tagged_sents=trn_sents, 
        t2i=t2i, msl=msl), bs=bs)
    val_dl = wrap_dataloader(PTBDataset(exp_type="val", tagged_sents=val_sents, 
        t2i=t2i, msl=msl), bs=bs)
    
    model = POSTagger.from_pretrained(BERT_MODEL, o_dim=len(t2i))
    validation_results = load_pkl_from("/mounts/work/mzhao/pse-bert/reprob_ops/legacy_ptb/validation_state_PTB.pkl")
    model.load_state_dict(validation_results["state_dict"])

    trainer = TaggingTrainer(
        cri=torch.nn.CrossEntropyLoss(),
        num_cards=num_cards, log_fn=log_fn,
        task=task, bs=bs, model=model,
        trn_dl=trn_dl, val_dl=val_dl, tst_dl=None)
    
    # validation_recorder = ValidationStateRecorder(
    #     where_="./workspace/temp/validation_state_{}.pkl".format(task))
    # gradient_monitor = GradientNormMonitor(
    #     grad_where_="./workspace/temp/selfattn_gradients_{}.pkl".format(task),
    #     weight_where_="./workspace/temp/selfattn_weights_{}.pkl".format(task)
    # )
    # hooks = [validation_recorder, gradient_monitor]
    hooks = []
    trn_names, trn_params = limit_params(trainer, log_fn, finetune)
    trainer.run(trn_params, optmizer, lr, 
            max_epoch, eval_every_batch, 
            hooks=hooks)


if __name__ == "__main__":
    main()
