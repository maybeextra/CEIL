from code.tools.eavl.eval_metrics import eval_sysu, eval_regdb, eval_llcm
from code.tools.eavl.get_test_data import creat_test_data_query,creat_test_data_gall
import logging
import torch
from code.train.trainer.base_trainer import Base_trainer
from code.tools import rerank

class Visualize(Base_trainer):
    def __init__(self, model, args, kind=None, writer=None, optimizer=None, scheduler=None, scaler=None):
        super().__init__(model, optimizer, scheduler, scaler, writer, args, kind)



