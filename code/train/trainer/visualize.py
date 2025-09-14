
from code.train.trainer.base_trainer import Base_trainer

class Visualize(Base_trainer):
    def __init__(self, model, args, kind=None, writer=None, optimizer=None, scheduler=None, scaler=None):
        super().__init__(model, optimizer, scheduler, scaler, writer, args, kind)



