from options import MonodepthOptions
from trainer import Trainer

options = MonodepthOptions()
opts = options.parse()

opts.data_path = 'images/'

##########
opts.log_frequency = 125
opts.num_epochs = 49
##########
if __name__ == "__main__":

    trainer = Trainer(opts)
    trainer.train()
