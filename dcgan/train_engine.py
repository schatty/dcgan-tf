import logging
from tqdm import tqdm
import numpy as np


class TrainEngine(object):
    """
    Engine that launches training per epochs and episodes.
    Contains hooks to perform certain actions when necessary.
    """
    def __init__(self):
        self.hooks = {name: lambda state: None
                      for name in ['on_start',
                                   'on_start_epoch',
                                   'on_end_epoch',
                                   'on_start_batch',
                                   'on_end_batch',
                                   'on_end']}

    def train(self, loader, epochs, batch, **kwargs):
        # State of the training procedure
        state = {
            'loader': loader,
            'sample': None,
            'epoch': 1,
            'epochs': epochs,
            'batch': batch,
            'step': 0
        }

        self.hooks['on_start'](state)
        for epoch in range(state['epochs']):
            self.hooks['on_start_epoch'](state)
            for batch_images in state['loader'].get_batches(batch):
                #print("Epoch ", state['epoch'], "Batch+1")
                state['sample'] = batch_images
                self.hooks['on_start_batch'](state)
                self.hooks['on_end_batch'](state)
                state['step'] += 1

            self.hooks['on_end_epoch'](state)
            state['epoch'] += 1

        self.hooks['on_end'](state)
        logging.info("Training succeed!")
