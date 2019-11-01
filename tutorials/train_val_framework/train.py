import time
from utils.logger import AverageMeter
from loss_functions import loss_func
def train(logger):
    train_loss_names = ['train1', 'train2']
    train_losses = AverageMeter(precision=2, i=len(train_loss_names))
    batch_time = AverageMeter()
    end = time.time()
    for batch_i in range(logger.valid_size):
        time.sleep(0.2)
        batch_time.update(time.time() - end)
        end = time.time()
        losses = loss_func()  # 2 losses
        train_losses.update(losses)

        logger.train_logger_update(batch=batch_i, time=batch_time,names=train_loss_names,values=train_losses)

    return train_loss_names, train_losses