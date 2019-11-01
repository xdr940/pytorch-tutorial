import time
import random
from utils.logger import AverageMeter
from loss_functions import loss_func

def val(logger):
    val_loss_names=['val1','val2']
    val_loss_errs=AverageMeter(precision=2,i=len(val_loss_names))
    batch_time = AverageMeter()
    end=time.time()
    for batch_i in range(logger.valid_size):
        time.sleep(0.2)
        batch_time.update(time.time() - end)
        end = time.time()
        err = loss_func()
        val_loss_errs.update(err)
        #if term_log

        logger.valid_logger_update(batch=batch_i,time=batch_time,names=val_loss_names,values=val_loss_errs)


    return val_loss_names,val_loss_errs.avg
