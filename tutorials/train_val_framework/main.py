from utils.logger import  TermLogger,AverageMeter
from validation import val
from train import train
import time
def main():
    '''
    TermLogger demo
    训练框架
    :return:
    '''
    epochs = 15
    train_size=10#batchs for train
    valid_size = 6

    logger = TermLogger(n_epochs=epochs,
                        train_size=train_size,
                        valid_size=valid_size)
    logger.reset_epoch_bar()


    #first val
    first_val = True
    val_losses = AverageMeter(precision=3)
    if first_val:
        val_names,val_losses = val(logger)
    else:
        val_loss = 0

    logger.reset_epoch_bar()
    #logger.epoch_logger_update(epoch=0,display)

    logger.epoch_bar.update(epoch=0)
    logger.epoch_writer.write('---\n---\n---')
    epoch_time = AverageMeter()



    end = time.time()
    for epoch in range(1,epochs):

        train_names,train_losses=train(logger)

        val_names,val_losses=val(logger)

        epoch_time.update(time.time()-end)
        end = time.time()


        logger.reset_train_bar()
        logger.reset_valid_bar()

        #if log_terminal
        logger.epoch_logger_update(epoch=epoch,time=epoch_time,names=val_names,values=val_losses)

    logger.epoch_bar.finish()
    print('over')
if __name__ =='__main__':
    main()