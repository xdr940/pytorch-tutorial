import os
import numpy as np
import shutil
import torch


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        src = os.path.join(save_path,filename)
        dst = os.path.join(save_path,'model_best.pth.tar')
        shutil.copyfile(src,dst)#把当前最好的覆盖写


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)

def paras_info_print(model):
    """
    import numpy as np
    :param model:
    :return:None
    """
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = torch.FloatTensor().element_size()  # 返回单个元素的字节大小. 这里32bit 4B
    print('Model {} : param_num: {:4f}M  param_size: {:4f}MB '.format(model._get_name(), para / 1000 / 1000,
                                                                      para * type_size / 1000 / 1000))
