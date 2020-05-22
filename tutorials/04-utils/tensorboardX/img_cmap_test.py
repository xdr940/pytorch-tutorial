from tensorboardX import SummaryWriter

import  torch
import matplotlib.pyplot as plt
from random import  *
from path import Path



from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import numpy as np

def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0,1,low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0,max_value,resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:,i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}



def tensor2array(tensor, max_value=None, colormap='rainbow',out_shape = 'CHW'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array[:,:,:3]
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        if (tensor.size(0) == 3):
            array = 0.5 + tensor.numpy()*0.5
        elif (tensor.size(0) == 2):
            array = tensor.numpy()

    if out_shape == 'HWC':
        array = array.transpose(1,2,0)
    return array

def tensor2array2(tensor,min=0,max=None):
    if max==None:
        max = tensor.max().item()

    if len(tensor.shape) ==2:
        tensor = tensor.unsqueeze(0)
    arr=  (tensor/max).cpu().numpy()
    return arr




if __name__ == '__main__':


    tb_save_dir = Path('./log_dir')
    tb_save_dir.mkdir_p()

    writer = SummaryWriter(log_dir=tb_save_dir)


    img = torch.linspace(start=0,end=99,steps=100).reshape([10,10])
    img2 = torch.randint(low=0,high=4,size=[10,10])
    img3 = torch.randint(low=0,high=2,size=[10,10])
    img4 = torch.linspace(start=0,end=1,steps=100).reshape([10,10])
    img5 = torch.randint(low=0,high=255,size=[10,10])
    img_ls = [img,img2,img3,img4,img5]

    row = 2
    col=3
    cnt =1
    for img in img_ls:
        img_t = tensor2array(img,colormap='magma',out_shape='CHW')
        writer.add_image('img',img_t,cnt)


        plt.subplot(row, col, cnt)
        img=img.cpu().numpy()
        plt.imshow(img,cmap='cool')

        cnt+=1

    writer.close()

    plt.show()
    print('ok')


