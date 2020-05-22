
import path
from tqdm import tqdm
import numpy as np
import torch
height=16
width=64
num_maps=4
#generate_min_dataset

def generate(out_path,num):
    error_maps = np.random.random_sample([num_maps,height,width])
    error_maps = torch.tensor(error_maps)
    min_map,idx = torch.min(error_maps,dim=0)
    out = torch.cat([error_maps.float(),idx.unsqueeze(dim=0).float()],dim=0)
    np.save(out_path/'{:03d}'.format(num),out.numpy())
if __name__=="__main__":
    out_path = path.Path('./data')
    out_path.mkdir_p()


    for i in tqdm(range(100)):
        generate(out_path,i)



