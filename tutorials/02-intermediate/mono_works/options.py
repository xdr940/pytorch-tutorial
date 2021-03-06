# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
from path import Path
file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MD_train_opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 training options")
        self.parser.add_argument("--num_epochs", type=int, help="number of epochs", default=5)
        self.parser.add_argument("--weights_save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=10)


        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 #default="/home/roit/models/monodepth2/identical_var_mean/last_model",
                                 default='/home/roit/models/monodepth2_official/mono_640x192',
                                 # default='/media/roit/hard_disk_2/Models/monodepth2/04-23-00:50/models/weights_10',#继续训练
                                 #default='/home/roit/models/monodepth2/fullwitherodil/last_model',
                                 #default='/home/roit/models/monodepth2/reproduction/models/weights_19',
                                 #default='/home/roit/models/monodepth2/var_id/models/weights_19',
                                 #default='/home/roit/models/monodepth2/05-20-17:20/models/weights_19',

                                 help="name of model to load, if not set, train from imgnt pretrained")
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default='/home/roit/datasets/kitti/')
                                 #default="/home/roit/datasets/MC")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 #default='/home/roit/models/monodepth2/checkpoints'
                                 default = '/home/roit/models/monodepth2/eval_test'
                                  )


        self.parser.add_argument("--masks",
                                 default=['identity_selection',
                                          'identity_selection_new',
                                          'final_selection',
                                          'var_mask',
                                          'mean_mask',
                                          'map_12',
                                          'map_34'
                                         ])
        self.parser.add_argument('--root',type=str,
                                 help="project root",
                                 default='/home/roit/aws/aprojects/xdr94_mono2')

        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 #default="mc",
                                 default='kitti',
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "mc"])


        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 default=True,
                                 action="store_true")
        self.parser.add_argument("--height",type=int,help="input image height",default=192)
        self.parser.add_argument("--width",type=int,help="input image width",default=640)


        self.parser.add_argument("--scales",nargs="+",type=int,help="scales used in the loss",default=[0, 1, 2, 3])

        self.parser.add_argument("--min_depth",type=float,help="minimum depth",default=0.1)#这里度量就代表m
        self.parser.add_argument("--max_depth",type=float,help="maximum depth",default=80.0)
        self.parser.add_argument("--dleta",default=0.01,help="variance threashold")
        #self.parser.add_argument("--use_stereo",help="if set, uses stereo pair for training",action="store_true")
        self.parser.add_argument("--frame_ids",nargs="+",type=int,help="frames to load",default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",type=int,help="batch size",default=4)#
        self.parser.add_argument("--learning_rate",type=float,help="learning rate",default=1e-4)
        self.parser.add_argument("--start_epoch",type=int,help="for subsequent training",
                                 #default=10,
                                 default=0,

                                 )

        self.parser.add_argument("--scheduler_step_size",type=int,help="step size of the scheduler",default=15)

        # LOADING args for subsquent training or train from pretrained/scratch

        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load, for training or test",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch or subsequent training from last",
                                 default="scratch",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--encoder_path",
                                 type=str,
                                 help="pretrained from here",
                                 default="/home/roit/models/torchvision/official/resnet18-5c106cde.pth",
                                 )





        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)



        # LOGGING options
        self.parser.add_argument("--tb_log_frequency",
                                 type=int,
                                 help="number of batches(step) between each tensorboard log",
                                 default=5)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options

class MD_eval_opts:
    def __init__(self):
        # EVALUATION options
        self.parser = argparse.ArgumentParser(description="Monodepthv2 evaluation options")
        self.parser.add_argument('--root', type=str,
                                 default='/home/roit/aws/aprojects/xdr94_mono2')
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default='/home/roit/datasets/kitti/')
        # default="/home/roit/datasets/MC")
        self.parser.add_argument("--depth_eval_path",
                                 help="",
                                 #default='/home/roit/models/monodepth2_official/mono_640x192',#官方给出的模型文件夹
                                 #default="/home/roit/models/monodepth2/reproduction/models/weights_4",
                                 #default="/home/roit/models/monodepth2/pure_var_mask_median/models/weights_19",
                                 #default="/home/roit/models/monodepth2/avg_loss/models/weights_19",
                                 #default="/home/roit/models/monodepth2/identicalmask/models/weights_4",
                                 #default="/home/roit/models/monodepth2/identical+var/models/weights_7",
                                 #default="/home/roit/models/monodepth2/identical_var_mean/last_model",
                                 #default="/home/roit/models/monodepth2/fullwitherodil/last_model",
                                 #default="/home/roit/models/monodepth2/fullwitherodil10epoch/last_model",
                                 #default="/home/roit/models/monodepth2/05-14-13:57/models/weights_4"#
                                 #default="/home/roit/models/monodepth2/checkpoints/05-14-19:27/models/weights_19"#
                                 #default="/home/roit/models/monodepth2/checkpoints/05-16-23:53/models/weights_9"
                                 #default = "/home/roit/models/monodepth2/checkpoints/05-15-14:40/models/weights_19"
                                 # default = "/home/roit/models/monodepth2/checkpoints/05-15-14:40/models/weights_19"
                                 #default="/home/roit/models/monodepth2/checkpoints/05-18-00:34/models/weights_4"
                                 #default="/home/roit/models/monodepth2/checkpoints/05-20-00:06/models/weights_19"
                                 default = "/home/roit/models/monodepth2/checkpoints/05-20-17:20/models/weights_19"

        )
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",  # eigen
                                 choices=["eigen_zhou","eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10", "custom", "mc"],
                                 help="which split to run eval on")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])

        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 default=True,
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)


        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir", default='eval_out_dir',
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",  # ??
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

        self.parser.add_argument("--eval_pose_data_path",
                                 default='/media/roit/hard_disk_2/Datasets/kitti_odometry_color')

        self.parser.add_argument("--eval_pose_save_path", default="./")
        self.parser.add_argument("--eval_batch_size", default=16, type=int)
        self.parser.add_argument("--eval_odom_batch_size", default=16, type=int)


        self.parser.add_argument("--min_depth",type=float,help="minimum depth",default=0.1)#这里度量就代表m
        self.parser.add_argument("--max_depth",type=float,help="maximum depth",default=80.0)
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options