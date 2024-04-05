import os
import argparse
from glob import glob
import numpy as np
from model import RetinexNet

parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_id', dest='gpu_id', 
                    default="0",
                    help='GPU ID (-1 for CPU)')
parser.add_argument('--data_dir', dest='data_dir',
                    default='./data/test',
                    help='directory storing the test data')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', 
                    default='./ckpts/',
                    help='directory for checkpoints')
parser.add_argument('--res_dir', dest='res_dir', 
                    default='.\\results\\test\\low\\',
                    help='directory for saving the results')

args = parser.parse_args()

def predict(model):

    test_low_data_names  = glob(args.data_dir + '\\' + '*.*')
    test_low_data_names.sort()
    print('Number of evaluation images: %d' % len(test_low_data_names))

    model.predict(test_low_data_names,
                res_dir=args.res_dir,
                ckpt_dir=args.ckpt_dir)

def test(model):
    test_folder = args.data_dir
    folder_list = os.listdir(test_folder)

    for folder in folder_list:
        folder_name = test_folder + '/' + folder
        file_list = glob(folder_name + "/*")

        for image in file_list:
            image = os.path.normpath(image)
            model.test(image,
                res_dir=args.res_dir,
                ckpt_dir=args.ckpt_dir)

if __name__ == '__main__':
    print('开始运行')
    if args.gpu_id != "-1":
        # Create directories for saving the results
        if not os.path.exists(args.res_dir):
            os.makedirs(args.res_dir)
        # Setup the CUDA env
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        # Create the model
        model = RetinexNet().cuda()
        # 一机多卡设置
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 设置所有可以使用的显卡，共计四块
        device_ids = [0, 1]  # 选中其中两块
        import torch
        # model = torch.nn.DataParallel(model, device_ids=device_ids)  # 并行使用两块
        net = torch.nn.Dataparallel(model)  # 默认使用所有的device_ids
        model = model.cuda()
        # Test the model
        test(model)
        # predict(model)
    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError


