import os
import argparse
from glob import glob
import numpy as np
from torch import device

from model import RetinexNet
import torch
from PIL import Image
from torch.autograd import Variable

import  torch.nn as nn

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

    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])

    for folder in folder_list:
        folder_name = test_folder + '/' + folder
        file_list = glob(folder_name + "/*")

        for image in file_list:
            image = os.path.normpath(image)
            # model.test(image,
            #     res_dir=args.res_dir,
            #     ckpt_dir=args.ckpt_dir)

            # Set this switch to True to also save the reflectance and shading maps
            save_R_L = False

            # Predict for the test images

            test_img_path = image
            test_img_name = test_img_path.split('/')[-1]
            print('Processing ', test_img_name)
            test_low_img = Image.open(test_img_path)
            test_low_img = np.array(test_low_img, dtype="float32") / 255.0
            test_low_img = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            input_low_test = Variable(torch.FloatTensor(torch.from_numpy(input_low_test))).cuda()
            input_low_test = input_low_test.to(device)
            result_4 = model.forward(input_low_test)
            del input_low_test
            result_4 = np.squeeze(result_4)
            # if save_R_L:
            #     cat_image = np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)
            # else:
            #     cat_image = np.concatenate([input, result_4], axis=2)
            result_4 = np.concatenate([result_4], axis=2)

            result_4 = np.transpose(result_4, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(result_4 * 255.0, 0, 255.0).astype('uint8'))
            del result_4

            image_folder_name = test_img_path.split('/')[-2]
            image_name = test_img_path.split("/")[-1]
            folder_name = 'result/' + image_folder_name + '/'
            result_path = folder_name + image_name

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            filepath = args.res_dir + '/' + test_img_path.split('/')[-2] + test_img_name
            im.save(folder_name + image_name)

if __name__ == '__main__':
    if args.gpu_id != "-1":
        print("任务开始")
        # Create directories for saving the results
        if not os.path.exists(args.res_dir):
            os.makedirs(args.res_dir)
        # Setup the CUDA env
        device_ids = [0, 1]
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

        # Create the model
        model = RetinexNet()

        model.train_phase = 'Decom'
        load_model_status, _ = model.load(args.ckpt_dir)
        if load_model_status:
            print(model.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception
        model.train_phase = 'Relight'
        load_model_status, _ = model.load(args.ckpt_dir)
        if load_model_status:
            print(model.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception


        # Test the model
        test(model)
        # predict(model)
    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError


