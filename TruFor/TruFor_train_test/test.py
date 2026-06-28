# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro
"""

import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
from torch.nn import functional as F

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

from lib.config import config, update_config
from lib.utils import get_model, get_device
from dataset.dataset_test import TestDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test TruFor')
    parser.add_argument('-g',   '--gpu',     type=int, default=0, help='device, use -1 for cpu')
    parser.add_argument('-in',  '--input',   type=str, default='../images', help='can be a single file, a directory or a glob statement')
    parser.add_argument('-out', '--output',  type=str, default='../output', help='output folder')
    parser.add_argument('-exp', '--experiment', type=str, default='trufor_ph3')
    parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
    parser.add_argument('-w',   '--workers', type=int, default=0,
                        help='dataloader workers that decode images in parallel with inference; worth enabling only for long runs')
    parser.add_argument('-mp',  '--max-pixels', type=int, default=0, dest='max_pixels',
                        help='downscale images above this pixel count before inference and upsample the maps back (0 = full resolution; multi-megapixel images can exceed the memory of smaller machines)')
    parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def collect_images(input):
    if '*' in input:
        list_img = glob(input, recursive=True)
        list_img = [img for img in list_img if not os.path.isdir(img)]
    elif os.path.isfile(input):
        list_img = [input]
    elif os.path.isdir(input):
        list_img = glob(os.path.join(input, '**/*'), recursive=True)
        list_img = [img for img in list_img if not os.path.isdir(img)]
    else:
        raise ValueError("input is neither a file or a folder")
    return list_img


def output_path(img_path, input, output):
    if os.path.splitext(os.path.basename(output))[1] == '':  # output is a directory
        root = input.split('*')[0]

        if os.path.isfile(input):
            sub_path = img_path.replace(os.path.dirname(root), '').strip()
        else:
            sub_path = img_path.replace(root, '').strip()

        # strip any leading separator(s); on Windows a leading '\' makes the
        # os.path.join below treat sub_path as drive-absolute and drop `output`,
        # writing the .npz to the root of C:\ (PermissionError)
        sub_path = sub_path.lstrip('/\\')

        filename_out = os.path.join(output, sub_path) + '.npz'
    else:  # output is a filename
        filename_out = output

    if not filename_out.endswith('.npz'):
        filename_out = filename_out + '.npz'

    return filename_out


def main():
    args = parse_args()
    update_config(config, args)

    # Auto-select CUDA / MPS / CPU (gpu < 0 forces CPU); portable across machines.
    device = get_device(args.gpu)

    if device.type == 'cuda':
        # cudnn setting (only applies to CUDA)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED

    list_img = collect_images(args.input)

    # by default it does not overwrite: drop already-processed images up front,
    # so they are never decoded at all
    out_paths = {img: output_path(img, args.input, args.output) for img in list_img}
    todo = [img for img in list_img if not os.path.isfile(out_paths[img])]
    if len(todo) < len(list_img):
        print('=> skipping {} image(s) with existing output'.format(len(list_img) - len(todo)))
    if not todo:
        return

    test_dataset = TestDataset(list_img=todo, max_pixels=args.max_pixels)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,   # 1 to allow arbitrary input sizes
        num_workers=min(args.workers, len(todo)),
        pin_memory=(device.type == 'cuda'))

    if not config.TEST.MODEL_FILE:
        raise ValueError("Model file is not specified.")

    print('=> loading model from {}'.format(config.TEST.MODEL_FILE))
    # load on CPU and move once, instead of bouncing the weights device -> CPU -> device
    checkpoint = torch.load(config.TEST.MODEL_FILE, map_location='cpu', weights_only=False)
    print("Epoch: {}".format(checkpoint['epoch']))

    model = get_model(config)
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    model = model.to(device).eval()

    with torch.inference_mode():
        for rgb, img_path, orig_size in tqdm(testloader):
            filename_out = out_paths[img_path[0]]
            if os.path.isfile(filename_out):
                continue

            try:
                rgb = rgb.to(device, non_blocking=True)
                orig_size = (int(orig_size[0, 0]), int(orig_size[0, 1]))

                pred, conf, det, npp = model(rgb, save_np=args.save_np)

                pred = torch.squeeze(pred, 0)
                pred = torch.softmax(pred, dim=0)[1]
                if pred.shape != orig_size:  # image was downscaled by max_pixels
                    pred = F.interpolate(pred[None, None], size=orig_size, mode='bilinear', align_corners=False)[0, 0]

                out_dict = dict()
                out_dict['map'    ] = pred.cpu().numpy()
                out_dict['imgsize'] = orig_size
                if det is not None:
                    out_dict['score'] = torch.sigmoid(det).item()
                if conf is not None:
                    conf = torch.sigmoid(torch.squeeze(conf, 0)[0])
                    if conf.shape != orig_size:
                        conf = F.interpolate(conf[None, None], size=orig_size, mode='bilinear', align_corners=False)[0, 0]
                    out_dict['conf'] = conf.cpu().numpy()
                if args.save_np and npp is not None:
                    out_dict['np++'] = torch.squeeze(npp, 0)[0].cpu().numpy()

                dir_out = os.path.dirname(filename_out)
                if dir_out:
                    os.makedirs(dir_out, exist_ok=True)
                np.savez(filename_out, **out_dict)
            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                if device.type == 'mps':
                    # MPS caches freed buffers at the high-watermark forever; on unified
                    # memory that starves the rest of the system, so release per image
                    # (after dropping the loop's references to the device tensors)
                    rgb = pred = conf = det = npp = None
                    torch.mps.empty_cache()


if __name__ == '__main__':
    main()
