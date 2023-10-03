# Visualization of feature maps

import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from backbones import get_model
from pylab import *


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col

def visualize_feature_map(img_batch, name):

    feature_map = img_batch.squeeze(0)
    print(feature_map.shape)
    feature_map_combination = []
    num_pic = feature_map.shape[0]

    for i in range(0, num_pic):
        feature_map_split = feature_map[i, :, :]
        feature_map_combination.append(feature_map_split)

    plt.figure()
    # 1：1 aggregation
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    axis('off')
    plt.savefig("%s.png"% name )


@torch.no_grad()
def inference(weight, name, img, model_reso, reso, upsample):
    if img is None:
        img = np.random.randint(0, 255, size=(reso, reso, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (reso, reso))
    if upsample == 1:
        img = cv2.resize(img, (112,112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    img = img.cuda()
    net = get_model(name, resolution=model_reso, pretrained=False, pretrained_path=None, fp16=False).cuda()
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img)[1]
    feats = [f.cpu().numpy() for f in feat]
    names=['l1','l2','l3','l4']

    for feat,name in zip(feats,names):
        visualize_feature_map(feat, name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize feature maps of BTNet')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='output/ms1mv3_r50/reso112/model.pt')
    parser.add_argument('--img', type=str, default='imgs/face.jpg')
    parser.add_argument('--reso', type=int, default=112)
    parser.add_argument('--model_reso', type=int, default=112)
    parser.add_argument('--upsample', type=int, default=0, help='1: upsample the img to model_reso')
    args = parser.parse_args()
    inference(args.weight, args.network, args.img, args.model_reso,args.reso, args.upsample)