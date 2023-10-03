import argparse
import cv2
import numpy as np
import torch
from backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    img = img.cuda()
    net = get_model(name, fp16=False).cuda()
    net.load_state_dict(torch.load(weight))
    net.eval()
    reso=112
    print(reso.dtype)
    print(reso.device)
    feat = net(img, [reso]).cpu()
    feat = feat.numpy()
    print(feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='output/ms1mv3_r50_reso112/model.pt')
    parser.add_argument('--img', type=str, default='imgs/face.jpg')
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)
