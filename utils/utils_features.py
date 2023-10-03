import time
import numpy as np
import torch
from torch.nn import functional as F

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)

    return img_flip

@torch.no_grad()
def extract_feature(model, dataloader):
    features, ids = [], []
    for batch_idx, (imgs, batch_labels) in enumerate(dataloader):
        if batch_idx % 100 == 0:
            print("batch_idx: ",batch_idx)
        flip_imgs = fliplr(imgs)
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs).data.cpu()
        batch_features_flip = model(flip_imgs).data.cpu()
        batch_features += batch_features_flip
        features.append(batch_features)
        ids.append(batch_labels)
    features = torch.cat(features, 0)
    ids = torch.cat(ids, 0).numpy()

    return features, ids

def save_feat(model, trainloader, save_dir):
    since = time.time()
    model.eval()
    # Extract features for train set
    tf, t_ids = extract_feature(model, trainloader)
    print("Extracted features for train set, obtained {} matrix".format(tf.shape))
    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    tf = F.normalize(tf, p=2, dim=1)

    features = {}
    for i, id in enumerate(t_ids):
        features.setdefault(id, []).append(tf[i].cpu().numpy())
        # set_default:如果键不存在于字典中，将会添加键并将值设为默认值
        # 如果没有pid，就创建一个以该pid为key，以[]为value，并将拥有相同pid的tf加入到该list中

    # generate pid class centers
    old_class_centers = []
    old_class_cos = []
    id_list = sorted(list(set(t_ids)))  # 将t_ids去重并按从小到大排序
    for id in id_list:  # pid=person id 代表每一个类别
        local_features = features[id]  # pid_features中按照pid存储每个人的不同图像的features
        # class center
        local_class_center = np.mean(np.array(local_features), axis=0)
        local_class_center = local_class_center / np.linalg.norm(local_class_center)
        old_class_centers.append(local_class_center)  # 将每个人中心点依次存入old_class_centers
        # cos values
        old_class_cos.append(min(np.dot(local_features, local_class_center)))  # 存储最小的cos distance，作为该类的boundary

    # save
    np.save('{}/old_class_centers.npy'.format(save_dir), np.array(old_class_centers))
    np.save('{}/old_class_cos.npy'.format(save_dir), np.array(old_class_cos))
    return
