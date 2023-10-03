# Evaluation on IJB-C 1:1 face verification
# coding: utf-8

import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
import cv2
import numpy as np
import torch
from skimage import transform as trans
from backbones import get_model
from sklearn.metrics import roc_curve, auc

from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path

import sys
import warnings

sys.path.insert(0, "../")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='do ijb 11 test')
# general
parser.add_argument('--model-prefix1', default='output/ms1mv3_r50_reso112/model.pt', help='path to load model1.')
parser.add_argument('--model-prefix2', default='output/ms1mv3_r50_reso14/model.pt', help='path to load model2.')
parser.add_argument('--image-path', default='/dataset/IJBC/', type=str, help='path to IJB-C')
parser.add_argument('--result-dir', default='IJBC_11_result', type=str, help='path to save the results')
parser.add_argument('--batch-size', default=128, type=int, help='')
parser.add_argument('--network1', default='r50', type=str, help='')
parser.add_argument('--network2', default='r50', type=str, help='')
parser.add_argument('--model_reso1', default=112, type=int, help='')
parser.add_argument('--model_reso2', default=14, type=int, help='')
parser.add_argument('--reso1', default=112, type=int, help='')
parser.add_argument('--reso2', default=14, type=int,  help='')
parser.add_argument('--job', default='', type=str, help='job name')
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
parser.add_argument('--upsample1', default=1, type=int, help='')
parser.add_argument('--upsample2', default=1, type=int, help='')
parser.add_argument('--method_name',default='ours',type=str, help='')
parser.add_argument('--downsample1', default=1, type=int, help='')
parser.add_argument('--downsample2', default=0, type=int, help='')
args = parser.parse_args()

method_name = args.method_name
target = args.target
model_path1 = args.model_prefix1
model_path2 = args.model_prefix2
reso1 = (args.reso1,args.reso1)
reso2 = (args.reso2,args.reso2)
upsample1 = args.upsample1
upsample2 = args.upsample2
image_path = args.image_path
result_dir = args.result_dir
gpu_id = None
use_norm_score =False  # if Ture, TestMode(N1)
use_detector_score = True  # if Ture, TestMode(D1)
use_flip_test = True  # if Ture, TestMode(F1)
job = args.job
batch_size = args.batch_size


class Embedding(object):
    def __init__(self, prefix1, prefix2, data_shape, batch_size=1, reso1=(112,112), reso2=(112,112)):
        image_size = (112, 112)
        self.image_size = image_size
        self.reso1 = reso1
        self.reso2 = reso2
        

        ##### Models  #########
        weight1 = torch.load(prefix1)
        resnet1 = get_model(args.network1, dropout=0, fp16=False, resolution=args.model_reso1).cuda()
        resnet1.load_state_dict(weight1)
        model1 = torch.nn.DataParallel(resnet1)
        self.model1 = model1
        self.model1.eval()

        weight2 = torch.load(prefix2)
        resnet2 = get_model(args.network2, dropout=0, fp16=False, resolution=args.model_reso2).cuda()
        resnet2.load_state_dict(weight2)
        model2 = torch.nn.DataParallel(resnet2)
        self.model2 = model2
        self.model2.eval()


        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape

    def get(self, rimg, landmark):

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg,
                             M, (self.image_size[1], self.image_size[0]),
                             borderValue=0.0)
        img_tmp = img
        if self.reso1[1] != self.image_size[1] or self.reso1[0] != self.image_size[0]:
            img = cv2.resize(img, (self.reso1[1], self.reso1[0]))
            if upsample1 == 1:
                img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        if args.downsample1 == 1:
            img = cv2.resize(img, (args.model_reso1, args.model_reso1))
        if self.reso2[1] != self.image_size[1] or self.reso2[0] != self.image_size[0]:
            img_tmp = cv2.resize(img_tmp, (self.reso2[1], self.reso2[0]))
            if upsample2 == 1:
                img_tmp = cv2.resize(img_tmp, (self.image_size[1], self.image_size[0]))
        if args.downsample2 == 1:
            img_tmp = cv2.resize(img_tmp, (args.model_reso2, args.model_reso2))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)

        img_flip = np.fliplr(img_tmp)

        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))

        input_blob1 = img
        input_blob2 = img_flip

        return input_blob1, input_blob2

    @torch.no_grad()
    def forward_db(self, batch_data1, batch_data2):
        imgs1 = torch.Tensor(batch_data1).cuda()
        imgs1.div_(255).sub_(0.5).div_(0.5)
        feat1 = self.model1(imgs1)[0]   #distill or not

        imgs2 = torch.Tensor(batch_data2).cuda()
        imgs2.div_(255).sub_(0.5).div_(0.5)
        feat2 = self.model2(imgs2)[0]   #distill or not

        feat = torch.cat((feat1, feat2), -1)

        return feat.cpu().numpy()


# Divide a list into n parts as much as possible, limit len(list)==n,
# and allocate an empty list[] if the number of copies is greater than the number of elements in the original list
def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def read_template_media_list(path):
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias

def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label

def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats

def get_image_feature(img_path, files_list, model_path1, model_path2, epoch, gpu_id):
    batch_size = args.batch_size
    data_shape = (3, 112, 112)

    files = files_list
    print('files:', len(files))
    rare_size = len(files) % batch_size
    faceness_scores = []
    batch = 0
    img_feats = np.empty((len(files), 1024), dtype=np.float32)

    if upsample1 == 1:
        batch_data1 = np.empty((batch_size, 3, 112, 112))
    else:
        batch_data1 = np.empty((batch_size, 3, args.model_reso1, args.model_reso1))
    if upsample2 == 1:
        batch_data2 = np.empty((batch_size, 3, 112, 112))
    else:
        batch_data2 = np.empty((batch_size, 3, args.model_reso2, args.model_reso2))

    embedding = Embedding(model_path1, model_path2, data_shape, batch_size, reso1, reso2)

    for img_index, each_line in enumerate(files[:len(files) - rare_size]):  # batch inference
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob1, input_blob2 = embedding.get(img, lmk)

        batch_data1[img_index - batch * batch_size][:] = input_blob1
        batch_data2[img_index - batch * batch_size][:] = input_blob2
        if (img_index + 1) % batch_size == 0:
            img_feats[batch * batch_size:batch * batch_size +
                                         batch_size][:] = embedding.forward_db(batch_data1, batch_data2)
            if batch % 10 == 0:
                print('batch',batch)
            batch+=1
        faceness_scores.append(name_lmk_score[-1])

    if upsample1 == 1:
        batch_data1 = np.empty((rare_size, 3, 112, 112))
    else:    
        batch_data1 = np.empty((rare_size, 3, args.model_reso1, args.model_reso1))
    if upsample2 == 1:        
        batch_data2 = np.empty((rare_size, 3, 112, 112))
    else:
        batch_data2 = np.empty((rare_size, 3, args.model_reso2, args.model_reso2))
    embedding = Embedding(model_path1, model_path2, data_shape, rare_size, reso1, reso2)
    for img_index, each_line in enumerate(files[len(files) - rare_size:]):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob1, input_blob2 = embedding.get(img, lmk)
        batch_data1[img_index][:] = input_blob1
        batch_data2[img_index][:] = input_blob2
        if (img_index + 1) % rare_size == 0:
            print('batch', batch)
            img_feats[len(files) -
                      rare_size:][:] = embedding.forward_db(batch_data1, batch_data2)
            batch += 1
        faceness_scores.append(name_lmk_score[-1])
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    return img_feats, faceness_scores


def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)           # get all the ids
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]     # img_feats belonging to ind_t
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    return template_norm_feats, unique_templates

def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):             # assign each template a new id
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score

def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# # Step1: Load Meta Data

assert target == 'IJBC' or target == 'IJBB'
# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
templates, medias = read_template_media_list(
    os.path.join('%s/meta' % image_path,
                 '%s_face_tid_mid.txt' % target.lower()))       #templates: template id
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % image_path,
                 '%s_template_pair_label.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 2: Get Image Features

# =============================================================
# load image features
# format:
#           img_feats: [image_num x feats_dim] (227630, 512)
# =============================================================
start = timeit.default_timer()
img_path = '%s/loose_crop' % image_path
img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())
img_list = open(img_list_path)
files = img_list.readlines()
files_list = files
img_feats, faceness_scores = get_image_feature(img_path, files_list,
                                               model_path1, model_path2, 0, gpu_id)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                          img_feats.shape[1]))

# # Step3: Get Template Features

# =============================================================
# compute template features from image features.
# =============================================================
start = timeit.default_timer()
# ==========================================================
# Norm feature before aggregation into template feature?
# Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
# ==========================================================
# 1. FaceScore （Feature Norm）
# 2. FaceScore （Detector）

if use_flip_test:
    # concat --- F1
    # img_input_feats = img_feats
    # add --- F2
    img_input_feats = img_feats[:, 0:img_feats.shape[1] //
                                     2] + img_feats[:, img_feats.shape[1] // 2:]
else:
    img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

if use_norm_score:
    img_input_feats = img_input_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats ** 2, -1, keepdims=True))

if use_detector_score:
    print(img_input_feats.shape, faceness_scores.shape)
    img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
else:
    img_input_feats = img_input_feats

template_norm_feats, unique_templates = image2template_feature(
    img_input_feats, templates, medias)          # get features for different ids
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 4: Get Template Similarity Scores

# =============================================================
# compute verification scores between template pairs.
# =============================================================
start = timeit.default_timer()
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

save_path = os.path.join(result_dir, args.job)

if not os.path.exists(save_path):
    os.makedirs(save_path)

score_save_file = os.path.join(save_path, "%s.npy" % method_name)
np.save(score_save_file, score)

# # Step 5: Get ROC Curves and TPR@FPR Table

files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
fig.savefig(os.path.join(save_path, '%s.pdf' % method_name))
print(tpr_fpr_table)
print("-------------")
print(method_name)
print(job)
print(model_path1)
print(model_path2) 
print(reso1) 
print(reso2) 
print("-------------")
