# Evaluation on IJB-C 1:N face identification
# coding: utf-8

import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import timeit
import argparse
import cv2
import numpy as np
import torch
from skimage import transform as trans
from backbones import get_model
import heapq
import math
import sys
import warnings

sys.path.insert(0, "../")
warnings.filterwarnings("ignore")

class Embedding(object):
    def __init__(self, args, prefix1, prefix2, data_shape, batch_size=1, reso1=(112,112), reso2=(112,112)):
        image_size = (112, 112)
        self.image_size = image_size
        self.reso1 = reso1
        self.reso2 = reso2
        self.upsample1 = args.upsample1
        self.upsample2 = args.upsample2
        self.downsample1 = args.downsample1
        self.downsample2 = args.downsample2

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
        img_flip = img
        if self.reso1[1] != self.image_size[1] or self.reso1[0] != self.image_size[0]:
            img = cv2.resize(img, (self.reso1[1], self.reso1[0]))
            if self.upsample1 == 1:
                img = cv2.resize(img, (112, 112))
        if self.downsample1 == 1:
            img = cv2.resize(img, (args.model_reso1, args.model_reso1))
        if self.reso2[1] != self.image_size[1] or self.reso2[0] != self.image_size[0]:
            img_flip = cv2.resize(img, (self.reso2[1], self.reso2[0]))
            if self.upsample2 == 1:
                img_flip = cv2.resize(img_flip, (112, 112))
        if self.downsample2 == 1:
            img_flip = cv2.resize(img_flip, (args.model_reso2, args.model_reso2))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = cv2.cvtColor(img_flip, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img_flip)
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

def read_template_subject_id_list(path):
    ijb_meta = np.loadtxt(path, dtype=str, skiprows=1, delimiter=',')
    templates = ijb_meta[:, 0].astype(np.int)
    subject_ids = ijb_meta[:, 1].astype(np.int)
    return templates, subject_ids

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

def get_image_feature(args, img_path, files_list, model_path1, model_path2):
    batch_size = args.batch_size
    data_shape = (3, 112, 112)

    files = files_list
    print('files:', len(files))
    rare_size = len(files) % batch_size
    faceness_scores = []
    batch = 0
    img_feats = np.empty((len(files), 1024), dtype=np.float32)

    if args.upsample1 == 1:
        batch_data1 = np.empty((batch_size, 3, 112, 112))
    else:
        batch_data1 = np.empty((batch_size, 3, args.model_reso1, args.model_reso1))
    if args.upsample2 == 1:
        batch_data2 = np.empty((batch_size, 3, 112, 112))
    else:
        batch_data2 = np.empty((batch_size, 3, args.model_reso2, args.model_reso2))

    embedding = Embedding(args, model_path1, model_path2, data_shape, batch_size, reso1, reso2)

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
                print('batch', batch)
            batch += 1
        faceness_scores.append(name_lmk_score[-1])
    if args.upsample1 == 1:
        batch_data1 = np.empty((rare_size, 3, 112, 112))
    else:
        batch_data1 = np.empty((rare_size, 3, reso1[1], reso1[0]))
    if args.upsample2 == 1:
        batch_data2 = np.empty((rare_size, 3, 112, 112))
    else:    # residual batch for inference
        batch_data2 = np.empty((rare_size, 3, reso2[1], reso2[0]))
    embedding = Embedding(args, model_path1, model_path2, data_shape, rare_size, reso1, reso2)
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


def image2template_feature(img_feats=None,
                           templates=None,
                           medias=None,
                           choose_templates=None,
                           choose_ids=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates, indices = np.unique(choose_templates, return_index=True)
    unique_subjectids = choose_ids[indices]
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t, ) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m, ) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], 0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    template_norm_feats = template_feats / np.sqrt(
        np.sum(template_feats**2, -1, keepdims=True))
    return template_norm_feats, unique_templates, unique_subjectids

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

def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


def evaluation(query_feats, gallery_feats, mask):
    Fars = [0.01, 0.1]
    print("shape of query_feat:", query_feats.shape)
    print("shape of gallery_feat:", gallery_feats.shape)

    query_num = query_feats.shape[0]
    gallery_num = gallery_feats.shape[0]

    similarity = np.dot(query_feats, gallery_feats.T)
    top_inds = np.argsort(-similarity)


    # calculate top1
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0]
        if j == mask[i]:
            correct_num += 1
    print("top1 = {}".format(correct_num / query_num))
    # calculate top5
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:5]
        if mask[i] in j:
            correct_num += 1
    print("top5 = {}".format(correct_num / query_num))
    # calculate 10
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:10]
        if mask[i] in j:
            correct_num += 1
    print("top10 = {}".format(correct_num / query_num))

    neg_pair_num = query_num * gallery_num - query_num
    required_topk = [math.ceil(query_num * x) for x in Fars]
    top_sims = similarity
    # calculate fars and tprs
    pos_sims = []
    for i in range(query_num):
        gt = mask[i]
        pos_sims.append(top_sims[i, gt])
        top_sims[i, gt] = -2.0

    pos_sims = np.array(pos_sims)
    neg_sims = top_sims[np.where(top_sims > -2.0)]
    neg_sims = heapq.nlargest(max(required_topk), neg_sims)  # heap sort
    for far, pos in zip(Fars, required_topk):
        th = neg_sims[pos - 1]
        recall = np.sum(pos_sims > th) / query_num
        print("far = {:.10f} pr = {:.10f} th = {:.10f}".format(
            far, recall, th))


def gen_mask(query_ids, reg_ids):
    mask = []
    for query_id in query_ids:
        pos = [i for i, x in enumerate(reg_ids) if query_id == x]
        if len(pos) != 1:
            raise RuntimeError(
                "RegIdsError with id = {}， duplicate = {} ".format(
                    query_id, len(pos)))
        mask.append(pos[0])
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='do ijb 1n test')
    parser.add_argument('--model-prefix1', default='output/ms1mv3_r50_reso112/model.pt', help='path to load model1.')
    parser.add_argument('--model-prefix2', default='output/ms1mv3_r50_reso14/model.pt', help='path to load model2.')
    parser.add_argument('--image-path', default='/dataset/IJBC/', type=str, help='path to IJB-C')
    parser.add_argument('--result-dir', default='IJBC_1n_result', type=str, help='path to save the results')
    parser.add_argument('--batch-size', default=128, type=int, help='')
    parser.add_argument('--network1', default='r50', type=str, help='')
    parser.add_argument('--network2', default='r50', type=str, help='')
    parser.add_argument('--model_reso1', default=112, type=int, help='')
    parser.add_argument('--model_reso2', default=14, type=int, help='')
    parser.add_argument('--reso1', default=112, type=int, help='')
    parser.add_argument('--reso2', default=14, type=int, help='')
    parser.add_argument('--job', default='', type=str, help='job name')
    parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
    parser.add_argument('--upsample1', default=1, type=int, help='')
    parser.add_argument('--upsample2', default=1, type=int, help='')
    parser.add_argument('--downsample1', default=0, type=int, help='')
    parser.add_argument('--downsample2', default=0, type=int, help='')
    parser.add_argument('--method_name', default='', type=str, help='')
    args = parser.parse_args()
    
    job = args.job
    target = args.target
    method_name = args.method_name
    image_path = args.image_path
    model_path1 = args.model_prefix1
    model_path2 = args.model_prefix2
    reso1 = (args.reso1,args.reso1)
    reso2 = (args.reso2,args.reso2)

    meta_dir = "%s/meta" % image_path  #meta root dir
    if target == 'IJBC':
        gallery_s1_record = "%s_1N_gallery_G1.csv" % (args.target.lower())
        gallery_s2_record = "%s_1N_gallery_G2.csv" % (args.target.lower())
    else:
        gallery_s1_record = "%s_1N_gallery_S1.csv" % (args.target.lower())
        gallery_s2_record = "%s_1N_gallery_S2.csv" % (args.target.lower())
    gallery_s1_templates, gallery_s1_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s1_record))
    print(gallery_s1_templates.shape, gallery_s1_subject_ids.shape)

    gallery_s2_templates, gallery_s2_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s2_record))
    print(gallery_s2_templates.shape, gallery_s2_templates.shape)

    gallery_templates = np.concatenate(
        [gallery_s1_templates, gallery_s2_templates])
    gallery_subject_ids = np.concatenate(
        [gallery_s1_subject_ids, gallery_s2_subject_ids])
    print(gallery_templates.shape, gallery_subject_ids.shape)

    media_record = "%s_face_tid_mid.txt" % args.target.lower()
    total_templates, total_medias = read_template_media_list(
        os.path.join(meta_dir, media_record))
    print("total_templates", total_templates.shape, total_medias.shape)

    start = timeit.default_timer()

    img_path = '%s/loose_crop' % image_path
    img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())
    img_list = open(img_list_path)
    files = img_list.readlines()
    files_list = files
    img_feats, faceness_scores = get_image_feature(args, img_path,files_list,
                                                   model_path1, model_path2)
    print('img_feats', img_feats.shape)
    print('faceness_scores', faceness_scores.shape)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                              img_feats.shape[1]))

    # compute template features from image features.
    start = timeit.default_timer()
    # ==========================================================
    # Norm feature before aggregation into template feature?
    # Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
    # ==========================================================
    use_norm_score = False  # if True, TestMode(N1)
    use_detector_score = True  # if True, TestMode(D1)
    use_flip_test = True  # if True, TestMode(F1)

    if use_flip_test:
        # concat --- F1
        #img_input_feats = img_feats
        # add --- F2
        img_input_feats = img_feats[:, 0:int(
            img_feats.shape[1] / 2)] + img_feats[:,
                                                 int(img_feats.shape[1] / 2):]
    else:
        img_input_feats = img_feats[:, 0:int(img_feats.shape[1] / 2)]

    if use_norm_score:
        img_input_feats = img_input_feats
    else:
        # normalise features to remove norm information
        img_input_feats = img_input_feats / np.sqrt(
            np.sum(img_input_feats**2, -1, keepdims=True))

    if use_detector_score:
        print(img_input_feats.shape, faceness_scores.shape)
        img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
    else:
        img_input_feats = img_input_feats

    print("input features shape", img_input_feats.shape)

    #load gallery feature
    gallery_templates_feature, gallery_unique_templates, gallery_unique_subject_ids = image2template_feature(
        img_input_feats, total_templates, total_medias, gallery_templates,
        gallery_subject_ids)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print("gallery_templates_feature", gallery_templates_feature.shape)
    print("gallery_unique_subject_ids", gallery_unique_subject_ids.shape)

    #load prope feature
    probe_mixed_record = "%s_1N_probe_mixed.csv" % target.lower()
    probe_mixed_templates, probe_mixed_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, probe_mixed_record))
    print(probe_mixed_templates.shape, probe_mixed_subject_ids.shape)
    probe_mixed_templates_feature, probe_mixed_unique_templates, probe_mixed_unique_subject_ids = image2template_feature(
        img_input_feats, total_templates, total_medias, probe_mixed_templates,
        probe_mixed_subject_ids)
    print("probe_mixed_templates_feature", probe_mixed_templates_feature.shape)
    print("probe_mixed_unique_subject_ids",
          probe_mixed_unique_subject_ids.shape)

    gallery_ids = gallery_unique_subject_ids
    gallery_feats = gallery_templates_feature
    probe_ids = probe_mixed_unique_subject_ids
    probe_feats = probe_mixed_templates_feature

    mask = gen_mask(probe_ids, gallery_ids)

    start = timeit.default_timer()
    evaluation(probe_feats, gallery_feats, mask)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    print("-------------")
    print(method_name)
    print(job)
    print(model_path1)
    print(model_path2)
    print(reso1)
    print(reso2)
    print("-------------")
