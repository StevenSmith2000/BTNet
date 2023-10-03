"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datetime
import os
import pickle

import mxnet as mx
import numpy as np
import sklearn
import torch
from mxnet import ndarray as nd
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import random
import argparse
import sys
sys.path.append("..")
from backbones import get_model

class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

@torch.no_grad()
def load_bin_dynamic(path, image_size, load_type):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    reso_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
        resos = torch.empty((len(issame_list) * 2))
        reso_list.append(resos)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if load_type==0:  # random size
           reso = random.randint(4, 112)
        elif load_type==1:  # multi-resolution training (v1)
           candidates = [7,14,28,56,112]
           reso = random.choice(candidates)
        elif load_type==2: # multi-resolution training (v2)
           candidates = [7,14,28,112]
           reso = random.choice(candidates)
        img = mx.image.imresize(img, reso, reso)
        img = mx.image.copyMakeBorder(img, 0, 112-reso, 0, 112-reso)  #top, down, left, right
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
            reso_list[flip][idx] = reso
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, reso_list, issame_list

@torch.no_grad()
def load_bin_test(path, reso1, reso2, border=False, upsample1=0,upsample2=0):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
   
    data1_list = []
    data2_list = []

    for flip in [0, 1]:
        if upsample1 == 1:
            data = torch.empty((len(issame_list) , 3, 112, 112)).cuda()
        else:
            data = torch.empty((len(issame_list) , 3, reso1, reso1)).cuda()
        data1_list.append(data)

    for flip in [0, 1]:
        if upsample2 == 1:
            data = torch.empty((len(issame_list) , 3, 112, 112)).cuda()
        else:
            data = torch.empty((len(issame_list) , 3, reso2, reso2)).cuda()
        data2_list.append(data)

    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if idx % 2 == 0 :  #rescale id1 to reso1
            img = mx.image.imresize(img, reso1, reso1)
            if border == True:
                img = mx.image.copyMakeBorder(img, 0, 112-reso1, 0, 112-reso1)  #top, down, left, right
            if upsample1 == 1:
                img = mx.image.imresize(img, 112, 112)
        else:  #rescale id2 to reso2
            img = mx.image.imresize(img, reso2, reso2)
            if border == True:
                img = mx.image.copyMakeBorder(img, 0, 112-reso2, 0, 112-reso2)  #top, down, left, right
            if upsample2 == 1:
                img = mx.image.imresize(img, 112, 112)
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            _img = torch.from_numpy(img.asnumpy())
            _img = _img.cuda()
            if idx % 2 == 0:
                data1_list[flip][idx//2][:] = _img
            else:
                data2_list[flip][(idx-1)//2][:] = _img
        if idx % 2000 == 0:
            print('loading bin', idx)
        
    return data1_list, data2_list, issame_list

@torch.no_grad()
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 2000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list

@torch.no_grad()
def test_dynamic(data_set1, data_set2, issame_list, backbone1, backbone2, batch_size, nfolds=10):
    print('testing verification..')
    data_list1 = data_set1
    data_list2 = data_set2
    
    embeddings_list1 = []
    embeddings_list2 = []
    time_consumed = 0.0

    for i in range(len(data_list1)):
        data = data_list1[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            
            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5

            net_out = backbone1(img)[0]  # use distill or not
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list1.append(embeddings)

    for i in range(len(data_list2)):
        data = data_list2[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]

            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5

            net_out = backbone2(img)[0]  # use distill or not

            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list2.append(embeddings)

    _xnorm1 = 0.0
    _xnorm_cnt1 = 0
    for embed in embeddings_list1:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm1 += _norm
            _xnorm_cnt1 += 1
    _xnorm1 /= _xnorm_cnt1

    _xnorm2 = 0.0
    _xnorm_cnt2 = 0
    for embed in embeddings_list2:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm2 += _norm
            _xnorm_cnt2 += 1
    _xnorm2 /= _xnorm_cnt2

    embeddings1 = embeddings_list1[0].copy()
    embeddings1 = sklearn.preprocessing.normalize(embeddings1)
    embeddings1 = embeddings_list1[0] + embeddings_list1[1]

    embeddings2 = embeddings_list2[0].copy()
    embeddings2 = sklearn.preprocessing.normalize(embeddings2)
    embeddings2 = embeddings_list2[0] + embeddings_list2[1]

    acc1 = 0.0
    std1 = 0.0

    embeddings1 = sklearn.preprocessing.normalize(embeddings1)
    embeddings2 = sklearn.preprocessing.normalize(embeddings2)
    print(embeddings1.shape)
    print(embeddings2.shape)
    print(len(issame_list))
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings1, embeddings2, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm1, _xnorm2, embeddings_list1, embeddings_list2

@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):  #2 flip
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:  #2*issame_list
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba  #equals bacth_size except last batch
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5
            net_out: torch.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


def dumpR(data_set,
          backbone,
          batch_size,
          name='',
          data_extra=None,
          label_shape=None):
    print('dump verification embedding..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba

            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra),
                                     label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    actual_issame = np.asarray(issame_list)
    outname = os.path.join('temp.bin')
    with open(outname, 'wb') as f:
        pickle.dump((embeddings, issame_list),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do verification on face benchmarks')
    # general
    parser.add_argument('--data-dir', default='/dataset/ms1m-retinaface-t1', help='path to verification datasets')
    parser.add_argument('--model1', type=str, default='r50', help='backbone network 1')
    parser.add_argument('--model2', type=str, default='r50', help='backbone network 2')
    parser.add_argument('--weight1', type=str, default='../output/ms1mv3_r50_reso112/model.pt')
    parser.add_argument('--weight2', type=str, default='../output/ms1mv3_r50_reso14/model.pt')
    parser.add_argument('--reso1', type=int, default=112)
    parser.add_argument('--reso2', type=int, default=14)
    parser.add_argument('--model_reso1', type=int, default=112)
    parser.add_argument('--model_reso2', type=int, default=14)
    parser.add_argument('--target',default='lfw', choices=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30','calfw','cplfw'],help='test targets.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--max', default='', type=str, help='')
    parser.add_argument('--mode', default=0, type=int, help='')
    parser.add_argument('--nfolds', default=10, type=int, help='')
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    parser.add_argument("--upsample1",type=int, default=0, help='1: upsample the img to model_reso')
    parser.add_argument("--upsample2",type=int, default=0, help='1: upsample the img to model_reso')
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    print('test setting:')
    print('reso1:',args.reso1)
    print('reso2:', args.reso2)
    print('upsample1:', args.upsample1)
    print('upsample2:', args.upsample2)

    # Load models
    model1 = get_model(args.model1, args.model_reso1, fp16=False).cuda()
    model1.load_state_dict(torch.load(args.weight1))
    model1.eval()

    model2 = get_model(args.model2, args.model_reso2, fp16=False).cuda()
    model2.load_state_dict(torch.load(args.weight2))
    model2.eval()

    #Load datasets
    ver_list1 = []
    ver_list2 = []
    ver_name_list = []
    issame_list = []
    for name in args.target.split(','):
        path = os.path.join(args.data_dir, name + ".bin")
        if os.path.exists(path):
            print('loading.. ', name)
            data1_set, data2_set, issame = load_bin_test(path, args.reso1, args.reso2, border=False, upsample1=args.upsample1,upsample2=args.upsample2)
        else:
            print(f"The path {path} doesn't exist")
        ver_list1.append(data1_set)
        ver_list2.append(data2_set)
        ver_name_list.append(name+str(args.reso1)+'&'+str(args.reso2))
        issame_list.append(issame)

    #Test
    if args.mode == 0:
        assert len(ver_list1) == len(ver_list2)
        for i in range(len(ver_list1)):
            results = []
            acc1, std1, acc2, std2, xnorm1, xnorm2, embeddings_list1, embeddings_list2 = test_dynamic(
                ver_list1[i], ver_list2[i], issame_list[i], model1, model2, args.batch_size, args.nfolds)
            print('[%s]XNorm1: %f' % (ver_name_list[i], xnorm1))
            print('[%s]XNorm2: %f' % (ver_name_list[i], xnorm2))
            print('[%s]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], acc1, std1))
            print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], acc2, std2))
            results.append(acc2)
            print('Max of [%s] is %1.5f' % (ver_name_list[i], np.max(results)))
    elif args.mode == 1:
        raise ValueError
