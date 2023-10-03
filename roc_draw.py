# Draw ROC curves
# 1: run eval_ijbc_11_multi_reso.py to get numpy files (.npy)
# 2: run roc_draw.py to draw roc curves in a common figure for method comparison
import numpy as np
from pathlib import Path
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
from prettytable import PrettyTable


def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label

if __name__ =="__main__":
    path1 = 'IJBC_11_result/7&7/HR.npy'
    path2 = 'IJBC_11_result/7&7/MM.npy'
    path3 = 'IJBC_11_result/7&7/MR.npy'
    path4 = 'IJBC_11_result/7&7/Ours.npy'
    image_path = '/dataset/IJBC/'
    save_path = 'IJBC_11_result/fig'
    target='IJBC'
    job='7&7'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    score_save_file = path1,path2,path3,path4

    print("Start")

    p1, p2, label = read_template_pair_list(
        os.path.join('%s/meta' % image_path,
                     '%s_template_pair_label.txt' % target.lower()))
    
    files = [path for path in score_save_file]
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
        if method == 'HR':
            name = r'$\varphi_{hr}$'
        if method == 'MM':
            name = r'$\varphi_{mm}$'
        if method == 'MR':
            name = r'$\varphi_{mr}$'
        if method == 'Ours':
            name = r'$\varphi_{bt}$ (Ours)'
        fpr, tpr, _ = roc_curve(label, scores[method])
        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)  # select largest tpr at same fpr
        plt.plot(fpr,
                 tpr,
                 color=colours[method],
                 lw=1,
                 label=('[%s (AUC = %0.2f %%)]' %
                        (name, roc_auc*100)))
        tpr_fpr_row = []
        tpr_fpr_row.append("%s-%s" % (method, target))
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
            tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
        tpr_fpr_table.add_row(tpr_fpr_row)
    plt.xlim([10 ** -6, 0.1])
    lower = 0.02
    plt.ylim([lower, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(x_labels)
    plt.yticks(np.linspace(lower, 1.0, 8, endpoint=True))
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC on IJB')
    plt.legend(loc="best")
    fig.savefig(os.path.join(save_path,'%s.png' % job.lower()),dpi=500)
    print(tpr_fpr_table)
