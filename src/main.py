from clustering import KMeans
from decompose import PCA
from utils import load_data
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

sns.set()


def load_args():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--pca', default=2, type=int,
                        help='set to 1 if we desire running pca, otherwise 0')
    parser.add_argument('--kmeans', default=1, type=int,
                        help='set to 1 if we desire running kmeans, otherwise 0')

    parser.add_argument('--pca_retain_ratio', default=.9, type=float)
    parser.add_argument('--kmeans_max_k', default=15, type=int)
    parser.add_argument('--kmeans_max_iter', default=20, type=int)
    parser.add_argument('--root_dir', default='../data/', type=str)
    args = parser.parse_args()

    return args


def plot_y_vs_x_list(y_vs_x, x_label, y_label, save_path):
    fld = os.path.join(args.root_dir, save_path)
    if not os.path.exists((fld)):
        os.mkdir(fld)

    plots_per_fig = 2

    ks_sses_keys = list(range(0, len(y_vs_x)))
    js = list(range(0, len(ks_sses_keys), plots_per_fig))

    for j in js:
        pp = ks_sses_keys[j:j + plots_per_fig]
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(len(pp), 1, figure=fig)
        i = 0
        for k in pp:
            ax = fig.add_subplot(gs[i, :])
            ax.set_ylabel('%s (k=%d)' % (y_label, k))
            ax.set_xlabel(x_label)
            ax.plot(range(1, len(y_vs_x[k]) + 1),
                    [x for x in y_vs_x[k]], linewidth=2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            i += 1

        fig.savefig(os.path.join(fld, '%d_%d.png' % (pp[0], pp[-1])))

    print('Saved at : %s' % fld)


def plot_y_vs_x(ys_vs_x, x_label, y_label, save_path):
    fld = os.path.join(args.root_dir, save_path)
    if not os.path.exists((fld)):
        os.mkdir(fld)

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, :])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.plot(range(1, len(ys_vs_x) + 1), ys_vs_x, linewidth=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(os.path.join(fld, 'plot.png'))

    print('Saved at : %s' % fld)


def visualize(x_train, y_train):
    x_train=x_train[:,0:2]

    plot_dict={}
    for i in range(len(y_train)):
        classID=y_train[i]
        val=x_train[i]
        if(classID in plot_dict):
            plot_dict[classID].append(val)
        else:
            plot_dict[classID]=[val]
    
    color=['red','green','blue','yellow','pink','orange']
    index=0
    for key,val in plot_dict.items():
        plt.scatter(*zip(*val),color=color[index])
        index+=1
    plt.savefig('../data/visualize.png')
    ##################################
    #      YOUR CODE GOES HERE       #
    ##################################


def apply_kmeans(do_pca, x_train, y_train, kmeans_max_iter, kmeans_max_k):
    print('kmeans\n')
    train_sses_vs_iter = []
    train_sses_vs_k = []
    train_purities_vs_k = []

    ##################################
    #      YOUR CODE GOES HERE       #
    ##################################

    for k in range(1, kmeans_max_k):
        kmeans = KMeans(k, kmeans_max_iter)
        sse_vs_iter = kmeans.fit(x_train)
        train_sses_vs_iter.append(sse_vs_iter)
        train_purities_vs_k.append(kmeans.get_purity(x_train, y_train))
        train_sses_vs_k.append(min(sse_vs_iter))

    plot_y_vs_x_list(train_sses_vs_iter, x_label='iter', y_label='sse',
                     save_path='plot_sse_vs_k_subplots_%d' % do_pca)
    plot_y_vs_x(train_sses_vs_k, x_label='k', y_label='sse',
                save_path='plot_sse_vs_k_%d' % do_pca)
    plot_y_vs_x(train_purities_vs_k, x_label='k', y_label='purities',
                save_path='plot_purity_vs_k_%d' % do_pca)


def apply_kmeans1(do_pca, x_train, y_train, kmeans_max_iter, kmeans_max_k):
    print('kmeans\n')
    train_sses_vs_iter = []
    train_sses_vs_k = []
    train_purities_vs_k = []

    ##################################
    #      YOUR CODE GOES HERE       #
    ##################################

    for run in range(0, 5):
        kmeans = KMeans(6, kmeans_max_iter)
        sse_vs_iter = kmeans.fit(x_train)
        train_sses_vs_iter.append(sse_vs_iter)
        train_purities_vs_k.append(kmeans.get_purity(x_train, y_train))
        train_sses_vs_k.append(min(sse_vs_iter))

    result = []
    for col in range(len(train_sses_vs_iter[0])):
        sum = 0
        for row in range(0, 5):
            sum += train_sses_vs_iter[row][col]
        sum = sum/5
        result.append(sum)
    result = [result]

    print(result)

    plot_y_vs_x_list(result, x_label='iter', y_label='sse',
                     save_path='plot_sse_vs_k_subplots_%d' % do_pca)


def apply_kmeans2(do_pca, x_train, y_train, kmeans_max_iter, kmeans_max_k):
    print('kmeans\n')
    train_sses_vs_iter = []
    train_sses_vs_k = []
    train_purities_vs_k = []

    ##################################
    #      YOUR CODE GOES HERE       #
    ##################################

    result = []
    for k in range(1, 11):
        print('k:', k)
        for times in range(0, 5):
            kmeans = KMeans(k, kmeans_max_iter)
            sse_vs_iter = kmeans.fit(x_train)
            train_sses_vs_iter.append(sse_vs_iter)
            train_purities_vs_k.append(kmeans.get_purity(x_train, y_train))
            train_sses_vs_k.append(min(sse_vs_iter))
        print(train_sses_vs_k)
        avg = sum(train_sses_vs_k)/len(train_sses_vs_k)
        result.append(avg)
        train_sses_vs_k = []

    print(len(result))
    print(result)
    plot_y_vs_x(result, x_label='k', y_label='sse',
                save_path='plot_sse_vs_k_%d' % do_pca)



def apply_kmeans3(do_pca, x_train, y_train, kmeans_max_iter, kmeans_max_k):
    print('kmeans\n')
    train_sses_vs_iter = []
    train_sses_vs_k = []
    train_purities_vs_k = []

    ##################################
    #      YOUR CODE GOES HERE       #
    ##################################

    result = []
    for k in range(1, 11):
        print('k:', k)
        for times in range(0, 5):
            kmeans = KMeans(k, kmeans_max_iter)
            sse_vs_iter = kmeans.fit(x_train)
            train_sses_vs_iter.append(sse_vs_iter)
            train_purities_vs_k.append(kmeans.get_purity(x_train, y_train))
            train_sses_vs_k.append(min(sse_vs_iter))
        print(train_purities_vs_k)
        avg = sum(train_purities_vs_k)/len(train_purities_vs_k)
        result.append(avg)
        train_purities_vs_k = []

    print(result)
    print('max purity',max(result))
    plot_y_vs_x(result, x_label='k', y_label='purities',
                save_path='plot_purity_vs_k_%d' % do_pca)


if __name__ == '__main__':
    args = load_args()
    x_train, y_train = load_data(args.root_dir)
    if args.kmeans == 1:
        # apply_kmeans(args.pca, x_train, y_train,args.kmeans_max_iter, args.kmeans_max_k)
        print('Running k-means test')
        apply_kmeans1(args.kmeans, x_train, y_train,args.kmeans_max_iter, args.kmeans_max_k)
        apply_kmeans2(args.kmeans, x_train, y_train,
                      args.kmeans_max_iter, args.kmeans_max_k)
        apply_kmeans3(args.kmeans, x_train, y_train,
                      args.kmeans_max_iter, args.kmeans_max_k)

    if args.pca == 2:
        print('Running PCA test')
        
        testR=[0.9,0.92,0.94,0.96]
        
        for r in testR:
            print('r: ',r)
            pca = PCA(r)
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            visualize(x_train, y_train)
            D=pca.findD()
            x_train=x_train=x_train[:,0:D]
            apply_kmeans3(args.pca, x_train, y_train,
                        args.kmeans_max_iter, args.kmeans_max_k)
            
    print('Done')
