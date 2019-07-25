import codecs
import numpy as np
import matplotlib.pyplot as plt 
from numpy import genfromtxt


COL_NUM = {
    'train_loss': 0,
    'train_acc': 1,
    'test_loss': 2,
    'test_acc': 3
}
TOTAL_NUM_EXP = 5
HIDDEN_DIM = [20, 50, 100, 200, 500, 1000, 2000]
COLORS = {
    20: "blue", 
    50: "green",
    100: "red", 
    200: "cyan", 
    500: "magenta", 
    1000: "yellow", 
    2000: "black"
}
LS = {
    0: '-',
    1: '-',
    2: '--',
    3: '--',
}
EPOCH = range(1, 201)


def get_mean_and_std():
    mean = dict()
    std = dict()
    res = dict()
    for hidden_dim in HIDDEN_DIM:
        res[hidden_dim] = []
        for num_exp in range(TOTAL_NUM_EXP):
            res[hidden_dim].append(genfromtxt(str(num_exp) + '_' + str(hidden_dim) + '.log', delimiter=','))
        res[hidden_dim] = np.stack(res[hidden_dim], axis=2) # res[hidden_dim]: (200, 4, 5)
        std[hidden_dim] = np.std(res[hidden_dim], axis=2)
        mean[hidden_dim] = np.mean(res[hidden_dim], axis=2)
    return mean, std


def draw(mean, std):
    plt.figure(figsize=(12,10))


    for hidden_dim in HIDDEN_DIM:
        plt.subplot(2, 1, 1)
        plt.title('Loss and Acc 20190725042633 @ GPU202\n\n')
        plt.xlim((0, 202))
        # plt.ylim((0, ))
        plt.errorbar(EPOCH, mean[hidden_dim][:, 0], std[hidden_dim][:, 0], color=COLORS[hidden_dim], ls=LS[0], label=str(hidden_dim)+'train')
        plt.errorbar(EPOCH, mean[hidden_dim][:, 2], std[hidden_dim][:, 2], color=COLORS[hidden_dim], ls=LS[2], label=str(hidden_dim)+'test')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right', ncol=1)

    for hidden_dim in HIDDEN_DIM:
        plt.subplot(2, 1, 2)
        plt.xlim((0, 202))
        # plt.ylim((0, ))
        plt.errorbar(EPOCH, mean[hidden_dim][:, 1], std[hidden_dim][:, 1], color=COLORS[hidden_dim], ls=LS[1], label=str(hidden_dim)+'train')
        plt.errorbar(EPOCH, mean[hidden_dim][:, 3], std[hidden_dim][:, 3], color=COLORS[hidden_dim], ls=LS[3], label=str(hidden_dim)+'test')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend(loc='upper right', ncol=1)

    plt.savefig('loss_and_acc.png', bbox_inches='tight')


def draw_epoch_acc(mean, std, epoch):
    train_acc_mean = [mean[hidden_dim][epoch-1][1] for hidden_dim in HIDDEN_DIM]
    train_acc_std = [std[hidden_dim][epoch-1][1] for hidden_dim in HIDDEN_DIM]
    test_acc_mean = [mean[hidden_dim][epoch-1][3] for hidden_dim in HIDDEN_DIM]
    test_acc_std = [std[hidden_dim][epoch-1][3] for hidden_dim in HIDDEN_DIM]

    if epoch == 100:
        plt.subplot(2, 1, 1)
        plt.title('epoch=100')
    else:
        plt.subplot(2, 1, 2)
        plt.title('\n\nepoch=200')

    dims = range(7)
    plt.xticks(dims, HIDDEN_DIM)
    plt.errorbar(dims, train_acc_mean, train_acc_std, color='blue', ls='-', label='train')
    plt.errorbar(dims, test_acc_mean, test_acc_std, color='blue', ls='--', label='test')
    plt.xlabel('hidden_dim')
    plt.ylabel('acc')
    plt.legend(loc='upper right', ncol=1)
    


if __name__ == '__main__':
    mean, std = get_mean_and_std()
    # draw(mean, std)
    plt.figure(figsize=(12,10))
    draw_epoch_acc(mean, std, 100)
    draw_epoch_acc(mean, std, 200)
    plt.savefig('epoch_acc.png', bbox_inches='tight')

    





