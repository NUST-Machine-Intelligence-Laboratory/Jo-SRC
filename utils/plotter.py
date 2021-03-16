import numpy as np
import matplotlib.pyplot as plt


def plot_results(result_file):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    metrics = ['Train Loss', 'Train Accuracy', 'Test Accuracy']
    with open(result_file, 'r') as f:
        lines = f.readlines()
    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    best_epoch, best_acc = 0, 0.0
    for line in lines:
        if not line.startswith('epoch'):
            continue
        line = line.strip()
        epoch, train_loss, train_acc, test_acc = line.split(' | ')[:4]
        epoch_list.append(int(epoch.split(': ')[1]))
        train_loss_list.append(float(train_loss.split(': ')[1]))
        train_acc_list.append(float(train_acc.split(': ')[1]))
        test_acc_list.append(float(test_acc.split(': ')[1]))
        best_accuracy_epoch = line.split(' | ')[-1]
        best_acc = float(best_accuracy_epoch.split(' @ ')[0].split(': ')[1])
        best_epoch = int(best_accuracy_epoch.split(' @ ')[1].split(': ')[1])
    results = [
        np.array(train_loss_list),
        np.array(train_acc_list),
        np.array(test_acc_list)
    ]

    for i in range(len(metrics)):
        ax[i].plot(epoch_list, results[i], '-', label=metrics[i], linewidth=2)
        ax[i].set_title(metrics[i])
        ax[i].set_xlim(0, np.max(epoch_list))
        # ax[i].legend()
    ax[2].hlines(best_acc, 0, np.max(epoch_list), colors='red', linestyles='dashed')
    ax[2].plot(best_epoch, best_acc, 'ro')
    ax[2].annotate(f'({best_epoch}, {best_acc:.2f}%)', xy=(best_epoch, best_acc), xytext=(-30, -15), textcoords='offset points', color='red')

    result_dir = result_file.rsplit('/', 1)[0]
    fig.savefig(f'{result_dir}/results.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    logfile_path = '../Results/20200622_113934-cifar100-clean_openset0.2_baseline-resnet18-bestAcc_81.0125/log.txt'
    plot_results(logfile_path)
    # plt.show()
