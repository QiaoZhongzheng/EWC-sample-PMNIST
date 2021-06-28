# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：EWC -> main
@IDE    ：PyCharm
@Author ：Qiao Zhongzheng
@Date   ：2021/6/23 20:33
@Desc   ：
=================================================='''
from pmnist import PmnistGenerate
from network import fcnn
from EWC_func import compute_FIM, EWC_loss
import tensorflow as tf
import numpy as np
import os
import copy
import matplotlib.pyplot as plt

if not os.path.exists('./saved'):
    os.mkdir('./saved')

path_model = './saved/models'
path_FIM = './saved/Fisher'
path_Acc = './saved/Acc'
path_Images = './saved/images'

if not os.path.exists(path_model):
    os.mkdir(path_model)

if not os.path.exists(path_FIM):
    os.mkdir(path_FIM)

if not os.path.exists(path_Acc):
    os.mkdir(path_Acc)

if not os.path.exists(path_Images):
    os.mkdir(path_Images)

optimizer = tf.optimizers.Adam(learning_rate=0.001)


def SFT(ntask=10, Batchsize=32, Epoch=5):
    PMNIST = PmnistGenerate(ntask)
    Acc = np.zeros((ntask,ntask))
    model = fcnn()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics='accuracy')

    if not os.path.exists(path_model):
        os.mkdir(path_model)

    if not os.path.exists(path_Acc):
        os.mkdir(path_Acc)

    for i in range(ntask):
        current_task = PMNIST[i]
        x_train, y_train, x_test, y_test = current_task[0], current_task[1], current_task[2], current_task[3]

        checkpoint_filepath = path_model+f'/SFT_task{i+1}.h5'

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )

        model.fit(x_train, y_train, batch_size=Batchsize, epochs=Epoch, shuffle=True,
                  validation_split=0.2, callbacks=model_checkpoint_callback
                  )

        model.load_weights(checkpoint_filepath)

        for j in range(i+1):
            learned_task = PMNIST[j]
            x_test, y_test = learned_task[2], learned_task[3]
            loss,accuracy = model.evaluate(x_test,y_test)
            Acc[i][j] = accuracy

    save_Acc = path_Acc+'/SFT'
    np.save(save_Acc, Acc, allow_pickle=True)

    return Acc


def EWC(ntask=10, Batchsize=32, Epoch=10, Lambda=10, num_samples=300):
    PMNIST = PmnistGenerate(ntask)
    Acc = np.zeros((ntask, ntask))
    model = fcnn()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy')

    if not os.path.exists(path_model):
        os.mkdir(path_model)

    if not os.path.exists(path_FIM):
        os.mkdir(path_FIM)

    if not os.path.exists(path_Acc):
        os.mkdir(path_Acc)

    for i in range(ntask):
        current_task = PMNIST[i]
        x_train, y_train = current_task[0], current_task[1]

        checkpoint_filepath = path_model+f'/EWC_task{i+1}.h5'

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='accuracy',
            mode='max',
            save_best_only=True
        )

        model.fit(x_train, y_train, batch_size=Batchsize, epochs=Epoch, shuffle=True,
                  validation_split=0.2, callbacks=model_checkpoint_callback
                  )

        model.load_weights(checkpoint_filepath)

        FIM = compute_FIM(model, x_train, num_samples)
        save_Fisher_dir = path_FIM + f'./Task{i}.npy'
        np.save(save_Fisher_dir, FIM, allow_pickle=True)
        prior_weights = copy.deepcopy(model.trainable_weights)

        model.compile(optimizer=optimizer, loss=EWC_loss(model, FIM, prior_weights, Lambda=Lambda), metrics='accuracy')


        for j in range(i + 1):
            learned_task = PMNIST[j]
            x_test, y_test = learned_task[2], learned_task[3]
            loss, accuracy = model.evaluate(x_test, y_test)
            Acc[i][j] = accuracy

    save_Acc = path_Acc+f'/EWC_lambda={Lambda}'
    np.save(save_Acc, Acc, allow_pickle=True)

    return Acc


def acc_curves(Acc, label):
    ntask = Acc.shape[0]
    plt.figure()

    for i in range(ntask):
        x = list(range(i+1, ntask+1))
        y = Acc[i:, i]
        #exec('plt.plot(x, y, color = "C{}",linestyle = "dotted",marker = "o",label = "Task{}" )'.format(i, i+1))
        plt.plot(x, y, color=f'C{i}', marker="o", label=f'Task{i+1}')

    # plt.figure(1)
    plt.xlabel('Tasks')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.axis([1, ntask+1, 0, 1])
    plt.xticks(np.arange(1, ntask+1, 1))

    plt.title(f'Accuracy curves on learned tasks for {label}')

    plt.savefig(path_Images+'/'+label)
    plt.show()


def avg_acc_curves(Acc_list, label_list):

    ntask = Acc_list[0].shape[0]

    x = list(range(1, ntask+1))
    plt.figure()

    for i in range(len(Acc_list)):
        Acc_i = Acc_list[i]
        avg_acc_tasks = []
        for t in range(ntask):
            avg_acc_t = np.mean(Acc_i[t][:t + 1])
            avg_acc_tasks.append(avg_acc_t)
        plt.plot(x, avg_acc_tasks, marker='o', label=label_list[i])

    plt.xlabel('Tasks')
    plt.ylabel('Avg Acc')
    plt.legend()
    plt.axis([1,ntask+1,0,1])
    plt.xticks(np.arange(1,ntask+1,1))

    plt.title(f'Average Accuracy on {ntask} tasks for different settings')

    plt.savefig(path_Images+'/avg_acc_curves')
    plt.show()


Acc_SFT = SFT()
acc_curves(Acc_SFT,'SFT')

Acc_EWC_1 = EWC(Lambda=1)
acc_curves(Acc_EWC_1,'EWC_1')

Acc_EWC_10 = EWC(Lambda=10)
acc_curves(Acc_EWC_10,'EWC_10')

Acc_EWC_100 = EWC(Lambda=100)
acc_curves(Acc_EWC_100,'EWC_100')

Acc_list = [Acc_SFT, Acc_EWC_1, Acc_EWC_10, Acc_EWC_100]
label_list = ['SFT', 'EWC,lambda=1', 'EWC,lambda=10', 'EWC,lambda=100']


# acc_curves(Acc_SFT,'SFT')
# acc_curves(Acc_EWC_1,'EWC_1')
# acc_curves(Acc_EWC_10,'EWC_10')
# acc_curves(Acc_EWC_100,'EWC_100')

avg_acc_curves(Acc_list, label_list)
