# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：EWC -> pmnist
@IDE    ：PyCharm
@Author ：Qiao Zhongzheng
@Date   ：2021/6/23 20:14
@Desc   ：
=================================================='''
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np


(x_train_origin, y_train_origin), (x_test_origin, y_test_origin) = mnist.load_data()
x_train_origin = x_train_origin.astype('float32')
x_test_origin = x_test_origin.astype('float32')
x_train_origin /= 255
x_test_origin /= 255

x_train_origin = x_train_origin.reshape((-1, 784))  # shape (50000, 784)
x_test_origin = x_test_origin.reshape((-1, 784))

y_train_origin = to_categorical(y_train_origin, 10)
y_test_origin = to_categorical(y_test_origin, 10)



def PmnistGenerate(ntask=5):
    '''
    Input:
        ntask: number of tasks

    Return:

    '''

    PMNIST = []
    for i in range(ntask):
        index = np.arange(x_train_origin.shape[1])
        np.random.seed(seed=i)
        np.random.shuffle(index)
        x_train = x_train_origin[:, index]
        x_test = x_test_origin[:, index]
        y_train = y_train_origin
        y_test = y_test_origin

        PMNIST.append([x_train,y_train,x_test,y_test])


    return PMNIST


