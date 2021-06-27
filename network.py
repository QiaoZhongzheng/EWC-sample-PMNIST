# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：EWC -> network
@IDE    ：PyCharm
@Author ：Qiao Zhongzheng
@Date   ：2021/6/23 20:28
@Desc   ：
=================================================='''
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D,LeakyReLU,MaxPool2D,Flatten,Input

def fcnn():
    input = Input(shape=784,dtype='float32',name='input')
    # x = Dense(128,activation='relu')(input)
    # x = Dense(64,activation='relu')(x)
    # x = Dense(32,activation='relu')(x)
    x = Dense(256,activation='relu')(input)
    x = Dense(256,activation='relu')(x)
    output = Dense(10,activation='softmax')(x)
    return Model(input, output)