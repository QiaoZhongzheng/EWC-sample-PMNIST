# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：EWC -> util
@IDE    ：PyCharm
@Author ：Qiao Zhongzheng
@Date   ：2020/12/17 19:56
@Desc   ：
=================================================='''
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def compute_FIM(model, imgset, num_sample=50):

    fim_accum = []
    for i in range(len(model.trainable_weights)):
        fim_accum.append(np.zeros(K.int_shape(model.trainable_weights[i])))
    fim_accum = np.array(fim_accum, dtype=object)

    img_index_list = []
    for j in range(num_sample):
        img_index = np.random.randint(imgset.shape[0])
        img_index_list.append(img_index)

    image_batch = imgset[img_index_list]
    epsilon = tf.constant(1e-6)

    with tf.GradientTape(persistent=True) as tape:
        output = K.log(model(image_batch) + epsilon)
        tape.watch(output)

    for m in range(len(model.trainable_weights)):
        gradients = tape.gradient(output, model.trainable_weights[m])
        fim_accum[m] += np.square(gradients)
    del tape
    fim_accum /= num_sample

    return fim_accum


def EWC_penalty(fisher, current_weights, prior_weights, Lambda=1):
    regularization = 0.

    for i in range(len(current_weights)):
        regularization += Lambda * K.sum(fisher[i] * K.square(current_weights[i] - prior_weights[i]))
    return regularization


class EWC_loss(keras.losses.Loss):
    def __init__(self,model,fisher,prior_weights,Lambda=1 ,name = "ewc_loss"):
        super().__init__(name=name)
        self.model = model
        self.fisher = fisher
        self.prior_weights = prior_weights
        self.Lambda = Lambda

    def call(self, y_true, y_pred):
        loss_new = tf.keras.losses.mae(y_true=y_true, y_pred=y_pred)
        loss_old = EWC_penalty(self.fisher,self.model.trainable_weights, self.prior_weights, self.Lambda)

        return loss_new + loss_old
