from __future__ import print_function
# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import random as rn
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve

### We modified Pahikkala et al. (2014) source code for cross-val process ###

import os

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(1)
rn.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import keras
from keras import backend as K

tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from datahelper import *
# import logging
from itertools import product
from arguments import argparser, logging

import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers

import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from emetrics import get_aupr, get_cindex, get_rm2

import pandas as pd

TABSY = "\t"
figdir = "figures/"


def build_combined_categorical(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')  ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size + 1, output_dim=128, input_length=FLAGS.max_smi_len)(
        XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    encode_protein = Embedding(input_dim=FLAGS.charseqset_size + 1, output_dim=128, input_length=FLAGS.max_seq_len)(
        XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein],
                                                  axis=-1)  # merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    # And add a logistic regression on top
    predictions = Dense(49)(FC2)
    #predictions = Dense(2, kernel_initializer='normal')(FC2)  # OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'
    predictions = Activation('softmax')(predictions)#加的

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=predictions)

    interactionModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # , metrics=['cindex_score']
    print(interactionModel.summary())
    #plot_model(interactionModel, to_file='figures/build_combined_categorical.png')

    return interactionModel

    # train_in = Dense(256, activation='relu')(train_in)
    # train_in = BatchNormalization()(train_in)
    # train_in = Dropout(droprate)(train_in)
    # train_in = Dense(event_num)(train_in)
    # out = Activation('sigmoid')(train_in)
    # model = Model(input=train_input, output=out)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds, measure, runmethod, FLAGS, dataset):
    bestparamlist = []
    # trainset是个二维数组
    test_set, outer_train_sets = dataset.read_sets(FLAGS)
    test_set2 = pd.read_table('test_fold.txt', header=None)
    test_set3 = test_set2.values.tolist()
    outer_train_sets2 = pd.read_table('train_folds.txt', header=None)
    outer_train_sets3 = outer_train_sets2.values.tolist()

    foldinds = len(outer_train_sets)

    test_sets = []
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []

    # logger.info('Start training')
    # 产生5组train，val，test
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))

    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(runmethod,FLAGS,train_sets,val_sets)

    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(measure, runmethod, FLAGS,train_sets, test_sets)



    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" %
            (bestparamind, bestparam), FLAGS)

    testperfs = []
    testloss = []

    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind]
        foldloss = all_losses[bestparamind][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logging("Test Performance CI", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Performance MSE", FLAGS)
    logging(testloss, FLAGS)

    return avgperf, avgloss, teststd


def general_nfold_cv(runmethod, FLAGS, labeled_sets, val_sets):  ## BURAYA DA FLAGS LAZIM????

    paramset1 = FLAGS.num_windows  # [32]#[32,  512] #[32, 128]  # filter numbers
    paramset2 = FLAGS.smi_window_lengths  # [4, 8]#[4,  32] #[4,  8] #filter length smi
    paramset3 = FLAGS.seq_window_lengths  # [8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    epoch = FLAGS.num_epoch  # 100
    batchsz = FLAGS.batch_size  # 256

    logging("---Parameter Search-----", FLAGS)

    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [[0 for x in range(w)] for y in range(h)]
    all_losses = [[0 for x in range(w)] for y in range(h)]
    print(all_predictions)

    for foldind in range(0,1):
        train_drugs4 = pd.read_table('train_drugs4.txt', header=None)
        train_prots4 = pd.read_table('train_prots4.txt', header=None)
        test_drugs4 = pd.read_table('test_drugs4.txt', header=None)
        test_prots4 = pd.read_table('test_prots4.txt', header=None)
        train_Y4 = pd.read_table('train_Y4.txt', header=None)
        val_Y4 = pd.read_table('test_Y4.txt', header=None)

        for param1ind in range(len(paramset1)):  # hidden neurons
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)):  # learning rate
                param2value = paramset2[param2ind]

                for param3ind in range(len(paramset3)):
                    param3value = paramset3[param3ind]

                    gridmodel = runmethod(FLAGS, param1value, param2value, param3value)
                    gridmodel.summary()
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

                    train_Y = np.squeeze(np.array(train_Y4))
                    train_Y_one_hot = (np.arange(train_Y.max() + 1) == train_Y[:, None]).astype(dtype='int')
                    val_Y = np.squeeze(np.array(val_Y4))
                    val_Y_one_hot = (np.arange(val_Y.max() + 1) == val_Y[:, None]).astype(dtype='int')

                    gridmodel.fit(([np.array(train_drugs4), np.array(train_prots4)]), train_Y_one_hot,
                                            batch_size=batchsz, epochs=epoch,
                                            validation_data=(
                                                ([np.array(test_drugs4), np.array(test_prots4)]), val_Y_one_hot),
                                            shuffle=True, callbacks=[es])
                    all_drugs4 = pd.read_table('all_drugs4.txt', header=None)
                    all_prots4 = pd.read_table('all_prots4.txt', header=None)
                    permute_layer_model = Model(input=gridmodel.input,output=gridmodel.get_layer('dense_3').output)
                    permute_layer_output = permute_layer_model.predict([np.array(all_drugs4),np.array(all_prots4) ])
                    print(permute_layer_output)
                    np.savetxt("output"+str(param1ind)+"_"+str(param2ind)+"_"+str(param3ind)+".txt", permute_layer_output)
                    print("output" + str(param1ind) + "_" + str(param2ind) + "_" + str(param3ind))

                    predicted_labels = gridmodel.predict([np.array(test_drugs4), np.array(test_prots4)])

                    pred_type = np.argmax(predicted_labels, axis=1)
                    predicted_labels2 = pd.DataFrame(predicted_labels)
                    predicted_labels2.to_csv('predicted_labels.txt', sep='\t', header=None, index=False)
                    a = accuracy_score(np.array(val_Y), pred_type)
                    b = precision_score(np.array(val_Y), pred_type, average='macro')
                    c = recall_score(np.array(val_Y), pred_type, average='macro')

                    logging("a = %d,  b = %d, c = %d, a = %d, b = %f, c = %f, a = %f" %
                            (a, b, c, a, b, c, a), FLAGS)


    bestperf = -float('Inf')
    bestpointer = None

    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
        for param2ind in range(len(paramset2)):
            for param3ind in range(len(paramset3)):

                avgperf = 0.
                for foldind in range(len(val_sets)):
                    foldperf = all_predictions[pointer][foldind]
                    avgperf += foldperf
                avgperf /= len(val_sets)
                # print(epoch, batchsz, avgperf)
                if avgperf > bestperf:
                    bestperf = avgperf
                    bestpointer = pointer
                    best_param_list = [param1ind, param2ind, param3ind]

                pointer += 1

    return bestpointer, best_param_list, bestperf, all_predictions, all_losses


def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g / f)  # select


def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity = []

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target = XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data, target_data, affinity


def experiment(FLAGS, perfmeasure, deepmethod, foldcount=6):  # 5-fold cross validation + test

    # Input
    # XD: [drugs, features] sized array (features may also be similarities with other drugs
    # XT: [targets, features] sized array (features may also be similarities with other targets
    # Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    # perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    # higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    # foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation

    dataset = DataSet(fpath=FLAGS.dataset_path,  ### BUNU ARGS DA GUNCELLE
                      setting_no=FLAGS.problem_type,  ##BUNU ARGS A EKLE
                      seqlen=FLAGS.max_seq_len,
                      smilen=FLAGS.max_smi_len,
                      need_shuffle=False)
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size

    XD, XT, Y = dataset.parse_data(FLAGS)

    XD = np.asarray(XD)
    # XD2 = pd.DataFrame(XD)
    # XD2.to_csv('XD.txt', sep='\t', header=None, index=False)
    XT = np.asarray(XT)
    # XT2 = pd.DataFrame(XT)
    # XT2.to_csv('XT.txt', sep='\t', header=None, index=False)
    Y = np.asarray(Y)
    # Y2 = pd.DataFrame(Y)
    # Y2.to_csv('Y.txt', sep='\t', header=None, index=False)

    drugcount = XD.shape[0]
    print(drugcount)
    targetcount = XT.shape[0]
    print(targetcount)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(
        np.isnan(Y) == False)  # basically finds the point address of affinity [x,y]

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)
    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds,
                                                                    perfmeasure, deepmethod, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
            (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)


def run_regression(FLAGS):
    perfmeasure = get_cindex
    deepmethod = build_combined_categorical
    experiment(FLAGS, perfmeasure, deepmethod)


if __name__ == "__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    run_regression(FLAGS)
