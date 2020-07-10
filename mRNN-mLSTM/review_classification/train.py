#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it
# under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for
# more details.

import json
import random
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from copy import deepcopy
from utils import padding, accuracy
import review_classification.models as models


# initialization
PARSER = argparse.ArgumentParser()
# In review classification scenario,
# RNN, LSTM, mRNN_fixD, mLSTM_fixD are tested.
PARSER.add_argument('--algorithm', type=str, default='mLSTM_fixD',
                    help='The test algorithm.')
PARSER.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
PARSER.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
PARSER.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
PARSER.add_argument('--hidden_size', type=int, default=128,
                    help='Number of hidden units.')
PARSER.add_argument('--batch_size', type=int, default=64,
                    help='Number of batch size.')
PARSER.add_argument('--nb_class', type=int, default=5,
                    help='Number of class.')
PARSER.add_argument('--pad_size', type=int, default=150,
                    help='The padding size.')
PARSER.add_argument('--K', type=int, default=50,
                    help='Truncate the infinite summation at lag K.')
PARSER.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
FLAGS = PARSER.parse_args()


def main():
    batch_size = FLAGS.batch_size
    hidden_size = FLAGS.hidden_size
    nb_class = FLAGS.nb_class
    dropout = FLAGS.dropout
    k = FLAGS.K

    # split train/val/test
    train_size = 282
    val_size = 50

    with open('../data/review_classification/data.json', 'r') as files:
        data_dict = json.load(files)
    data = data_dict['data']
    label = np.array(data_dict['label'])
    length_list = np.array([len(term)-1 if len(term) < FLAGS.pad_size else
                           FLAGS.pad_size - 1 for term in data])
    input_size = len(data[0][0])
    data_pad = padding(data, FLAGS.pad_size, input_size)
    permutation = np.random.RandomState(seed=0).permutation(len(label))
    data_pad = data_pad[permutation]
    label = label[permutation]
    length_list = length_list[permutation]

    train_data = data_pad[0:train_size]
    train_label = label[0:train_size]
    train_length = length_list[0:train_size]

    val_data = data_pad[train_size:train_size+val_size]
    val_label = label[train_size:train_size+val_size]
    val_length = length_list[train_size:train_size+val_size]

    test_data = data_pad[train_size+val_size:]
    test_label = label[train_size+val_size:]
    test_length = length_list[train_size+val_size:]

    train_data = np.reshape(train_data, (train_data.shape[1],
                                         train_data.shape[0],
                                         train_data.shape[2]))
    val_data = np.reshape(val_data, (val_data.shape[1],
                                     val_data.shape[0],
                                     val_data.shape[2]))
    test_data = np.reshape(test_data, (test_data.shape[1],
                                       test_data.shape[0],
                                       test_data.shape[2]))

    train_data = torch.FloatTensor(train_data)
    train_label = torch.LongTensor(train_label)
    train_length = torch.LongTensor(train_length)

    val_data = torch.FloatTensor(val_data)
    val_label = torch.LongTensor(val_label)
    val_length = torch.LongTensor(val_length)

    test_data = torch.FloatTensor(test_data)
    test_label = torch.LongTensor(test_label)
    test_length = torch.LongTensor(test_length)

    acc_list = []
    loss_list = []
    f1_list = []
    pre_list = []
    recall_list = []

    for times in range(100):
        random.seed(times)
        np.random.seed(times)
        torch.manual_seed(times)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(times)
        if FLAGS.algorithm == 'RNN':
            model = models.RNN(input_size, hidden_size, nb_class, dropout)
        elif FLAGS.algorithm == 'LSTM':
            model = models.LSTM(input_size, hidden_size, nb_class, dropout)
        elif FLAGS.algorithm == 'mRNN_fixD':
            model = models.MRNNFixD(input_size=input_size,
                                    hidden_size=hidden_size,
                                    output_size=nb_class,
                                    k=k,
                                    dropout=dropout)
        elif FLAGS.algorithm == 'mLSTM_fixD':
            model = models.MLSTMFixD(input_size=input_size,
                                     hidden_size=hidden_size,
                                     output_size=nb_class,
                                     k=k)
        else:
            print('Algorithm selection ERROR!!!')
        optimizer = optim.Adam(model.parameters(),
                               lr=FLAGS.lr,
                               weight_decay=FLAGS.weight_decay)

        if torch.cuda.is_available():
            model.cuda()
            train_data = train_data.cuda()
            train_label = train_label.cuda()
            val_data = val_data.cuda()
            val_label = val_label.cuda()
            test_data = test_data.cuda()
            test_label = test_label.cuda()
            test_length = test_length.cuda()
            train_length = train_length.cuda()
            val_length = val_length.cuda()

        def train(epoch):
            t_0 = time.time()
            total_batch = np.ceil(train_data.shape[1] / batch_size)
            loss_train_avg = 0.0
            acc_train_avg = 0.0

            for batch_num in range(int(total_batch)):
                if batch_num == total_batch - 1:
                    batch_input = train_data[:, batch_num * batch_size:]
                    batch_label = train_label[batch_num * batch_size:]
                    batch_length = train_length[batch_num * batch_size:]
                else:
                    batch_input = train_data[:, batch_num * batch_size:
                                                (batch_num + 1) * batch_size]
                    batch_label = train_label[batch_num * batch_size:
                                              (batch_num + 1) * batch_size]
                    batch_length = train_length[batch_num * batch_size:
                                                (batch_num + 1) * batch_size]
                model.train()
                optimizer.zero_grad()
                logit = model(batch_input, batch_length)
                loss_train = F.nll_loss(logit, batch_label)
                acc_train = accuracy(logit, batch_label)
                loss_train_avg += loss_train.data.item()
                acc_train_avg += acc_train.item()
                loss_train.backward()
                optimizer.step()
            loss_train_avg = loss_train_avg / total_batch
            acc_train_avg = acc_train_avg / total_batch

            total_batch = np.ceil(val_data.shape[1] / batch_size)
            loss_val_avg = 0.0
            acc_val_avg = 0.0
            with torch.no_grad():
                for batch_num in range(int(total_batch)):
                    if batch_num == total_batch - 1:
                        batch_input = val_data[:, batch_num * batch_size:]
                        batch_label = val_label[batch_num * batch_size:]
                        batch_length = val_length[batch_num * batch_size:]
                    else:
                        batch_input = val_data[:, batch_num * batch_size:
                                                  (batch_num + 1) * batch_size]
                        batch_label = val_label[batch_num * batch_size:
                                                (batch_num + 1) * batch_size]
                        batch_length = val_length[batch_num * batch_size:
                                                  (batch_num + 1) * batch_size]
                    logit_val = model(batch_input, batch_length)
                    loss_val = F.nll_loss(logit_val, batch_label)
                    acc_val = accuracy(logit_val, batch_label)
                    loss_val_avg += loss_val.data.item()
                    acc_val_avg += acc_val.item()
                loss_val_avg = loss_val_avg/total_batch
                acc_val_avg = acc_val_avg / total_batch

            print('Train Stage, Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train_avg),
                  'acc_train: {:.4f}'.format(acc_train_avg),
                  'loss_val: {:.4f}'.format(loss_val_avg),
                  'acc_val: {:.4f}'.format(acc_val_avg),
                  'time_cost: {:.4f}s'.format(time.time()-t_0))
            return loss_val_avg, acc_val_avg
        loss_val_list = []
        best_loss = 100.
        best_epoch = 0
        bad_counter = 0
        for epoch in range(100):
            loss_val_avg, acc_val_avg = train(epoch)
            if loss_val_avg < best_loss:
                best_loss = loss_val_avg
                best_epoch = epoch
                best_model = deepcopy(model)

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        model = best_model
        model.eval()
        total_batch = np.ceil(test_data.shape[1] / batch_size)
        loss_test_avg = 0.0
        acc_test_avg = 0.0
        f1_test_avg = 0.0
        pre_test_avg = 0.0
        recall_test_avg = 0.0
        with torch.no_grad():
            for batch_num in range(int(total_batch)):
                if batch_num == total_batch - 1:
                    batch_input = test_data[:, batch_num * batch_size:]
                    batch_label = test_label[batch_num * batch_size:]
                    batch_length = test_length[batch_num * batch_size:]
                else:
                    batch_input = test_data[:, batch_num * batch_size:
                                               (batch_num + 1) * batch_size]
                    batch_label = test_label[batch_num * batch_size:
                                             (batch_num + 1) * batch_size]
                    batch_length = test_length[batch_num * batch_size:
                                               (batch_num + 1) * batch_size]
                logit_test = model(batch_input, batch_length)
                loss_test = F.nll_loss(logit_test, batch_label)
                acc_test = accuracy(logit_test, batch_label)
                pred = logit_test.max(1)[1].type_as(test_label).cpu().\
                    detach().numpy()
                loss_test_avg += loss_test.data.item()
                acc_test_avg += acc_test.item()
                f1_test_avg += f1_score(test_label.cpu().detach().numpy(),
                                        pred, average='macro')
                pre_test_avg += precision_score(test_label.cpu().detach().
                                                numpy(), pred, average='macro')
                recall_test_avg += recall_score(test_label.cpu().detach().
                                                numpy(), pred, average='macro')
            loss_test_avg = loss_test_avg/total_batch
            acc_test_avg = acc_test_avg / total_batch
            f1_test_avg = f1_test_avg / total_batch
            pre_test_avg = pre_test_avg / total_batch
            recall_test_avg = recall_test_avg / total_batch

            print("Test set results:",
                  "loss= {:.4f}".format(loss_test_avg),
                  "accuracy= {:.4f}".format(acc_test_avg))
        acc_list.append(acc_test_avg)
        loss_list.append(loss_test_avg)
        f1_list.append(f1_test_avg)
        pre_list.append(pre_test_avg)
        recall_list.append(recall_test_avg)
        print(acc_list)
        print(loss_list)
        print(f1_list)
        print(pre_list)
        print(recall_list)

    print('ACC avg:', np.mean(acc_list), 'std:', np.std(acc_list),
          'max:', np.max(acc_list))
    print('LOSS avg:', np.mean(loss_list), 'std:', np.std(loss_list),
          'min', np.min(loss_list))
    print('F1 avg:', np.mean(f1_list), 'std:', np.std(f1_list),
          'max:', np.max(f1_list))
    print('Presicion avg:', np.mean(pre_list), 'std:', np.std(pre_list),
          'max:', np.max(pre_list))
    print('Recall avg:', np.mean(recall_list), 'std:', np.std(recall_list),
          'max:', np.max(recall_list))


if __name__ == "__main__":
    main()