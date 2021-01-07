#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under
#the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import create_dataset, to_torch
import time_series_prediction.models as models

# initialization
PARSER = argparse.ArgumentParser()
# In time series prediction scenario, There are four dataset:
# 'tree7', 'traffic', 'arfima', 'DJI'.
PARSER.add_argument('--dataset', type=str, default='traffic',
                    help='The test dataset')
# In review classification scenario,
# RNN, LSTM, mRNN_fixD, mLSTM_fixD, mRNN, mLSTM are tested.
PARSER.add_argument('--algorithm', type=str, default='mLSTM',
                    help='The test algorithm')
PARSER.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
PARSER.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
PARSER.add_argument('--hidden_size', type=int, default=1,
                    help='Number of hidden units.')
PARSER.add_argument('--input_size', type=int, default=1,
                    help='Number of input units, as for time series it is 1.')
PARSER.add_argument('--output_size', type=int, default=1,
                    help='Number of output units, as for time series it is 1.')
PARSER.add_argument('--K', type=int, default=100,
                    help='Truncate the infinite summation at lag K.')
PARSER.add_argument('--patience', type=int, default=100, help='Patience.')
FLAGS = PARSER.parse_args()


def main():
    batch_size = 1
    start = 0
    end = 100
    # read data
    df_data = pd.read_csv('../data/time_series_prediction/'
                          + FLAGS.dataset + '.csv')
    # split train/val/test
    if FLAGS.dataset == 'tree7':
        train_size = 2500
        validate_size = 1000
    if FLAGS.dataset == 'DJI':
        train_size = 2500
        validate_size = 1500
    if FLAGS.dataset == 'traffic':
        train_size = 1200
        validate_size = 200
    if FLAGS.dataset == 'arfima':
        train_size = 2000
        validate_size = 1200
    rmse_list = []
    mae_list = []
    for i in range(start, end):
        seed = i
        print('seed ----------------------------------', seed)
        x = np.array(df['x'])
        y = np.array(df['x'])
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        # normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        x = scaler.fit_transform(x)
        y = scaler.fit_transform(y)
        # use this function to prepare the data for modeling
        data_x, data_y = create_dataset(x, y)

        # split into train and test sets
        train_x, train_y = data_x[0:train_size], data_y[0:train_size]
        validate_x, validate_y = data_x[train_size:train_size +
                                                   validate_size], \
                                 data_y[train_size:train_size +
                                                   validate_size]
        test_x, test_y = data_x[train_size + validate_size:len(data_y)], \
                         data_y[train_size + validate_size:len(data_y)]

        # reshape input to be [time steps,samples,features]
        train_x = np.reshape(train_x,
                             (train_x.shape[0], batch_size, FLAGS.input_size))
        validate_x = np.reshape(validate_x, (validate_x.shape[0],
                                             batch_size, FLAGS.input_size))

        test_x = np.reshape(test_x,
                            (test_x.shape[0], batch_size, FLAGS.input_size))
        train_y = np.reshape(train_y,
                             (train_y.shape[0], batch_size, FLAGS.output_size))
        validate_y = np.reshape(validate_y, (validate_y.shape[0],
                                             batch_size, FLAGS.output_size))
        test_y = np.reshape(test_y, (test_y.shape[0],
                                     batch_size, FLAGS.output_size))

        torch.manual_seed(seed)
        # initialize model
        if FLAGS.algorithm == 'RNN':
            model = models.RNN(input_size=FLAGS.input_size,
                               hidden_size=FLAGS.hidden_size,
                               output_size=FLAGS.output_size)
        elif FLAGS.algorithm == 'LSTM':
            model = models.LSTM(input_size=FLAGS.input_size,
                                hidden_size=FLAGS.hidden_size,
                                output_size=FLAGS.output_size)
        elif FLAGS.algorithm == 'mRNN_fixD':
            model = models.MRNNFixD(input_size=FLAGS.input_size,
                                    hidden_size=FLAGS.hidden_size,
                                    output_size=FLAGS.output_size,
                                    k=FLAGS.K)
        elif FLAGS.algorithm == 'mRNN':
            model = models.MRNN(input_size=FLAGS.input_size,
                                hidden_size=FLAGS.hidden_size,
                                output_size=FLAGS.output_size,
                                k=FLAGS.K)
        elif FLAGS.algorithm == 'mLSTM_fixD':
            model = models.MLSTMFixD(input_size=FLAGS.input_size,
                                     hidden_size=FLAGS.hidden_size,
                                     output_size=FLAGS.output_size,
                                     k=FLAGS.K)
        elif FLAGS.algorithm == 'mLSTM':
            model = models.MLSTM(input_size=FLAGS.input_size,
                                 hidden_size=FLAGS.hidden_size,
                                 output_size=FLAGS.output_size,
                                 k=FLAGS.K)
        else:
            print('Algorithm selection ERROR!!!')
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
        best_loss = np.infty
        best_train_loss = np.infty
        stop_criterion = 1e-5
        rec = np.zeros((FLAGS.epochs, 3))
        epoch = 0
        val_loss = -1
        train_loss = -1
        cnt = 0

        def train():
            model.train()
            optimizer.zero_grad()
            target = torch.from_numpy(train_y).float()
            output, hidden_state = model(torch.from_numpy(train_x).float())
            with torch.no_grad():
                val_y, _ = model(torch.from_numpy(validate_x).float(),
                                 hidden_state)
                target_val = torch.from_numpy(validate_y).float()
                val_loss = criterion(val_y, target_val)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            return loss, val_loss

        def compute_test(best_model):
            model = best_model
            train_predict, hidden_state = model(to_torch(train_x))
            train_predict = train_predict.detach().numpy()
            val_predict, hidden_state = model(to_torch(validate_x),
                                              hidden_state)
            test_predict, _ = model(to_torch(test_x), hidden_state)
            test_predict = test_predict.detach().numpy()
            # invert predictions
            test_predict_r = scaler.inverse_transform(test_predict[:, 0, :])
            test_y_r = scaler.inverse_transform(test_y[:, 0, :])
            # calculate error
            test_rmse = math.sqrt(mean_squared_error(test_y_r[:, 0],
                                                     test_predict_r[:, 0]))
            test_mape = (abs((test_predict_r[:, 0] - test_y_r[:, 0]) /
                             test_y_r[:, 0])).mean()
            test_mae = mean_absolute_error(test_predict_r[:, 0],
                                           test_y_r[:, 0])
            return test_rmse, test_mape, test_mae

        while epoch < FLAGS.epochs:
            _time = time.time()
            loss, val_loss = train()
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_model = deepcopy(model)
            # stop_criteria = abs(criterion(val_Y, target_val) - val_loss)
            if (best_train_loss - loss) > stop_criterion:
                best_train_loss = loss
                cnt = 0
            else:
                cnt += 1
            if cnt == FLAGS.patience:
                break
            # save training records
            time_elapsed = time.time()-_time
            rec[epoch, :] = np.array([loss, val_loss, time_elapsed])
            print("epoch: {:2.0f} train_loss: {:2.5f} val_loss: {:2.5f} "
                  "time: {:2.1f}s".format(epoch, loss.item(), val_loss.item(),
                                          time_elapsed))
            epoch = epoch + 1

        # make predictions
        test_rmse, test_mape, test_mae = compute_test(best_model)

        rmse_list.append(test_rmse)
        mae_list.append(test_mae)
        print('RMSE:{}'.format(rmse_list))
        print('MAE:{}'.format(mae_list))



if __name__ == "__main__":
    main()
