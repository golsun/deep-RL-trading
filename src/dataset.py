# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division


import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from utils import mkdir_p


class PriceDataSampler(object):
    '''

    ----------------------------------Full TS Dataset-------------------------------------------------
                                | split into |
    ---(train)----- ----(val)-----      ------(train)------------  -----(val)------  ...............



    ----------------------------------(train)-----------------------------------------------------------
                                        |(sampling)|
    --(lookback_period)-- --(reward_period)-- --(lookback_period)-- --(reward_period)--
                                        |(shuffle)|
    [--(lookback_period)-- --(reward_period)--,   --(lookback_period)-- --(reward_period)--,  ....]

    '''

    def __init__(self):
        self.lookback_period = 40
        self.reward_period = 5
        # only 1 dimension: close price
        self.n_var = 1
        self.full_ts = None

        self.dataset_train = None
        self.dataset_val = None

    def load_price_data(self, csv_file_name):
        '''

        :param csv_file_name: a csv file with OHLC header: Date,Open,High,Low,Close,Volume
        :return:
        '''
        df = pd.read_csv(csv_file_name, index_col='Date', parse_dates=True)
        self.full_ts = df

        # t_start='9:00'
        # t_end='15:00'
        # df = df.between_time(t_start, t_end)
        # #
        # # Resample data to 10min
        # #
        # price_10min_df = spread_df[['Current_Close', 'Next_Close']].resample('10Min', how='last', closed='left',
        #                                                                      label='left')
        # vol_10min_df = spread_df[['Current_Volume', 'Next_Volume']].resample('10Min', how=np.sum, closed='left',
        #                                                                      label='left')

    def split_data(self, partition_size, split_ratio=0.1):
        ''' split data into 'train' and 'val' dataset

        :return:
        '''
        assert (isinstance(partition_size, int))

        def sample_ndarray(array, unit_size, mode='seq', sample_ratio=None):
            '''

            :param array:
            :param unit_size: the target window (unit) size, unit_size = loopback len + reward len
            :param mode: 'seq' or 'random'
            :param sample_ratio: float, only useful in random mode. num of % in total available samples
            :return: a list of ndarray
            '''
            slices = []
            # |-------------------input ndarray size----------------------|
            # |------total len minus unit_size ---- |----unit_size----|
            input_len = len(array)
            sample_len = len(array) - unit_size

            if mode == 'seq':
                slices = [array[start: start + unit_size] for start in range(sample_len - 1)]
                # slices.append(array[start: start+unit_size]) for start in range(sample_len-1)
            elif mode == 'random':
                repeat = int(0.5 * sample_len if sample_ratio is None else sample_len * sample_ratio)
                for _ in range(repeat):
                    start = random.randint(0, (input_len - unit_size))
                    slices.append(array[start: start + unit_size])
            else:
                raise ValueError

            return slices

        if int(split_ratio * partition_size) < (self.lookback_period + self.reward_period):
            raise ValueError

        prices = self.full_ts['Close'].values

        # example of np.split
        # np.split(range(100), [a*11 for a in range(1,9)])
        n_parts = int(len(prices) / partition_size)
        prices_partition = np.split(prices, [separator * partition_size for separator in range(1, n_parts)])

        prices_partition_train = []
        prices_partition_val = []

        for _p in prices_partition:
            train_slice, val_slice = np.split(_p, [int(len(_p) * (1 - split_ratio))])
            prices_partition_train.append(train_slice)
            prices_partition_val.append(val_slice)

        # sample the lookback_period + reward_period window from train and val price partitions
        dataset_train = []
        dataset_val = []
        window_size = self.lookback_period + self.reward_period

        for _ts_part in prices_partition_train:
            #dataset_train += sample_ndarray(_ts_part, window_size, 'random', sample_ratio=0.8)
            dataset_train += sample_ndarray(_ts_part, window_size, 'seq')
        random.shuffle(dataset_train)

        for _ts_part in prices_partition_val:
            dataset_val += sample_ndarray(_ts_part, window_size, 'seq')
        random.shuffle(dataset_val)

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val


    def sampler(self):
        pass

    def save_db(self):
        ''' pickle the sampler (data and config) '''
        pass


if __name__ == '__main__':
    #
    # unit test
    #
    test_input_file = '../data/index/spy.csv'
    sampler = PriceDataSampler()
    sampler.load_price_data(test_input_file)
    print(sampler.full_ts.head(10))
    print(sampler.full_ts.describe())

    sampler.split_data(500, split_ratio=0.2)
    print('items in dataset_train: {}'.format(len(sampler.dataset_train)))
    print('items in dataset_val: {}'.format(len(sampler.dataset_val)))
    print('-' * 80)
    print('len of dataset_train[0]: {}'.format(len(sampler.dataset_train[0])))
    print('dataset_train[0]: {}'.format(sampler.dataset_train[0]))
    print('-' * 80)
    print('len of dataset_val[0]: {}'.format(len(sampler.dataset_val[0])))
    print('dataset_val[0]: {}'.format(sampler.dataset_val[0]))
