# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import os
from os import path
import argparse
from utils import mkdir_p

from sampler import *
from agents import *
from emulator import *
from simulators import *
from visualizer import *
from dataset import PriceDataSampler
from emulator import MarketEmulator


def train():

    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('--mode', action="store",
                        dest='mode', type=str, default='train')
    parser.add_argument('--run', action="store",
                        dest='run', type=str, default='RUN')
    parser.add_argument('--project', action="store",
                        dest='project', type=str, default='project-00')
    parser.add_argument('--net', action="store",
                        dest='net', type=str, default='LSTM')
    parser.add_argument('--verbose', action="store_const",
                        dest='verbose', const=True, default=False)
    args = parser.parse_args()


    mkdir_p(args.run)

    model_type = 'conv'
    n_episode = 1000

    # runtime folder to save logs/models
    rt_folder = path.join(args.run, args.project)
    mkdir_p(rt_folder)
    print('project training folder: {}'.format(rt_folder))

    test_input_file = '../data/index/spy.csv'
    sampler = PriceDataSampler()
    sampler.load_price_data(test_input_file)
    sampler.split_data(600, 0.2)
    market = MarketEmulator(sampler)

    # model param
    batch_size = 32
    learning_rate = 0.001
    epsilon = 0.25

    agent = SimpleDQNAgent(learning_rate=learning_rate, batch_size=batch_size, epsilon=epsilon)
    agent.build_model(model_type, market.state_shape, market.n_action)

    simulator = SimpleSimulator(agent, market, None, rt_folder)

    if args.mode == 'train':
        simulator.train(n_episode, save_per_episode=50)


if __name__ == '__main__':
    train()
