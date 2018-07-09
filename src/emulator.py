# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
from utils import mkdir_p

def find_ideal(p, just_once):
    if not just_once:
        diff = np.array(p[1:]) - np.array(p[:-1])
        return sum(np.maximum(np.zeros(diff.shape), diff))
    else:
        best = 0.
        i0_best = None
        for i in range(len(p) - 1):
            best = max(best, max(p[i + 1:]) - p[i])

        return best


class Market:
    """
    state 			MA of prices, normalized using values at t
                    ndarray of shape (window_state, n_instruments * n_MA), i.e., 2D
                    which is self.state_shape

    action 			three action
                    0:	no action;
                    1:	open a position
                    2: 	close a position
    """

    def __init__(self, sampler, window_state, trading_cost, direction=1., risk_averse=0.):
        '''
        :param sampler:
        :param window_state: each 'window' of full time series, like a 'frame' in a full game
        :param trading_cost: commission + slipper cost for each buy/sell transaction
        :param direction:
        :param risk_averse:
        '''

        self.sampler = sampler
        self.window_state = window_state
        self.trading_cost = trading_cost
        self.direction = direction
        self.risk_averse = risk_averse

        self.n_action = 3
        self.state_shape = (window_state, self.sampler.n_var)
        self.action_labels = ['no-action', 'open', 'close']
        self.t0 = window_state - 1

        self.no_position_flag = True

        # Assume rand price
        prices, self.title = self.sampler.sample()
        price = np.reshape(prices[:, 0], prices.shape[0])

        self.prices = prices.copy()
        self.price = price / price[0] * 100
        self.t_max = len(self.price) - 1

        self.max_profit = find_ideal(self.price[self.t0:], False)
        self.t = self.t0

    def reset(self, rand_price=True):
        self.no_position_flag = True
        if rand_price:
            prices, self.title = self.sampler.sample()
            price = np.reshape(prices[:, 0], prices.shape[0])

            self.prices = prices.copy()
            self.price = price / price[0] * 100
            self.t_max = len(self.price) - 1

        self.max_profit = find_ideal(self.price[self.t0:], False)
        self.t = self.t0
        return self.get_state(), self.get_valid_actions()

    def get_state(self, t=None):
        if t is None:
            t = self.t

        state = self.prices[t - self.window_state + 1: t + 1, :].copy()
        for i in range(self.sampler.n_var):
            norm = np.mean(state[:, i])
            state[:, i] = (state[:, i] / norm - 1.) * 100
        return state

    def get_valid_actions(self):
        if self.no_position_flag:
            return [0, 1]  # no action or open a position
        else:
            return [0, 2]  # no action or close a position

    def step(self, action):

        # action must be in valid action list
        if action not in self.get_valid_actions():
            raise ValueError('no such action or action is not valid: action={}, valid_actions={}, no_position={}'.format(action, self.get_valid_actions(), self.no_position_flag))

        # TODO: check the 'self.direction'; currently it is 1.0
        # 'self.direction' could be uself when short  sell allowed in emulations
        reward = self.direction * (self.price[self.t + 1] - self.price[self.t])

        # # TODO: need to double check risk_averse setting
        # add add'l punishment on negative reward
        if reward < 0:
            reward *= (1. + self.risk_averse)

        if action == 0:    # no action, no trading cost
            pass
        elif action == 1:  # open
            reward -= self.trading_cost
            self.no_position_flag = False
        elif action == 2:  # close
            reward -= self.trading_cost
            self.no_position_flag = True

        self.t += 1
        return self.get_state(), reward, self.t == self.t_max, self.get_valid_actions()


class MarketEmulator(object):

    def __init__(self, dataset, trading_cost=0.0, direction=1., risk_averse=0.):
        '''
        :param sampler:
        :param window_state: each 'window' of full time series, like a 'frame' in a full game
        :param trading_cost: commission + slipper cost for each buy/sell transaction
        :param direction:
        :param risk_averse:
        '''

        self.dataset = dataset
        self.lookback_period = dataset.lookback_period
        self.reward_period = dataset.reward_period
        self.n_var = dataset.n_var
        # X's shape
        self.state_shape = (self.lookback_period, self.n_var)

        # hard coded for now
        # value is leverage, or % of position in current portfolio
        # leverage is the direct result of trading actions
        self.action_universe = [0.0, 0.5, 1.0]
        self.n_action = len(self.action_universe)
        self.prices_one_frame = None

        # reserve for future
        self.trading_cost = trading_cost
        self.direction = direction
        self.risk_averse = risk_averse

    def init_one_frame(self, prices, is_training=True):
        self.prices_one_frame = prices
        return self.get_state()

    def calc_best_reward(self):
        # find the reward window
        prices_in_reward_period = self.prices_one_frame[-1 * self.reward_period:]

        # Design Note
        # must avoid any forward looking bias here
        # use max price in reward period could introduce a bias
        # probably just use the close price in reward period

        # find the max and min value
        allow_short_sell = True if min(self.action_universe) < 0 else False
        if allow_short_sell:
            raise NotImplementedError
        else:
            # all_possible_results = []
            # for _action in self.action_universe:
            #     all_possible_results += (
            #                 _action * np.log(prices_in_reward_period / prices_in_reward_period[0])).tolist()
            # best_reward = max(all_possible_results)
            best_reward = max(np.array(self.action_universe) * np.log(prices_in_reward_period[-1] / prices_in_reward_period[0]))

        return best_reward

    def calc_reward(self, action):
        ''' calculate the reward given the input action

        :param action:
        :return:
        '''
        # find the reward window
        prices_in_reward_period = self.prices_one_frame[self.lookback_period:]

        # TODO: not yet fact in the trading cost;
        # we use log return
        reward = action * np.log(prices_in_reward_period[-1] / prices_in_reward_period[0])

        best_reward = self.calc_best_reward()
        reward_adj = reward - float(best_reward)

        return reward_adj

    def get_state(self):
        state = self.prices_one_frame[: self.lookback_period].copy()
        state = np.log(state)

        return state

    def step(self, action):
        state = self.get_state()
        reward = self.calc_reward(action)
        return state, reward

    def iter_train_dataset(self):
        for record in self.dataset.dataset_train:
            yield record

    def iter_val_dataset(self):
        for record in self.dataset.dataset_val:
            yield record

if __name__ == '__main__':
    import random
    from dataset import PriceDataSampler
    #
    # unit test
    #
    test_input_file = '../data/index/spy.csv'
    sampler = PriceDataSampler()
    sampler.load_price_data(test_input_file)
    print(sampler.full_ts.head())
    print(sampler.full_ts.describe())
    sampler.split_data(500, split_ratio=0.2)
    print('items in dataset_train: {}'.format(len(sampler.dataset_train)))
    print('items in dataset_val: {}'.format(len(sampler.dataset_val)))


    market = MarketEmulator(sampler)

    for _prices in sampler.dataset_train[20: 30]:
        action = random.sample(market.action_universe, 1)[0]

        market.init_one_frame(_prices)

        state, reward = market.run_one_frame(action)
        best_reward = market.calc_best_reward()

        print('reward (adj.) when action is {}: {}; best reward: {}'.format(action, reward, best_reward))
        print('current state: {}'.format(state))
