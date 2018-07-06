# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np

from lib import *


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
        :param window_state:
        :param trading_cost:
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
        reward = self.direction * (self.price[self.t + 1] - self.price[self.t])
        # TODO: need to double check risk_averse setting
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


if __name__ == '__main__':
    pass
