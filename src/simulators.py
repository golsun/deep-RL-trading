# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
from utils import mkdir_p
from collections import deque


class Simulator:
    '''
    The place where all major parties orchestra together
    '''

    def __init__(self, agent, market_emulator, visualizer, log_folder, enable_epsilon_decay=False):
        self.agent = agent
        self.market_emulator = market_emulator
        self.visualizer = visualizer
        self.log_folder = log_folder
        self.enable_epsilon_decay = enable_epsilon_decay

    def epsilon_scheduler(self):
        ''' reduce agent's epsilon (explosion ratio）'''
        if self.agent.epsilon >= self.agent.epsilon_min:
            self.agent.epsilon = self.agent.epsilon * self.agent.epsilon_decay

    def play_one_episode(self, training=True, rand_price=True, verbose=True):

        state, valid_actions = self.market_emulator.reset(rand_price=rand_price)
        done = False
        env_t = 0
        try:
            env_t = self.market_emulator.t
        except AttributeError:
            pass

        cum_rewards = [np.nan] * env_t
        actions = [np.nan] * env_t
        states = [None] * env_t
        prev_cum_rewards = 0.

        if verbose:
            print('-' * 20 + 'play_one_episode initial status' + '-' * 20)
            # print('state:{} valid_actions: {} env_t: {}'.format(state, valid_actions, env_t))
            print('state:{} valid_actions: {} env_t: {}'.format(state, valid_actions, env_t))

        if training:
            self.market_emulator.init_one_frame()
            action = self.agent.act()
            state, reward = self.market_emulator.step()

            # record
            self.agent.remember(state, action, reward, next_state, done, valid_actions)
            # train one step (fit)
            self.agent.replay()
            pass

        step = 0
        while not done:
            action = self.agent.act(state, valid_actions)

            # TODO: check if the 'valid_actions' in next line should put to 'next_valid_actions'
            next_state, reward, done, valid_actions = self.market_emulator.step(action)

            cum_rewards.append(prev_cum_rewards + reward)
            prev_cum_rewards = cum_rewards[-1]
            actions.append(action)
            states.append(next_state)

            if training:
                # record
                self.agent.remember(state, action, reward, next_state, done, valid_actions)
                # train one step (fit)
                self.agent.replay()

            if verbose:
                # print('step {}: action {} state {}, next_state {}, reward {}, done {}, (next)valid_actions {}'.format(step, action, state, next_state, reward, done, valid_actions))
                print(
                    'step {}: action {} reward {}, cum_rewards {}, done {}, (next)valid_actions {}'.format(step, action,
                                                                                                           reward,
                                                                                                           cum_rewards,
                                                                                                           done,
                                                                                                           valid_actions))

            state = next_state
            step += 1

        if verbose:
            print('cum_rewards: {}'.format(cum_rewards))
            print('actions: {}'.format(actions))
            # print('states: {}'.format(states))

        # decay epsilon
        if training and self.enable_epsilon_decay:
            self.epsilon_scheduler()

        return cum_rewards, actions, states

    def train(self, n_episode, save_per_episode=100):

        mkdir_p(self.log_folder)

        with open(os.path.join(self.log_folder, 'QModel.txt'), 'w') as f:
            f.write(self.agent.model.qmodel)

        train_out_folder = os.path.join(self.log_folder, 'training')
        mkdir_p(train_out_folder)

        MA_window = 100  # MA of performance
        eval_total_rewards = []
        train_total_rewards = []
        explorations = []
        record_fpath = os.path.join(train_out_folder, 'record.csv')

        print('\nStart training...')
        with open(record_fpath, 'w') as f:
            f.write('episode,game,exploration,train_rewards,val_rewards,train_median_rewards,eval_median_rewards\n')
            print('episode\tgame\texploration\ttrain_rewards\tval_rewards\ttrain_median_rewards\teval_median_rewards\n')

        for n in range(n_episode):

            train_cum_rewards, explored_actions, _ = self.play_one_episode(training=True, rand_price=True,
                                                                           verbose=False)
            train_total_rewards.append(train_cum_rewards[-1])
            explorations.append(self.agent.epsilon)

            eval_cum_rewards, eval_actions, _ = self.play_one_episode(training=False, rand_price=False, verbose=False)
            eval_total_rewards.append(eval_cum_rewards[-1])

            MA_total_rewards = np.median(train_total_rewards[-MA_window:])
            MA_eval_total_rewards = np.median(eval_total_rewards[-MA_window:])

            ss = [str(n), self.market_emulator.title.replace(',', ';'), '%.1f' % (self.agent.epsilon * 100.),
                  '%.1f' % (train_total_rewards[-1]),
                  '%.1f' % (eval_total_rewards[-1]),
                  '%.1f' % MA_total_rewards,
                  '%.1f' % MA_eval_total_rewards]

            with open(record_fpath, 'a') as f:
                f.write(','.join(ss) + '\n')
                print('\t'.join(ss))

            if save_per_episode > 0 and n % save_per_episode == 0:
                self.agent.save(self.log_folder)

                self.visualizer.plot_a_episode(
                    self.market_emulator, self.agent.model,
                    train_cum_rewards, explored_actions,
                    eval_cum_rewards, eval_actions,
                    os.path.join(train_out_folder, 'episode_%i.png' % (n)))

                self.visualizer.plot_episodes(
                    train_total_rewards, eval_total_rewards, explorations,
                    os.path.join(train_out_folder, 'total_rewards.png'),
                    MA_window)

    def test(self, n_episode, save_per_episode=10, subfld='testing'):

        log_folder = os.path.join(self.log_folder, subfld)
        mkdir_p(log_folder)
        MA_window = 100  # MA of performance
        eval_total_rewards = []
        record_fpath = os.path.join(log_folder, 'record.csv')

        with open(record_fpath, 'w') as f:
            f.write('episode,game,pnl,rel,MA\n')

        for n in range(n_episode):
            print('\ntesting...')

            eval_cum_rewards, eval_actions, _ = self.play_one_episode(training=False, rand_price=True)
            eval_total_rewards.append(eval_cum_rewards[-1])
            MA_eval_total_rewards = np.median(eval_total_rewards[-MA_window:])
            ss = [str(n), self.market_emulator.title.replace(',', ';'),
                  '%.1f' % (eval_cum_rewards[-1]),
                  '%.1f' % (eval_total_rewards[-1]),
                  '%.1f' % MA_eval_total_rewards]

            with open(record_fpath, 'a') as f:
                f.write(','.join(ss) + '\n')
                print('\t'.join(ss))

            if save_per_episode > 0 and n % save_per_episode == 0:
                print('saving results...')
                self.visualizer.plot_a_episode(
                    self.market_emulator, self.agent.model,
                    [np.nan] * len(eval_cum_rewards), [np.nan] * len(eval_actions),
                    eval_cum_rewards, eval_actions,
                    os.path.join(log_folder, 'episode_%i.png' % (n)))

                self.visualizer.plot_episodes(None, eval_total_rewards, None,
                                              os.path.join(log_folder, 'total_rewards.png'), MA_window)


class SimpleSimulator(object):
    '''
        only current state, do not consider next state
        do not consider the sequences of each time window while treat each a standalone (loopback + reward) period

        do not consider future rewards
        skip visualization part
    '''

    def __init__(self, agent, market_emulator, visualizer, log_folder, enable_epsilon_decay=False):
        self.agent = agent
        self.market_emulator = market_emulator
        self.visualizer = visualizer
        self.log_folder = log_folder

        self.train_history = []
        self.eval_history = []

    def play_one_episode(self, training=True, verbose=True):

        _context = dict(state_history=[], reward_history=[], action_history=[])
        if training:
            for prices_in_a_frame in self.market_emulator.iter_train_dataset():
                state = self.market_emulator.init_one_frame(prices_in_a_frame)
                action = self.agent.act(state, self.market_emulator.action_universe)
                state, reward = self.market_emulator.step(action)

                _context['state_history'].append(state)
                _context['action_history'].append(action)
                _context['reward_history'].append(reward)

                action_np = np.where(np.array(self.market_emulator.action_universe) == action, 1, 0)
                # record
                self.agent.remember(state, action_np, reward)
                # train one step (fit)
                self.agent.replay()
        else:
            for prices_in_a_frame in self.market_emulator.iter_val_dataset():
                state = self.market_emulator.init_one_frame(prices_in_a_frame)
                action = self.agent.act(state, self.market_emulator.action_universe, is_training=False)
                state, reward = self.market_emulator.step(action)

                _context['state_history'].append(state)
                _context['action_history'].append(action)
                _context['reward_history'].append(reward)
        # if verbose:
        #     print(_context['reward_history'][0])
        #     print(_context['reward_history'])
        if training:
            self.train_history.append(_context)
        else:
            self.eval_history.append(_context)

    def train(self, n_episode, save_per_episode=100):

        with open(os.path.join(self.log_folder, 'QModel.txt'), 'w') as f:
            f.write(self.agent.model.qmodel)

        train_out_folder = os.path.join(self.log_folder, 'training')
        mkdir_p(train_out_folder)
        record_fpath = os.path.join(train_out_folder, 'record.csv')

        print('\nStart training...')
        with open(record_fpath, 'w') as f:
            f.write('episode,epsilon,train_avg_rewards\n')
            print('episode\tepsilon\ttrain_avg_rewards\n')

        for n in range(n_episode):
            self.play_one_episode(training=True, verbose=True)
            avg_reward = np.mean(self.train_history[-1]['reward_history'])
            ss = [str(n), '%.1f' % (self.agent.epsilon), '%.6f' % (avg_reward)]

            with open(record_fpath, 'a') as f:
                f.write(','.join(ss) + '\n')
                print('\t'.join(ss))

            if save_per_episode > 0 and n % save_per_episode == 0:
                self.agent.save(self.log_folder)

            if n % 1 == 0:
                self.evaluate()

    def evaluate(self):
        self.play_one_episode(training=False, verbose=True)
        avg_reward = np.mean(self.eval_history[-1]['reward_history'])
        ss = ['evaluate result:', 'epsilon %.1f' % (self.agent.epsilon), 'avg_reward %.6f' % (avg_reward)]
        print('\t'.join(ss))



if __name__ == '__main__':
    pass
