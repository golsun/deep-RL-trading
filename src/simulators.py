# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
from utils import mkdir_p

class Simulator:

    def __init__(self, agent, env, visualizer, fld_save):
        self.agent = agent
        self.env = env
        self.visualizer = visualizer
        self.fld_save = fld_save

    def epsilon_scheduler(self):
        ''' reduce agent's epsilon (explosion ratio）'''
        if self.agent.epsilon >= self.agent.epsilon_min:
            self.agent.epsilon = self.agent.epsilon * self.agent.epsilon_decay

    def play_one_episode(self, training=True, rand_price=True, verbose=True):

        state, valid_actions = self.env.reset(rand_price=rand_price)
        done = False
        env_t = 0
        try:
            env_t = self.env.t
        except AttributeError:
            pass

        cum_rewards = [np.nan] * env_t
        actions = [np.nan] * env_t
        states = [None] * env_t
        prev_cum_rewards = 0.

        if verbose:
            print('-'*20 + 'play_one_episode initial status' + '-'*20)
            #print('state:{} valid_actions: {} env_t: {}'.format(state, valid_actions, env_t))
            print('state:{} valid_actions: {} env_t: {}'.format(state, valid_actions, env_t))

        step = 0
        while not done:
            action = self.agent.act(state, valid_actions)

            # TODO: check if the 'valid_actions' in next line should put to 'next_valid_actions'
            next_state, reward, done, valid_actions = self.env.step(action)

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
                #print('step {}: action {} state {}, next_state {}, reward {}, done {}, (next)valid_actions {}'.format(step, action, state, next_state, reward, done, valid_actions))
                print('step {}: action {} reward {}, cum_rewards {}, done {}, (next)valid_actions {}'.format(step, action, reward, cum_rewards, done, valid_actions))

            state = next_state
            step += 1

        if verbose:
            print('cum_rewards: {}'.format(cum_rewards))
            print('actions: {}'.format(actions))
            #print('states: {}'.format(states))

        # decay epsilon
        if training:
            self.epsilon_scheduler()

        return cum_rewards, actions, states

    def train(self, n_episode, save_per_episode=100):

        fld_model = os.path.join(self.fld_save, 'model')
        mkdir_p(fld_model)  # don't overwrite if already exists
        with open(os.path.join(fld_model, 'QModel.txt'), 'w') as f:
            f.write(self.agent.model.qmodel)

        fld_save = os.path.join(self.fld_save, 'training')

        mkdir_p(fld_save)
        MA_window = 100  # MA of performance
        eval_total_rewards = []
        train_total_rewards = []
        explorations = []
        path_record = os.path.join(fld_save, 'record.csv')

        print('\nStart training...')
        with open(path_record, 'w') as f:
            f.write('episode,game,exploration,train_rewards,val_rewards,train_median_rewards,eval_median_rewards\n')
            print('episode\tgame\texploration\ttrain_rewards\tval_rewards\ttrain_median_rewards\teval_median_rewards\n')

        for n in range(n_episode):

            train_cum_rewards, explored_actions, _ = self.play_one_episode(training=True, rand_price=True, verbose=False)
            train_total_rewards.append(train_cum_rewards[-1])
            explorations.append(self.agent.epsilon)

            eval_cum_rewards, eval_actions, _ = self.play_one_episode(training=False, rand_price=False, verbose=False)
            eval_total_rewards.append(eval_cum_rewards[-1])

            MA_total_rewards = np.median(train_total_rewards[-MA_window:])
            MA_eval_total_rewards = np.median(eval_total_rewards[-MA_window:])

            ss = [str(n), self.env.title.replace(',', ';'), '%.1f' % (self.agent.epsilon * 100.),
                  '%.1f' % (train_total_rewards[-1]),
                  '%.1f' % (eval_total_rewards[-1]),
                  '%.1f' % MA_total_rewards,
                  '%.1f' % MA_eval_total_rewards]

            with open(path_record, 'a') as f:
                f.write(','.join(ss) + '\n')
                print('\t'.join(ss))

            if save_per_episode > 0 and n % save_per_episode == 0:

                self.agent.save(fld_model)

                self.visualizer.plot_a_episode(
                    self.env, self.agent.model,
                    train_cum_rewards, explored_actions,
                    eval_cum_rewards, eval_actions,
                    os.path.join(fld_save, 'episode_%i.png' % (n)))

                self.visualizer.plot_episodes(
                    train_total_rewards, eval_total_rewards, explorations,
                    os.path.join(fld_save, 'total_rewards.png'),
                    MA_window)


    def test(self, n_episode, save_per_episode=10, subfld='testing'):

        fld_save = os.path.join(self.fld_save, subfld)
        mkdir_p(fld_save)
        MA_window = 100  # MA of performance
        eval_total_rewards = []
        path_record = os.path.join(fld_save, 'record.csv')

        with open(path_record, 'w') as f:
            f.write('episode,game,pnl,rel,MA\n')

        for n in range(n_episode):
            print('\ntesting...')

            eval_cum_rewards, eval_actions, _ = self.play_one_episode(training=False, rand_price=True)
            eval_total_rewards.append(eval_cum_rewards[-1])
            MA_eval_total_rewards = np.median(eval_total_rewards[-MA_window:])
            ss = [str(n), self.env.title.replace(',', ';'),
                  '%.1f' % (eval_cum_rewards[-1]),
                  '%.1f' % (eval_total_rewards[-1]),
                  '%.1f' % MA_eval_total_rewards]

            with open(path_record, 'a') as f:
                f.write(','.join(ss) + '\n')
                print('\t'.join(ss))

            if save_per_episode > 0 and n % save_per_episode == 0:
                print('saving results...')
                self.visualizer.plot_a_episode(
                    self.env, self.agent.model, 
                    [np.nan]*len(eval_cum_rewards), [np.nan]*len(eval_actions),
                    eval_cum_rewards, eval_actions,
                    os.path.join(fld_save, 'episode_%i.png'%(n)))
    
                self.visualizer.plot_episodes(None, eval_total_rewards, None,
                                        os.path.join(fld_save, 'total_rewards.png'), MA_window)


if __name__ == '__main__':
    # print 'episode%i, init%i'%(1,2)
    a = [1, 2, 3]
    print(np.mean(a[-100:]))
