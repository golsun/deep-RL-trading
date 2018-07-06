# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import os
import pickle
import random
import keras
import numpy as np
from utils import mkdir_p


class DQNAgent:
    '''
    refer to https://keon.io/deep-q-learning/ for a few design mindsets
    '''

    def __init__(self, learning_rate=0.001, batch_size=32, discount_factor=0.96, epsilon=1.0, epsilon_min=0.001,
                 epsilon_decay=0.995):

        self.model = None

        # the memory list
        # memory = [(state, action, reward, next_state, done)...]
        self.memory = []

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = discount_factor  # discount rate
        # exploration rate, in which an agent randomly decides its action rather than prediction.
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.enable_epsilon_decay = False

        self.__print_hpye_params()

    def __print_hpye_params(self):
        print(20 * '-' + 'Init DQNAgent' + 20 * '-')
        print("batch_size                :{}".format(self.batch_size))
        print("learning_rate             :{}".format(self.learning_rate))
        print("gamma                     :{}".format(self.gamma))
        print("epsilon                   :{}".format(self.epsilon))
        print("epsilon_min               :{}".format(self.epsilon_min))
        print("epsilon_decay             :{}".format(self.epsilon_decay))
        print("enable_epsilon_decay      :{}".format(self.enable_epsilon_decay))
        print('\n')

    def remember(self, state, action, reward, next_state, done, next_valid_actions):
        ''' simply store states, actions and resulting rewards to the memory

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done: boolean flag indicates if the state is the final state.
        :param next_valid_actions:
        :return:
        '''
        self.memory.append((state, action, reward, next_state, done, next_valid_actions))

    def replay(self):
        # Sample minibatch from the memory
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        for state, action, reward, next_state, done, next_valid_actions in batch:
            # TODO: double check if it's okay to change 'q' ('reward') value
            target = reward
            if not done:
                target += self.gamma * np.nanmax(self.get_q_valid(next_state, next_valid_actions))
            self.model.fit(state, action, target)

    def get_q_valid(self, state, valid_actions):
        q = self.model.predict(state)
        q_valid = [np.nan] * len(q)

        for action in valid_actions:
            q_valid[action] = q[action]

        return q_valid

    def act(self, state, valid_actions, is_training=True):

        if np.random.random() <= self.epsilon and is_training:
            # only in training mode
            # randomly pick 1 action from valid actions, and return a value (not list)
            return random.sample(valid_actions, 1)[0]
        else:
            # for both training and eval
            q_valid = self.get_q_valid(state, valid_actions)
            # if np.nanmin(q_valid) != np.nanmax(q_valid):
            #     return np.nanargmax(q_valid)
            # NOTE: the index is an action number
            return np.nanargmax(q_valid)

    def save(self, fld):
        ''' Save agent attribute and model
        :param fld: output folder
        '''
        mkdir_p(fld)

        attr = {
            'batch_size': self.batch_size,
            'discount_factor': self.gamma,
            # 'memory':self.memory
        }

        pickle.dump(attr, open(os.path.join(fld, 'agent_attr.pickle'), 'wb'))
        self.model.save(fld)

    def load(self, fld):
        ''' recover attributes from pickle file and load model from folder
        :param fld:
        :return:
        '''

        attr = pickle.load(open(os.path.join(fld, 'agent_attr.pickle'), 'rb'))
        for k in attr:
            setattr(self, k, attr[k])
        self.model.load(fld)

    def build_model(self, model_type, env, verbose=True):

        learning_rate = self.learning_rate

        if model_type == 'MLP':
            m = 16
            layers = 5
            hidden_size = [m] * layers
            model = QModelMLP(env.state_shape, env.n_action)
            model.build_model(hidden_size, learning_rate=learning_rate, activation='tanh')

        elif model_type == 'conv':
            m = 16
            layers = 2
            filter_num = [m] * layers
            filter_size = [3] * len(filter_num)
            # use_pool = [False, True, False, True]
            # use_pool = [False, False, True, False, False, True]
            use_pool = None
            # dilation = [1,2,4,8]
            dilation = None
            dense_units = [48, 24]
            model = QModelConv(env.state_shape, env.n_action)
            model.build_model(filter_num, filter_size, dense_units, learning_rate,
                              dilation=dilation, use_pool=use_pool)

        elif model_type == 'RNN':
            m = 32
            layers = 3
            hidden_size = [m] * layers
            dense_units = [m, m]
            model = QModelGRU(env.state_shape, env.n_action)
            model.build_model(hidden_size, dense_units, learning_rate=learning_rate)

        elif model_type == 'LSTM':
            m = 32
            layers = 3
            hidden_size = [m] * layers
            dense_units = [m, m]
            model = QModelLSTM(env.state_shape, env.n_action)
            model.build_model(hidden_size, dense_units, learning_rate=learning_rate)

        elif model_type == 'ConvRNN':
            m = 8
            conv_n_hidden = [m, m]
            RNN_n_hidden = [m, m]
            dense_units = [m, m]
            model = QModelConvGRU(env.state_shape, env.n_action)
            model.build_model(conv_n_hidden, RNN_n_hidden, dense_units, learning_rate=learning_rate)

        else:
            raise ValueError

        self.model = model

        if verbose:
            model.model.summary()

        return self.model


def add_dim(x, shape):
    return np.reshape(x, (1,) + shape)


class QModelBase:
    # ref: https://keon.io/deep-q-learning/

    def __init__(self, state_shape, n_action):
        self.state_shape = state_shape
        self.n_action = n_action
        self.attr2save = ['state_shape', 'n_action', 'model_name']
        self.qmodel = ''

    def save(self, fld):
        mkdir_p(fld)
        with open(os.path.join(fld, 'model.json'), 'w') as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(os.path.join(fld, 'weights.hdf5'))

        attr = dict()
        for a in self.attr2save:
            attr[a] = getattr(self, a)
        pickle.dump(attr, open(os.path.join(fld, 'Qmodel_attr.pickle'), 'wb'))

    def load(self, fld, learning_rate):
        json_str = open(os.path.join(fld, 'model.json')).read()
        self.model = keras.models.model_from_json(json_str)
        self.model.load_weights(os.path.join(fld, 'weights.hdf5'))
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

        attr = pickle.load(open(os.path.join(fld, 'Qmodel_attr.pickle'), 'rb'))
        for a in attr:
            setattr(self, a, attr[a])

    def predict(self, state):
        q = self.model.predict(
            add_dim(state, self.state_shape)
        )[0]

        if np.isnan(max(q)):
            print('state' + str(state))
            print('q' + str(q))
            raise ValueError

        return q

    def fit(self, state, action, q_action):
        q = self.predict(state)
        q[action] = q_action

        self.model.fit(
            add_dim(state, self.state_shape),
            add_dim(q, (self.n_action,)),
            epochs=1, verbose=0)


class QModelMLP(QModelBase):
    # multi-layer perception (MLP), i.e., dense only

    def __init__(self, state_shape, n_action):
        super().__init__(state_shape, n_action)
        self.qmodel = 'MLP'

    def build_model(self, n_hidden, learning_rate, activation='relu'):
        model = keras.models.Sequential()
        model.add(keras.layers.Reshape(
            (self.state_shape[0] * self.state_shape[1],),
            input_shape=self.state_shape))

        for i in range(len(n_hidden)):
            model.add(keras.layers.Dense(n_hidden[i], activation=activation))
        # model.add(keras.layers.Dropout(drop_rate))

        model.add(keras.layers.Dense(self.n_action, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
        self.model = model
        self.model_name = self.qmodel + str(n_hidden)


class QModelRNN(QModelBase):
    """
    https://keras.io/getting-started/sequential-model-guide/#example
    note param doesn't grow with len of sequence
    """

    def __init__(self, state_shape, n_action):
        super().__init__(state_shape, n_action)
        self.qmodel = 'RNN'

    def _build_model(self, Layer, n_hidden, dense_units, learning_rate, activation='relu'):

        model = keras.models.Sequential()
        model.add(keras.layers.Reshape(self.state_shape, input_shape=self.state_shape))
        m = len(n_hidden)
        for i in range(m):
            model.add(Layer(n_hidden[i],
                            return_sequences=(i < m - 1)))
        for i in range(len(dense_units)):
            model.add(keras.layers.Dense(dense_units[i], activation=activation))
        model.add(keras.layers.Dense(self.n_action, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
        self.model = model
        self.model_name = self.qmodel + str(n_hidden) + str(dense_units)


class QModelLSTM(QModelRNN):

    def __init__(self, state_shape, n_action):
        super().__init__(state_shape, n_action)
        self.qmodel = 'LSTM'

    def build_model(self, n_hidden, dense_units, learning_rate, activation='relu'):
        self._build_model(keras.layers.LSTM, n_hidden, dense_units, learning_rate, activation)


class QModelGRU(QModelRNN):
    def init(self):
        self.qmodel = 'GRU'

    def build_model(self, n_hidden, dense_units, learning_rate, activation='relu'):
        self._build_model(keras.layers.GRU, n_hidden, dense_units, learning_rate, activation)


class QModelConv(QModelBase):
    """
    ref: https://keras.io/layers/convolutional/
    """

    def __init__(self, state_shape, n_action):
        super().__init__(state_shape, n_action)
        self.qmodel = 'Conv'

    def build_model(self,
                    filter_num, filter_size, dense_units,
                    learning_rate, activation='relu', dilation=None, use_pool=None):

        if use_pool is None:
            use_pool = [True] * len(filter_num)
        if dilation is None:
            dilation = [1] * len(filter_num)

        model = keras.models.Sequential()
        model.add(keras.layers.Reshape(self.state_shape, input_shape=self.state_shape))

        for i in range(len(filter_num)):
            model.add(keras.layers.Conv1D(filter_num[i], kernel_size=filter_size[i], dilation_rate=dilation[i],
                                          activation=activation, use_bias=True))
            if use_pool[i]:
                model.add(keras.layers.MaxPooling1D(pool_size=2))

        model.add(keras.layers.Flatten())
        for i in range(len(dense_units)):
            model.add(keras.layers.Dense(dense_units[i], activation=activation))
        model.add(keras.layers.Dense(self.n_action, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

        self.model = model

        self.model_name = self.qmodel + str([a for a in
                                             zip(filter_num, filter_size, dilation, use_pool)
                                             ]) + ' + ' + str(dense_units)


class QModelConvRNN(QModelBase):
    """
    https://keras.io/getting-started/sequential-model-guide/#example
    note param doesn't grow with len of sequence
    """

    def _build_model(self, RNNLayer, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate,
                     conv_kernel_size=3, use_pool=False, activation='relu'):

        model = keras.models.Sequential()
        model.add(keras.layers.Reshape(self.state_shape, input_shape=self.state_shape))

        for i in range(len(conv_n_hidden)):
            model.add(keras.layers.Conv1D(conv_n_hidden[i], kernel_size=conv_kernel_size,
                                          activation=activation, use_bias=True))
            if use_pool:
                model.add(keras.layers.MaxPooling1D(pool_size=2))
        m = len(RNN_n_hidden)
        for i in range(m):
            model.add(RNNLayer(RNN_n_hidden[i],
                               return_sequences=(i < m - 1)))
        for i in range(len(dense_units)):
            model.add(keras.layers.Dense(dense_units[i], activation=activation))

        model.add(keras.layers.Dense(self.n_action, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
        self.model = model
        self.model_name = self.qmodel + str(conv_n_hidden) + str(RNN_n_hidden) + str(dense_units)


class QModelConvLSTM(QModelConvRNN):
    def __init__(self, state_shape, n_action):
        super().__init__(state_shape, n_action)
        self.qmodel = 'ConvLSTM'

    def build_model(self, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate,
                    conv_kernel_size=3, use_pool=False, activation='relu'):
        Layer = keras.layers.LSTM
        self._build_model(Layer, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate,
                          conv_kernel_size, use_pool, activation)


class QModelConvGRU(QModelConvRNN):
    def __init__(self, state_shape, n_action):
        super().__init__(state_shape, n_action)
        self.qmodel = 'ConvGRU'

    def build_model(self, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate,
                    conv_kernel_size=3, use_pool=False, activation='relu'):
        Layer = keras.layers.GRU
        self._build_model(Layer, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate,
                          conv_kernel_size, use_pool, activation)


def restore_model(fld, learning_rate):
    ''' restore model from folder
    :param fld:
    :param learning_rate:
    :return:
    '''

    s = open(os.path.join(fld, 'QModel.txt'), 'r').read().strip()
    qmodels = {
        'Conv': QModelConv,
        'DenseOnly': QModelMLP,
        'MLP': QModelMLP,
        'LSTM': QModelLSTM,
        'GRU': QModelGRU,
    }

    qmodel = qmodels[s](None, None)
    qmodel.load(fld, learning_rate)
    return qmodel
