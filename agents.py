from lib import *

class Agent:

	def __init__(self, model, 
		batch_size=32, discount_factor=0.95):

		self.model = model
		self.batch_size = batch_size
		self.discount_factor = discount_factor
		self.memory = []


	def remember(self, state, action, reward, next_state, done, next_valid_actions):
		self.memory.append((state, action, reward, next_state, done, next_valid_actions))


	def replay(self):
		batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
		for state, action, reward, next_state, done, next_valid_actions in batch:
			q = reward
			if not done:
				q += self.discount_factor * np.nanmax(self.get_q_valid(next_state, next_valid_actions))
			self.model.fit(state, action, q)


	def get_q_valid(self, state, valid_actions):
		q = self.model.predict(state)
		q_valid = [np.nan] * len(q)
		for action in valid_actions:
			q_valid[action] = q[action]
		return q_valid


	def act(self, state, exploration, valid_actions):
		if np.random.random() > exploration:
			q_valid = self.get_q_valid(state, valid_actions)
			if np.nanmin(q_valid) != np.nanmax(q_valid):
				return np.nanargmax(q_valid)
		return random.sample(valid_actions, 1)[0]


	def save(self, fld):
		makedirs(fld)

		attr = {
			'batch_size':self.batch_size, 
			'discount_factor':self.discount_factor, 
			#'memory':self.memory
			}

		pickle.dump(attr, open(os.path.join(fld, 'agent_attr.pickle'),'wb'))
		self.model.save(fld)

	def load(self, fld):
		path = os.path.join(fld, 'agent_attr.pickle')
		print(path)
		attr = pickle.load(open(path,'rb'))
		for k in attr:
			setattr(self, k, attr[k])
		self.model.load(fld)


def add_dim(x, shape):
	return np.reshape(x, (1,) + shape)



class QModelKeras:
	# ref: https://keon.io/deep-q-learning/
	
	def init(self):
		pass

	def build_model(self):
		pass

	def __init__(self, state_shape, n_action):
		self.state_shape = state_shape
		self.n_action = n_action
		self.attr2save = ['state_shape','n_action','model_name']
		self.init()


	def save(self, fld):
		makedirs(fld)
		with open(os.path.join(fld, 'model.json'), 'w') as json_file:
			json_file.write(self.model.to_json())
		self.model.save_weights(os.path.join(fld, 'weights.hdf5'))

		attr = dict()
		for a in self.attr2save:
			attr[a] = getattr(self, a)
		pickle.dump(attr, open(os.path.join(fld, 'Qmodel_attr.pickle'),'wb'))

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
			print('state'+str(state))
			print('q'+str(q))
			raise ValueError

		return q

	def fit(self, state, action, q_action):
		q = self.predict(state)
		q[action] = q_action

		self.model.fit(
			add_dim(state, self.state_shape), 
			add_dim(q, (self.n_action,)), 
			epochs=1, verbose=0)



class QModelMLP(QModelKeras):
	# multi-layer perception (MLP), i.e., dense only

	def init(self):
		self.qmodel = 'MLP'	

	def build_model(self, n_hidden, learning_rate, activation='relu'):

		model = keras.models.Sequential()
		model.add(keras.layers.Reshape(
			(self.state_shape[0]*self.state_shape[1],), 
			input_shape=self.state_shape))

		for i in range(len(n_hidden)):
			model.add(keras.layers.Dense(n_hidden[i], activation=activation))
			#model.add(keras.layers.Dropout(drop_rate))
		
		model.add(keras.layers.Dense(self.n_action, activation='linear'))
		model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
		self.model = model
		self.model_name = self.qmodel + str(n_hidden)
		


class QModelRNN(QModelKeras):
	"""
	https://keras.io/getting-started/sequential-model-guide/#example
	note param doesn't grow with len of sequence
	"""

	def _build_model(self, Layer, n_hidden, dense_units, learning_rate, activation='relu'):

		model = keras.models.Sequential()
		model.add(keras.layers.Reshape(self.state_shape, input_shape=self.state_shape))
		m = len(n_hidden)
		for i in range(m):
			model.add(Layer(n_hidden[i],
				return_sequences=(i<m-1)))
		for i in range(len(dense_units)):
			model.add(keras.layers.Dense(dense_units[i], activation=activation))
		model.add(keras.layers.Dense(self.n_action, activation='linear'))
		model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
		self.model = model
		self.model_name = self.qmodel + str(n_hidden) + str(dense_units)
		


class QModelLSTM(QModelRNN):
	def init(self):
		self.qmodel = 'LSTM'
	def build_model(self, n_hidden, dense_units, learning_rate, activation='relu'):
		Layer = keras.layers.LSTM
		self._build_model(Layer, n_hidden, dense_units, learning_rate, activation)


class QModelGRU(QModelRNN):
	def init(self):
		self.qmodel = 'GRU'
	def build_model(self, n_hidden, dense_units, learning_rate, activation='relu'):
		Layer = keras.layers.GRU
		self._build_model(Layer, n_hidden, dense_units, learning_rate, activation)



class QModelConv(QModelKeras):
	"""
	ref: https://keras.io/layers/convolutional/
	"""
	def init(self):
		self.qmodel = 'Conv'

	def build_model(self, 
		filter_num, filter_size, dense_units, 
		learning_rate, activation='relu', dilation=None, use_pool=None):

		if use_pool is None:
			use_pool = [True]*len(filter_num)
		if dilation is None:
			dilation = [1]*len(filter_num)

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
			])+' + '+str(dense_units)

		

class QModelConvRNN(QModelKeras):
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
				return_sequences=(i<m-1)))
		for i in range(len(dense_units)):
			model.add(keras.layers.Dense(dense_units[i], activation=activation))

		model.add(keras.layers.Dense(self.n_action, activation='linear'))
		model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
		self.model = model
		self.model_name = self.qmodel + str(conv_n_hidden) + str(RNN_n_hidden) + str(dense_units)
		

class QModelConvLSTM(QModelConvRNN):
	def init(self):
		self.qmodel = 'ConvLSTM'
	def build_model(self, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate, 
		conv_kernel_size=3, use_pool=False, activation='relu'):
		Layer = keras.layers.LSTM
		self._build_model(Layer, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate, 
		conv_kernel_size, use_pool, activation)


class QModelConvGRU(QModelConvRNN):
	def init(self):
		self.qmodel = 'ConvGRU'
	def build_model(self, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate, 
		conv_kernel_size=3, use_pool=False, activation='relu'):
		Layer = keras.layers.GRU
		self._build_model(Layer, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate, 
		conv_kernel_size, use_pool, activation)







def load_model(fld, learning_rate):
	s = open(os.path.join(fld,'QModel.txt'),'r').read().strip()
	qmodels = {
		'Conv':QModelConv,
		'DenseOnly':QModelMLP,
		'MLP':QModelMLP,
		'LSTM':QModelLSTM,
		'GRU':QModelGRU,
		}
	qmodel = qmodels[s](None, None)
	qmodel.load(fld, learning_rate)
	return qmodel


