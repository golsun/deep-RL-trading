from lib import *



class Simulator:

	def play_one_episode(self, exploration, training=True, rand_price=True, print_t=False):

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

		while not done:
			if print_t:
				print(self.env.t)
    

			action = self.agent.act(state, exploration, valid_actions)
			next_state, reward, done, valid_actions = self.env.step(action)

			cum_rewards.append(prev_cum_rewards+reward)
			prev_cum_rewards = cum_rewards[-1]
			actions.append(action)
			states.append(next_state)

			if training:
				self.agent.remember(state, action, reward, next_state, done, valid_actions)
				self.agent.replay()

			state = next_state

		return cum_rewards, actions, states


	def train(self, n_episode, 
		save_per_episode=10, exploration_decay=0.995, exploration_min=0.01, print_t=False, exploration_init=1.):

		fld_model = os.path.join(self.fld_save,'model')
		makedirs(fld_model)	# don't overwrite if already exists
		with open(os.path.join(fld_model,'QModel.txt'),'w') as f:
			f.write(self.agent.model.qmodel)

		exploration = exploration_init
		fld_save = os.path.join(self.fld_save,'training')

		makedirs(fld_save)
		MA_window = 100		# MA of performance
		safe_total_rewards = []
		explored_total_rewards = []
		explorations = []
		path_record = os.path.join(fld_save,'record.csv')

		with open(path_record,'w') as f:
			f.write('episode,game,exploration,explored,safe,MA_explored,MA_safe\n')

		for n in range(n_episode):

			print('\ntraining...')
			exploration = max(exploration_min, exploration * exploration_decay)
			explorations.append(exploration)
			explored_cum_rewards, explored_actions, _ = self.play_one_episode(exploration, print_t=print_t)
			explored_total_rewards.append(100.*explored_cum_rewards[-1]/self.env.max_profit)
			safe_cum_rewards, safe_actions, _ = self.play_one_episode(0, training=False, rand_price=False, print_t=False)
			safe_total_rewards.append(100.*safe_cum_rewards[-1]/self.env.max_profit)

			MA_total_rewards = np.median(explored_total_rewards[-MA_window:])
			MA_safe_total_rewards = np.median(safe_total_rewards[-MA_window:])

			ss = [
				str(n), self.env.title.replace(',',';'), '%.1f'%(exploration*100.), 
				'%.1f'%(explored_total_rewards[-1]), '%.1f'%(safe_total_rewards[-1]),
				'%.1f'%MA_total_rewards, '%.1f'%MA_safe_total_rewards,
				]
			
			with open(path_record,'a') as f:
				f.write(','.join(ss)+'\n')
				print('\t'.join(ss))

			
			if n%save_per_episode == 0:
				print('saving results...')
				self.agent.save(fld_model)

				"""
				self.visualizer.plot_a_episode(
					self.env, self.agent.model, 
					explored_cum_rewards, explored_actions,
					safe_cum_rewards, safe_actions,
					os.path.join(fld_save, 'episode_%i.png'%(n)))

				self.visualizer.plot_episodes(
					explored_total_rewards, safe_total_rewards, explorations, 
					os.path.join(fld_save, 'total_rewards.png'),
					MA_window)
					"""




	def test(self, n_episode, save_per_episode=10, subfld='testing'):

		fld_save = os.path.join(self.fld_save, subfld)
		makedirs(fld_save)
		MA_window = 100		# MA of performance
		safe_total_rewards = []
		path_record = os.path.join(fld_save,'record.csv')

		with open(path_record,'w') as f:
			f.write('episode,game,pnl,rel,MA\n')

		for n in range(n_episode):
			print('\ntesting...')
			
			safe_cum_rewards, safe_actions, _ = self.play_one_episode(0, training=False, rand_price=True)
			safe_total_rewards.append(100.*safe_cum_rewards[-1]/self.env.max_profit)
			MA_safe_total_rewards = np.median(safe_total_rewards[-MA_window:])
			ss = [str(n), self.env.title.replace(',',';'), 
				'%.1f'%(safe_cum_rewards[-1]),
				'%.1f'%(safe_total_rewards[-1]), 
				'%.1f'%MA_safe_total_rewards]
			
			with open(path_record,'a') as f:
				f.write(','.join(ss)+'\n')
				print('\t'.join(ss))

			
			if n%save_per_episode == 0:
				print('saving results...')

				"""
				self.visualizer.plot_a_episode(
					self.env, self.agent.model, 
					[np.nan]*len(safe_cum_rewards), [np.nan]*len(safe_actions),
					safe_cum_rewards, safe_actions,
					os.path.join(fld_save, 'episode_%i.png'%(n)))

				self.visualizer.plot_episodes(
					None, safe_total_rewards, None, 
					os.path.join(fld_save, 'total_rewards.png'),
					MA_window)
					"""
					



	def __init__(self, agent, env, 
		visualizer, fld_save):

		self.agent = agent
		self.env = env
		self.visualizer = visualizer
		self.fld_save = fld_save





if __name__ == '__main__':
	#print 'episode%i, init%i'%(1,2)
	a = [1,2,3]
	print(np.mean(a[-100:]))