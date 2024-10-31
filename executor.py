# This file implements the executors used in the paper. Reference: https://github.com/lorangpi/HyGOAL/blob/main/executor.py

from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed

set_random_seed(0, using_cuda=True)

def load_policy(env, path, lr=0.0003, log_dir=None, seed=0):
    # Load the model
    set_random_seed(seed, using_cuda=True)
    model = SAC.load(path, env=env, learning_rate=lr, tensorboard_log=log_dir, seed=seed)
    return model

class Executor():
	def __init__(self, id, mode, Initiation=None, Termination=None):
		super().__init__()
		self.id = id
		self.mode = mode
		self.Initiation = Initiation
		self.Termination = Termination
		self.policy = None

	def path_to_json(self):
		return {self.id:self.policy}

class Executor_RL(Executor):
	def __init__(self, id, alg, policy, Initiation, Termination):
		super().__init__(id, "RL", Initiation, Termination)
		self.alg = alg
		self.policy = policy
		self.model = None

	def execute(self, env, operator, render, obs):
		'''
		This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
		produced by the policy on that state. 
		'''
		print("Loading policy {}".format(self.policy))
		if self.model is None:
			self.model = load_policy(self.alg, env, self.policy, seed=0)
		step_executor = 0
		done = False
		success = False
		while not done:
			action, _states = self.model.predict(obs)
			try: 
				obs, reward, terminated, truncated, info = env.step(action)
				done = terminated or truncated
			except:
				obs, reward, done, info = env.step(action)
			step_executor += 1
			success = self.Termination(operator=operator, env=env)
			if step_executor > 500:
				done = True
			if render:
				env.render()
		return obs, success