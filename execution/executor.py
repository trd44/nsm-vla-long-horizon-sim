# This file implements the executors used in the paper. Reference: https://github.com/lorangpi/HyGOAL/blob/main/executor.py
from detection.detector import Detector
from tarski import fstrips as fs
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from planning.planning_utils import unpickle_goal_node

set_random_seed(0, using_cuda=True)

def load_policy(env, path, lr=0.0003, log_dir=None, seed=0):
    # Load the model
    set_random_seed(seed, using_cuda=True)
    model = SAC.load(path, env=env, learning_rate=lr, tensorboard_log=log_dir, seed=seed)
    return model

class Executor():
	def __init__(self, mode, operator:fs.Action, policy):
		super().__init__()
		self.id = self.operator.name
		self.mode = mode
		self.operator = operator
		self.policy = policy

	def path_to_json(self):
		return {self.id:self.policy}

class Executor_RL(Executor):
	def __init__(self, alg:str, operator:fs.Action, policy):
		super().__init__("RL", operator, policy)
		self.alg = alg
		self.model = None

	def execute(self, detector:Detector, render=False):
		'''
		This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
		produced by the policy on that state. 
		'''
		print("Loading policy {}".format(self.policy))
		env = detector.get_env()
		obs = detector.get_obs()
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
	
	def check_precondition(self, detector:Detector) -> bool:
		"""check that the precondition of the operator holds in the current state

		Args:
			detector (Detector): the detector
		Returns;
			bool: True if the precondition holds in the current state
		"""
		groundings = detector.get_groundings()
		precondition = self.operator.precondition
		unsatisfied_conditions = []
		for precond in precondition.subformulas: # assume precondition is a conjunction of literals
			# account for the fact that some preconditions are negated e.g. `not(free(gripper1))`
			# negated formula has the form `(not ...)` e.g. `(not free(gripper1))`
			if precond.is_negation:
				precond = precond.subformulas[0]
				if precond not in groundings or not groundings[precond]:
					continue
		return True
	
if __name__	== "__main__":
	# testing the executor
	plans = unpickle_goal_node('planning/PDDL/coffee/goal_node_1.pkl')
	print(plans)
