# This file implements the executors used in the paper. Reference: https://github.com/lorangpi/HyGOAL/blob/main/executor.py
import logging
from detection.detector import Detector
from tarski import fstrips as fs
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from planning.planning_utils import *
from utils import *

set_random_seed(0, using_cuda=True)

def load_policy(env, path, lr=0.0003, log_dir=None, seed=0):
    # Load the model
    set_random_seed(seed, using_cuda=True)
    model = SAC.load(path, env=env, learning_rate=lr, tensorboard_log=log_dir, seed=seed)
    return model

class Executor():
	def __init__(self, mode, operator_name:str, policy):
		self.name = operator_name
		self.mode = mode
		self.policy = policy

	def path_to_json(self):
		return {self.name:self.policy}
	
	def execute(self):
		"""Execute the operator

		Raises:
			NotImplementedError: This method should be implemented by the subclass
		"""
		raise NotImplementedError

class Executor_RL(Executor):
	def __init__(self, operator_name:str, alg:str, policy):
		super().__init__("RL", operator_name=operator_name, policy=policy)
		self.alg = alg
		self.model = None

	def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
		'''
		This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
		produced by the policy on that state. 
		'''
		print("Loading policy {}".format(self.policy))
		# check that the grounded operator is the same as the operator to be executed
		grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
		assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
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
			success, _ = self.check_effects(detector)
			if step_executor > 500:
				done = True
			if render:
				env.render()
		return obs, success
	
	def check_precondition(self, detector:Detector) -> Tuple[bool, Set[str]]:
		"""check that the precondition of the operator holds in the current state

		Args:
			detector (Detector): the detector
		Returns;
			Tuple[bool, Set[str]]: a tuple of a boolean value indicating whether the precondition holds and a set of unsatisfied preconditions
		"""
		groundings = detector.get_groundings()
		precondition = self.grounded_operator.precondition
		unsatisfied_conditions = set()
		for precond in precondition.subformulas: # assume precondition is a conjunction of literals
			# account for the fact that some preconditions are negated e.g. `not(free(gripper1))`
			# negated formula has the form `(not ...)` e.g. `(not free(gripper1))`
			if precond.connective.name == 'Not':
				literal:str = precond.pddl_repr().replace('not ', '')
				literal = split_by_parentheses(literal)[0] # remove the parentheses
				if groundings[literal]: # if the literal is true in groundings, then the precondition is not satisfied
					unsatisfied_conditions.add(f'not {literal}')
			else: # positive literal
				literal:str = precond.pddl_repr()
				literal = split_by_parentheses(literal)[0]
				if not groundings[literal]: # if the literal is false in groundings, then the precondition is not satisfied
					unsatisfied_conditions.add(literal)
		if len(unsatisfied_conditions) > 0:
			logging.warning(f"Unsatisfied preconditions: {unsatisfied_conditions}")
			return False, unsatisfied_conditions
		return True, []
	
	def check_effects(self, detector:Detector) -> Tuple[bool, Set[str]]:
		"""check that the effects of the operator hold in the current state

		Args:
			detector (Detector): the detector
		Returns;
			Tuple[bool, Set[str]]: a tuple of a boolean value indicating whether the effects hold and a set of unintended effects
		"""
		effects:set = set(effect.pddl_repr() for effect in self.grounded_operator.effects)
		_, unsatisfied_preconditions = self.check_precondition(detector)
		# check whether there are elements in unsatisfied_preconditions that are not in effects
		unintended_effects = unsatisfied_preconditions - effects
		if len(unintended_effects) > 0:
			logging.info(f"Unintended effects: {unintended_effects}")
			return False, list(unintended_effects)
		# There are no unintended effects. Check if all intended effects are satisfied
		groundings = detector.get_groundings()
		unsatisfied_effects = set()
		for effect in effects:
			# check if effect is negated
			if isinstance(effect, fs.DelEffect): # effect is negated
				literal:str = effect.pddl_repr().replace('not ', '')
				literal = split_by_parentheses(literal)[0] # remove the parentheses
				if groundings[literal]: # if the literal is true in groundings, then the precondition is not satisfied
					unsatisfied_effects.add(f'not {literal}')
			else: # positive literal
				literal:str = effect.pddl_repr()
				literal = split_by_parentheses(literal)[0]
				if not groundings[literal]: # if the literal is false in groundings, then the precondition is not satisfied
					unsatisfied_effects.add(literal)
		if len(unsatisfied_effects) > 0:
			logging.warning(f"Unsatisfied effects: {unsatisfied_effects}")
			return False, unsatisfied_effects
		return True, []
	
if __name__	== "__main__":
	# testing the executor
	plan = reverse_engineer_plan(unpickle_goal_node('planning/PDDL/coffee/goal_node_1.pkl'))
	print(plan)
