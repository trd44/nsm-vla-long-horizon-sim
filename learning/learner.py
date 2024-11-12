import os
import detection.detector
import execution.executor
import gymnasium as gym
import importlib
from tarski import fstrips as fs
from robosuite.robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from typing import *
from reward_functions.rewardFunctionPrompts import *
from VLM.LlmApi import chat_completion

class OperatorWrapper(gym.Wrapper):
    def __init__(self, env, detector:detection.detector.Detector, grounded_operator:fs.Action, executed_operators:Dict[fs.Action:execution.executor.Executor], config:dict):
        super().__init__(env)
        self.detector = detector
        self.grounded_operator = grounded_operator
        self.executed_operators:Dict[fs.Action:execution.executor.Executor] = executed_operators
        self.config = config

    def step(self, action):
        try:
            obs, reward, done, truncated, info = self.env.step(action)
        except:
            obs, reward, done, info = self.env.step(action)
        truncated = truncated or self.env.done
        self.detector.update_obs()
        obs:dict = self.detector.get_obs()
        binary_states_with_semantics:dict = self.detector.get_groundings()
        # combine obs with binary_states_with_semantics
        obs.update(binary_states_with_semantics)
        reward = self.compute_reward(obs)
        #TODO: may need to turn obs into a numpy array
        return obs, reward, done, truncated, info

    def reset(self):
        reset_success = False
        while not reset_success:
            # first, reset the environment to the very beginning
            try:
                obs, info = self.env.reset(seed=self.config['learning']['seed'])
            except:
                obs = self.env.reset(seed=self.config['learning']['seed'])
                info = {}
            # second, execute the executors that should be executed before the operator to learn
            reset_success = True
            for op, ex in self.executed_operators.items():
                ex_success = ex.execute(self.detector, op)
                if not ex_success:
                    reset_success = False
                    break

        self.detector.update_obs()
        obs = self.detector.get_obs()
        return obs, info

    def check_effect_satisfied(effect:fs.SingleEffect, binary_obs:dict) -> bool:
        """check if the effect is satisfied in the observation

        Args:
            effect (fs.SingleEffect): the effect to check
            binary_obs (dict): the binary observation with semantics

        Returns:
            bool: True if the effect is satisfied, False otherwise
        """
        effect_name = effect.atom.pddl_repr()# e.g. `free gripper1``
        # check if effect is negated
        if isinstance(effect, fs.DelEffect): # effect is negated e.g., `not (free gripper1)`
            return not binary_obs[effect_name]
        return binary_obs[effect_name]
    
    def check_duplicate_grasp_effects(self) -> bool:
        """check if the grounded operator has both `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` effects. If so, they should count as one effect

        Returns:
            bool: True if both effects are present, False otherwise
        """
        effects:List[fs.SingleEffect] = self.grounded_operator.effects
        # check if `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` are both in the effects. If so, they should count as one effect
        not_free_gripper_effect_present = False
        exclusively_occupying_gripper_effect_present = False
        for effect in effects:
            if effect.atom.pddl_repr() == 'free gripper1' and isinstance(effect, fs.DelEffect): # `not (free gripper1)` is present
                not_free_gripper_effect_present = True
            elif effect.atom.predicate.name == 'exclusively-occupying-gripper':
                exclusively_occupying_gripper_effect_present = True
        return not_free_gripper_effect_present and exclusively_occupying_gripper_effect_present

    def compute_reward(self, obs:dict) -> float:
        """compute the reward by calling a LLM generated reward function on an observation with semantics

        Args:
            obs (dict): the observation in which the keys have semantics and the values are arrays of numeric values

        Returns:
            float: the reward between -1, 0
        """
        # import the llm generated sub-goal reward shaping function
        llm_reward_func_module = importlib.import_module(f'learning.reward_functions.{self.config['planning']['domain']}.{self.grounded_operator.name}')

        llm_reward_shaping_func = getattr(llm_reward_func_module, 'reward_shaping')

        # get the binary detector observation whose keys are predicates and values are True/False
        binary_obs:dict = self.detector.get_groundings()
        
        # there is a step cost of -1 regardless
        step_cost = -1
        effects:List[fs.SingleEffect] = self.grounded_operator.effects # the effects have been ordered by the LLM
        # check if `not (free gripper1)` and `exclusively-occupying-gripper ?object gripper1` are both in the effects. If so, they should count as one effect

        duplicate_grasp_effects = self.check_duplicate_grasp_effects()
        num_effects = len(effects) if not duplicate_grasp_effects else len(effects) - 1
         
        sub_goal_reward = 0
        for effect in effects:
            if effect.pddl_repr() == 'not (free gripper1)' and duplicate_grasp_effects: # if the effect is `not (free gripper1)`, skip it since it is the same effect as `exclusively-occupying-gripper ?object gripper1`
                continue
            # in addition to the step cost, the robot gets a reward in the range of [0, 1]. The reward is given based on sub-goals achieved. Each effect of the operator is a sub-goal. Therefore, the robot would get `1/len(effects)` reward for each effect achieved.
            if self.check_effect_satisfied(effect, binary_obs):
                sub_goal_reward += 1/num_effects
            else:
                sub_goal_reward += llm_reward_shaping_func(effect, obs) * 1/num_effects
                return step_cost + sub_goal_reward # return the reward as soon as one effect is not satisfied. Assume later effects are at 0% progress therefore would get a shaping reward of 0 anyway.
        
        return step_cost + sub_goal_reward

        


class Learner:
    def __init__(self, env, domain:str, detector:detection.detector.Detector, grounded_operator_to_learn:fs.Action, executed_operators:Dict[fs.Action:execution.executor.Executor], config:dict):
        self.config = config
        self.detector = detector
        self.domain = domain
        self.env = self._wrap_env(env)
        self.executed_operators = executed_operators
        self.grounded_operator = grounded_operator_to_learn
        self._llm_order_effects()
        self._llm_sub_goal_reward_shaping()

    def learn(self) -> execution.executor.Executor_RL:
        """Train an RL agent to learn the operator.

        Returns:
            execution.executor.Executor_RL: an RL executor for the operator and executes the policy for the operator when called
        """
        # TODO: add a customeval callback and pass it into `model.save`
        model = SAC(
            "MlpPolicy",
            env = self.env,
            tensorboard_log=f'learning/policies/{self.domain}/{self.grounded_operator.name}/seed_{self.config["seed"]}/tensorboard_logs',
            **self.config['learning']
        )
        model.learn(
            total_timesteps=self.config['timesteps'])
        model.save(f'learning/policies/{self.domain}/{self.grounded_operator.name}/seed_{self.config["seed"]}/model')
        # TODO: create an Executor_RL object associated with the newly learned policy. Pickle the Executor_RL object and save it to a file. Return the Executor_RL object
    
    def _llm_order_effects(self):
        """Prompt the LLM to order the effects of the grounded operator in terms of which effects are expected to be achieved before others.
        """
        #TODO: dynamically fill in the prompt with operator specific information such as the operator's name and effects
        out = chat_completion(order_effects_prompt)
        #TODO: reorder the effects based on the LLM's response
    
    def _llm_sub_goal_reward_shaping(self):
        """Prompt the LLM to write a sub-goal reward shaping function that takes in an effect (sub-goal) and the observation with semantics and returns a reward depending on the progress towards achieving the effect.
        """
        #TODO: dynamically fill in the prompt with operator specific information such as the operator's name and effects
        out = chat_completion(reward_shaping_prompt)
        #TODO: save the output python function to a file in the reward_functions directory
        with open(f'learning/reward_functions/{self.domain}/{self.config['planning']['domain']}.{self.grounded_operator.name}.py', 'w') as f:
            f.write(out)
    

    def _wrap_env(self, env) -> gym.Wrapper:
        """Wrap the environment in a GymWrapper.

        Args:
            env (gym environment): the environment to wrap

        Returns:
            gym.Wrapper: the wrapped environment
        """
        env = GymWrapper(env)
        env = OperatorWrapper(env, self.detector, self.grounded_operator, self.executed_operators, self.config)
        env = Monitor(env, f'learning/policies/{self.domain}/{self.grounded_operator.name}/seed_{self.config["seed"]}/monitor_logs', allow_early_resets=True)
        return env
    
