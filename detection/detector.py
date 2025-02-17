import itertools
from typing import *

class Detector:
    def __init__(self, env, return_int=False):
        if hasattr(env, 'unwrapped'): # make sure that the environment is unwrapped
            self.env = env.unwrapped
        else:
            self.env = env
        self.obs = self.env.viewer._get_observations() if self.env.viewer_get_obs else self.env._get_observations() # detect objects' state using the observation
        self.return_int = return_int

        self.grounded_object_to_pddl_object = {}

    def get_env(self):
        return self.env
    
    def get_obs(self):
        self.update_obs()
        return self.obs
    
    def set_env(self, env):
        if hasattr(env, 'unwrapped'): # make sure that the environment is unwrapped
            self.env = env.unwrapped
        else:
            self.env = env

    def update_obs(self, obs=None):
        """update the observation

        Args:
            obs (OrderedDict, optional): the observation returned by `env.step(...)`. Defaults to None.
        """
        if obs is not None:
            self.obs = obs
        else:
            self.obs = self.env.viewer._get_observations() if self.env.viewer_get_obs else self.env._get_observations()
    
    def verify_env(self, env) -> bool:
        """Verify that the environment is the correct environment class
        Args:
            env (MujocoEnv): the environment
        Returns:
            bool: True if the environment is correct
        """
        # must be implemented by the subclass
        raise NotImplementedError("The verify_env method must be implemented in the subclass")
    
    def detect_binary_states(self, env) -> dict:
        """Returns the groundings for the coffee detector.
        Args:
            env (MujocoEnv): the environment
        Returns:
            dict: the groundings for the detector
        """
        self.verify_env(env)
        self.set_env(env)
        obs = {}
        for predicate_name, predicate in self.predicates.items():
            param_list = []
            # e.g. for predicate_name = 'inside', predicate['params'] = ['tabletop_object', 'container']
            for param_type in predicate['params']:
                # e.g. for predicate_name = 'inside', param_list = [['coffee_pod', 'coffee_machine_lid', 'coffee_pod_holder', 'mug', 'drawer'], ['coffee_pod_holder', 'drawer', 'mug']]
                param_list.append(self.object_types[param_type])
            # e.g param_combinations = [('coffee_pod', 'coffee_pod_holder'), ('coffee_pod', 'drawer'), ('coffee_pod', 'mug'), ('coffee_machine_lid', 'coffee_pod_holder'), ('coffee_machine_lid', 'drawer'), ('coffee_machine_lid', 'mug'), ('coffee_pod_holder', 'coffee_pod_holder'), ('coffee_pod_holder', 'drawer'), ('coffee_pod_holder', 'mug'), ('mug', 'coffee_pod_holder'), ('mug', 'drawer'), ('mug', 'mug'), ('drawer', 'coffee_pod_holder'), ('drawer', 'drawer'), ('drawer', 'mug')]
            param_combinations = list(itertools.product(*param_list))
            callable_func = predicate['func']
            for comb in param_combinations:
                # skip if the same object is used twice
                if len(set(comb)) < len(comb):
                    continue
                truth_value = callable_func(*comb)
                predicate_str = f'{predicate_name} {" ".join(self._to_pddl_format(comb))}'
                obs[predicate_str] = truth_value
        return obs
    
    def r_int(self, value):
        return int(value) if self.return_int else value
    
    def _is_type(self, obj, obj_type):
        """Returns True if the object is of the specified type.

        Args:
            obj (str): the object
            obj_type (str): the object type

        Returns:
            bool: True if the object is of the specified type
        """
        return obj in self.object_types[obj_type]
    
    def _to_pddl_format(self, objs):
        """Converts the grounded object to their pddl format.

        Args:
            objs (List[str]): the grounded objects

        Returns:
            List[str]: the pddl objects
        """
        return [self.grounded_object_to_pddl_object.get(obj) for obj in objs]
                
        
    
    