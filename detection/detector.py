import itertools
from typing import *

class Detector:
    def __init__(self, env, return_int=False):
        self.env = env
        self.obs = self.env.viewer._get_observations() if self.env.viewer_get_obs else self.env._get_observations() # detect objects' state using the observation
        self.return_int = return_int

        self.grounded_object_to_pddl_object = {}

    def get_env(self):
        return self.env
    
    def get_obs(self):
        return self.obs

    def update_obs(self, obs=None):
        """update the observation

        Args:
            obs (OrderedDict, optional): the observation returned by `env.step(...)`. Defaults to None.
        """
        if obs is not None:
            self.obs = obs
        else:
            self.obs = self.env.viewer._get_observations() if self.env.viewer_get_obs else self.env._get_observations()
    
    def get_groundings(self) -> dict:
        """Returns the groundings for the coffee detector.

        Returns:
            dict: the groundings for the coffee detector
        """
        groundings = {}
        for predicate_name, predicate in self.predicates.items():
            groundings[predicate_name] = {}
            param_list = []
            # e.g. for predicate_name = 'inside', predicate['params'] = ['tabletop_object', 'container']
            for param_type in predicate['params']:
                # e.g. for predicate_name = 'inside', param_list = [['coffee_pod', 'coffee_machine_lid', 'coffee_pod_holder', 'mug', 'drawer'], ['coffee_pod_holder', 'drawer', 'mug']]
                param_list.append(self.object_types[param_type])
            # e.g param_combinations = [('coffee_pod', 'coffee_pod_holder'), ('coffee_pod', 'drawer'), ('coffee_pod', 'mug'), ('coffee_machine_lid', 'coffee_pod_holder'), ('coffee_machine_lid', 'drawer'), ('coffee_machine_lid', 'mug'), ('coffee_pod_holder', 'coffee_pod_holder'), ('coffee_pod_holder', 'drawer'), ('coffee_pod_holder', 'mug'), ('mug', 'coffee_pod_holder'), ('mug', 'drawer'), ('mug', 'mug'), ('drawer', 'coffee_pod_holder'), ('drawer', 'drawer'), ('drawer', 'mug')]
            param_combinations = list(itertools.product(*param_list))
            callable_func = predicate['func']
            for comb in param_combinations:
                truth_value = callable_func(*comb)
                predicate_str = f'{predicate_name}({",".join(self._to_pddl_format(comb))})'
                groundings[predicate_str] = truth_value
        return groundings
    
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
                
        
    
    