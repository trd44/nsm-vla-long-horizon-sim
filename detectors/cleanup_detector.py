import itertools
import numpy as np
from typing import *
from scipy.spatial.transform import Rotation as R
from robosuite.models.base import MujocoModel
from robosuite.models.grippers import GripperModel
from robosuite.utils.mjcf_utils import find_elements

class Cleanup_Detector:
    def __init__(self, env, return_int=False):
        self.env = env
        self.obs = self.env.viewer._get_observations() if self.env.viewer_get_obs else self.env._get_observations() # detect objects' state using the observation
        self.return_int = return_int
        # predicate mappings
        self.predicates = {
            'small-enough-for-gripper-to-pick-up': {
                'func':self.small_enough_for_gripper_to_pick_up,
                'params':['tabletop_object']
                }, 
            'directly-on-table': {
                'func':self.directly_on_table,
                'params':['tabletop_object', 'table']
            }, 
            'large-enough-for-gripper-to-reach-inside': {
                'func':self.large_enough_for_gripper_to_reach_inside,
                'params':['tabletop_object']
            }, 
            'inside': {
                'func':self.inside,
                'params':['tabletop_object', 'container']
            }, 
            'open': {
                'func':self.open,
                'params':['container']
            }, 
            'free': {
                'func':self.free,
                'params':['gripper']
            }, 
            'small-enough-to-fit-in-container': {
                'func':self.small_enough_to_fit_in_container,
                'params':['tabletop_object', 'container']
            }
        }
        # mapping from relevant object types to objects in the environment
        self.object_types = {'tabletop_object':['block', 'mug', 'drawer'], 'container':['drawer', 'mug'], 'gripper':['gripper'], 'mug':['mug'], 'drawer':['drawer'], 'table':['table']}
        self.grounded_objects = {'mug':['mug1'], 'block':['block1'], 'drawer':['drawer1'], 'table':['table1'], 'gripper':['gripper1']}
    
    def update_obs(self, obs=None):
        """update the observation

        Args:
            obs (OrderedDict, optional): the observation returned by `env.step(...)`. Defaults to None.
        """
        if obs is not None:
            self.obs = obs
        else:
            self.obs = self.env.viewer._get_observations() if self.env.viewer_get_obs else self.env._get_observations()
    
    def r_int(self, value):
        return int(value) if self.return_int else value
    
    def small_enough_for_gripper_to_pick_up(self, tabletop_obj:str) -> bool:
        """
        Returns True if the object is small enough for the gripper to pick up.
        """
        #hardcoding mug and pod to be small enough to pick up
        if tabletop_obj == 'mug' or tabletop_obj == 'block':
            return True
        return False

    def directly_on_table(self, tabletop_obj:str, table='table') -> bool:
        """Returns True if the object is directly on the table.

        Args:
            tabletop_obj (str): the tabletop object

        Returns:
            bool: True if the object is directly on the table
        """
        assert tabletop_obj in self.object_types['tabletop_object']
        return self.env.check_directly_on_table(tabletop_obj)

    def large_enough_for_gripper_to_reach_inside(self, container:str) -> bool:
        """Returns True if the container is large enough for the gripper to reach inside.

        Args:
            container (str): the container object

        Returns:
            bool: True if the object is large enough for the gripper to reach inside
        """
        # hardcoding the drawer to be large enough for the gripper to reach inside
        if container == 'drawer':
            return True
        return False

    def occupying_gripper(self, tabletop_obj:str, gripper='gripper') -> bool:
        """Returns True if the object is exclusively occupying the gripper.

        Args:
            tabletop_obj (str): the tabletop object
            gripper (str): the gripper object

        Returns:
            bool: True if the object is exclusively occupying the gripper
        """
        assert tabletop_obj in self.object_types['tabletop_object'] and gripper in self.object_types['gripper']
        gripper = self.env.robots[0].gripper
        tabletop_obj_contact_geoms = getattr(self.env, tabletop_obj).contact_geoms
        return self.env._check_grasp(gripper, tabletop_obj_contact_geoms)
    
    def inside(self, tabletop_obj:str, container:str) -> bool:
        """Returns True if the object is inside the container.

        Args:
            tabletop_obj (str): the tabletop object
            container (str): the container object

        Returns:
            bool: True if the object is inside the container
        """
        assert tabletop_obj in self.object_types['tabletop_object'] and container in self.object_types['container']
        if container == 'drawer':
            if tabletop_obj in ('mug', 'block'):
                return self.env.check_in_drawer(tabletop_obj)
            else:
                return False # only mugs and blocks can be inside the drawer
        elif container == 'mug':
            if tabletop_obj == 'block':
                return self.env.check_in_mug(tabletop_obj)
            else:
                return False # only blocks can be inside the mug
        else:  # only drawers and mugs can contain objects
            return False


    def open(self, container:str) -> bool:
        """Returns True if the container is open.

        Args:
            container (_type_): the container object

        Returns:
            bool: True if the container is open
        """
        assert container in self.object_types['container']
        if container == 'drawer':
            return self.env.check_drawer_open()
        elif container == 'mug':
            return True # mugs are always open

    def free(self, gripper) -> bool:
        """Returns True if the gripper is free.

        Args:
            gripper (_type_): the gripper object

        Returns:
            bool: True if the gripper is free
        """
        for obj in self.object_types['tabletop_object']:
            if self.occupying_gripper(obj, gripper):
                return False
        return True
    
    def small_enough_to_fit_in_container(self, tabletop_obj:str, container:str) -> bool:
        """Returns True if the object is small enough to fit in the container.

        Args:
            tabletop_obj (str): the tabletop object
            container (str): the container object

        Returns:
            bool: True if the object is small enough to fit in the container
        """
        assert tabletop_obj in self.object_types['tabletop_object'] and container in self.object_types['container']
        if container == 'drawer':
            if tabletop_obj in ('mug', 'block'):
                return True
        elif container == 'mug':
            if tabletop_obj == 'block':
                return True
        return False

    def get_groundings(self, as_dict=False, binary_to_float=False) -> dict:
        """Returns the groundings for the coffee detector.

        Args:
            as_dict (bool, optional): whether to return the groundings as a dictionary. Defaults to False.
            binary_to_float (bool, optional): whether to convert binary values to float. Defaults to False.

        Returns:
            dict: the groundings for the coffee detector
        """
        groundings = {}
        for predicate_name, predicate in self.predicates.items():
            groundings[predicate_name] = {}
            param_list = []
            for param_type in predicate['params']:
                param_list.append(self.object_types[param_type])
            param_combinations = list(itertools.product(*param_list))
            callable_func = predicate['func']
            for comb in param_combinations:
                truth_value = callable_func(*comb)
                predicate_str = f'{predicate_name}({",".join([self.grounded_objects[param] for param in comb])})'
                groundings[predicate_str] = truth_value
        return groundings
                
        