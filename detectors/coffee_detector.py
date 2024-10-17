import itertools
import numpy as np
from typing import *
from scipy.spatial.transform import Rotation as R
from robosuite.models.base import MujocoModel
from robosuite.models.grippers import GripperModel
from robosuite.utils.mjcf_utils import find_elements

class Coffee_Detector:
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
            'can-flip-up': {
                'func': self.can_flip_up,
                'params':['coffee_machine_lid']
             },
            'can-flip-down': {
                'func': self.can_flip_down,
                'params':['coffee-machine-lid']
            },
            'directly-on-table': {
                'func':self.directly_on_table,
                'params':['tabletop_object', 'table']
            }, 
            'exclusively-occupying-gripper': {
                'func':self.exclusively_occupying_gripper,
                'params':['tabletop_object', 'gripper']
            }, 
            'attached': {
                'func':self.attached,
                'params':['coffee_machine_lid', 'coffee_pod_holder']
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
            'under': {
                'func':self.under,
                'params':['mug', 'coffee_pod_holder']
            }
        }
        # mapping from relevant object types to objects in the environment
        self.object_types = {'tabletop_object':['coffee_pod', 'coffee_machine_lid', 'coffee_pod_holder', 'mug', 'drawer'], 'container':['coffee_pod_holder', 'drawer', 'mug'], 'gripper':['gripper'], 'mug':['mug'], 'coffee_machine_lid':['coffee_machine_lid'], 'coffee_pod_holder':['coffee_pod_holder'], 'drawer':['drawer'], 'coffee_pod':['coffee_pod'], 'table':['table']}
        self.grounded_objects = {'mug':['mug1'], 'coffee_pod':['coffee_pod1'], 'coffee_machine_lid':['coffee_machine_lid1'], 'coffee_pod_holder':['coffee_pod_holder1'], 'drawer':['drawer1'], 'table':['table1'], 'gripper':['gripper1']}
    
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
        if tabletop_obj == 'mug' or tabletop_obj == 'coffee_pod':
            return True
        return False
    
    def can_flip_up(self, coffee_machine_lid:str) -> bool:
        """Returns True if the coffee pod lid can be flipped up.

        Args:
            coffee_machine_lid (str: the coffee machine lid object

        Returns:
            bool: True if the coffee pod lid can be flipped up
        """
        assert coffee_machine_lid == 'coffee_machine_lid'
        return self.env.check_can_flip_up_lid()
        

    def can_flip_down(self, coffee_machine_lid:str) -> bool:
        """Returns True if the coffee pod lid can be flipped down.

        Args:
            coffee_machine_lid (str): the coffee pod lid object

        Returns:
            bool: True if the coffee pod lid can be flipped down
        """
        assert coffee_machine_lid == 'coffee_machine_lid'
        return not self.can_flip_up(coffee_machine_lid)

    def directly_on_table(self, tabletop_obj:str, table='table') -> bool:
        """Returns True if the object is directly on the table.

        Args:
            tabletop_obj (str): the tabletop object

        Returns:
            bool: True if the object is directly on the table
        """
        assert tabletop_obj in self.object_types['tabletop_object']
        return self.env.check_directly_on_table(tabletop_obj)


    def exclusively_occupying_gripper(self, tabletop_obj:str, gripper='gripper') -> bool:
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

    def attached(self, coffee_machine_lid:str, coffee_pod_holder:str) -> bool:
        """Returns True if the coffee machine lid is attached to the coffee pod holder.

        Args:
            coffee_machine_lid (_type_): the coffee machine lid object
            coffee_pod_holder (_type_): the coffee pod holder object

        Returns:
            bool: True if the coffee machine lid is attached to the coffee pod holder
        """
        # hardcoding the coffee machine lid to be attached to the coffee pod holder since we are only dealing with one coffee machine lid and one coffee pod holder that are always attached
        assert coffee_machine_lid == 'coffee_machine_lid' and coffee_pod_holder == 'coffee_pod_holder'
        return True
    
    def inside(self, tabletop_obj:str, container:str) -> bool:
        """Returns True if the object is inside the container.

        Args:
            tabletop_obj (str): the tabletop object
            container (str): the container object

        Returns:
            bool: True if the object is inside the container
        """
        assert tabletop_obj in self.object_types['tabletop_object'] and container in self.object_types['container']
        if container == 'coffee_pod_holder':
            if tabletop_obj == 'coffee_pod':
                return self.env.check_pod()
            else:
                return False # only coffee pods can be inside the coffee pod holder
        elif container == 'drawer':
            if tabletop_obj in ('mug', 'coffee_pod'):
                return self.env.check_in_drawer(tabletop_obj)
            else:
                return False # only mugs and coffee pods can be inside the drawer
        else: # container is mug
            if tabletop_obj == 'coffee_pod':
                return self.env.check_in_mug(tabletop_obj)
            else:
                return False # only coffee pods can be inside the mug


    def open(self, container:str) -> bool:
        """Returns True if the container is open.

        Args:
            container (_type_): the container object

        Returns:
            bool: True if the container is open
        """
        assert container in self.object_types['container']
        if container == 'coffee_pod_holder':
            return self.can_flip_down('coffee_machine_lid') and self.attached('coffee_machine_lid', 'coffee_pod_holder') # lid is currently up and attached to the coffee pod holder
        elif container == 'drawer':
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
            if self.exclusively_occupying_gripper(obj, gripper):
                return False
        return True
    
    def upright(self, mug) -> bool:
        """Returns True if the mug is upright.

        Args:
            mug (str): the mug object

        Returns:
            bool: True if the mug is upright
        """
        assert mug == 'mug'
        return self.env.check_mug_upright()

    def under(self, mug, coffee_pod_holder) -> bool:
        """Returns True if the mug is under the coffee pod holder.

        Args:
            mug (_type_): the mug object
            coffee_pod_holder (_type_): the coffee pod holder object

        Returns:
            bool: True if the mug is under the coffee pod holder
        """
        assert mug == 'mug' and coffee_pod_holder == 'coffee_pod_holder'
        return self.env.check_mug_under_pod_holder()

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
                
        