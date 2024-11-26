from detection.detector import Detector
from typing import *
from scipy.spatial.transform import Rotation as R
from mimicgen.envs.robosuite.coffee import Coffee_Drawer_Novelty

class CoffeeDetector(Detector):
    def __init__(self, env, return_int=False):
        super().__init__(env, return_int)
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
                'params':['coffee_machine_lid']
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
            'in': {
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
        self.object_types = {
            'tabletop_object':['coffee_pod', 'coffee_machine_lid', 'coffee_pod_holder', 'mug', 'drawer'], 
            'container':['coffee_pod_holder', 'drawer', 'mug'], 'gripper':['gripper'], 
            'mug':['mug'], 
            'coffee_machine_lid':['coffee_machine_lid'], 
            'coffee_pod_holder':['coffee_pod_holder'], 
            'drawer':['drawer'], 
            'coffee_pod':['coffee_pod'], 
            'table':['table']
        }

        # this is a hack to map the grounded object to their pddl format. This is needed because the grounded object is in the format e.g. 'mug' while the pddl object is in the format 'mug1'
        self.grounded_object_to_pddl_object = {'mug':'mug1', 'coffee_machine_lid':'coffee_machine_lid1', 'coffee_pod_holder':'coffee_pod_holder1', 'drawer':'drawer1', 'coffee_pod':'coffee_pod1', 'table':'table1', 'gripper':'gripper1'}

        self.grounded_tabletop_object_to_coffee_class_object = {'mug':self.env.mug, 'coffee_machine_lid':self.env.coffee_machine.lid, 'coffee_pod_holder':self.env.coffee_machine.pod_holder, 'drawer':self.env.cabinet_object, 'coffee_pod':self.env.coffee_pod, 'gripper':self.env.robots[0].gripper}

    
    def small_enough_for_gripper_to_pick_up(self, tabletop_obj:str) -> bool:
        """Returns True if the object is small enough for the gripper to pick up.
        Args:
            tabletop_obj (str): the tabletop object
            
        Returns:
            bool: True if the object is small enough for the gripper to pick up
        """
        #hardcoding mug and pod to be small enough to pick up
        if 'mug' in tabletop_obj or 'coffee_pod' in tabletop_obj:
            return True
        return False
    
    def can_flip_up(self, coffee_machine_lid:str) -> bool:
        """Returns True if the coffee pod lid can be flipped up.

        Args:
            coffee_machine_lid (str: the coffee machine lid object

        Returns:
            bool: True if the coffee pod lid can be flipped up
        """
        assert self._is_type(coffee_machine_lid, 'coffee_machine_lid')
        return self.env.check_can_flip_up_lid()
        

    def can_flip_down(self, coffee_machine_lid:str) -> bool:
        """Returns True if the coffee pod lid can be flipped down.

        Args:
            coffee_machine_lid (str): the coffee pod lid object

        Returns:
            bool: True if the coffee pod lid can be flipped down
        """
        assert self._is_type(coffee_machine_lid, 'coffee_machine_lid')
        return not self.can_flip_up(coffee_machine_lid)

    def directly_on_table(self, tabletop_obj:str, table:str) -> bool:
        """Returns True if the object is directly on the table.

        Args:
            tabletop_obj (str): the tabletop object
            table (str): the name of the table object

        Returns:
            bool: True if the object is directly on the table
        """
        assert self._is_type(tabletop_obj, 'tabletop_object') and self._is_type(table, 'table')
        # hardcoding coffee_machine_lid to not be directly on table
        if self._is_type(tabletop_obj, 'coffee_machine_lid'):
            return False
        return self.env.check_directly_on_table(tabletop_obj)


    def exclusively_occupying_gripper(self, tabletop_obj:str, gripper:str) -> bool:
        """Returns True if the object is exclusively occupying the gripper.

        Args:
            tabletop_obj (str): the tabletop object
            gripper (str): the gripper object

        Returns:
            bool: True if the object is exclusively occupying the gripper
        """
        assert self._is_type(tabletop_obj, 'tabletop_object') and self._is_type(gripper, 'gripper')
        gripper = self.grounded_tabletop_object_to_coffee_class_object[gripper]
        tabletop_obj_contact_geoms = self.grounded_tabletop_object_to_coffee_class_object[tabletop_obj].contact_geoms
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
        assert self._is_type(tabletop_obj, 'tabletop_object') and self._is_type(container, 'container')
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
        assert self._is_type(container, 'container')
        if self._is_type(container, 'coffee_pod_holder'):
            return self.can_flip_down('coffee_machine_lid') and self.attached('coffee_machine_lid', 'coffee_pod_holder') # lid is currently up and attached to the coffee pod holder
        elif self._is_type(container, 'drawer'):
            return self.env.check_drawer_open()
        elif self._is_type(container, 'mug'):
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
        assert self._is_type(mug, 'mug')
        return self.env.check_mug_upright()

    def under(self, mug, coffee_pod_holder) -> bool:
        """Returns True if the mug is under the coffee pod holder.

        Args:
            mug (_type_): the mug object
            coffee_pod_holder (_type_): the coffee pod holder object

        Returns:
            bool: True if the mug is under the coffee pod holder
        """
        assert self._is_type(mug, 'mug') and self._is_type(coffee_pod_holder, 'coffee_pod_holder')
        return self.env.check_mug_under_pod_holder()
    
    def verify_env(self, env) -> bool:
        """Verify that the environment is the correct environment class.

        Args:
            env (MujocoEnv): the environment
        Returns:
            bool: True if the environment is correct
        """
        while hasattr(env, 'env'):
            env = env.env
            if isinstance(env, Coffee_Drawer_Novelty):
                return True
        return False
