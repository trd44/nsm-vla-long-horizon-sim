from detection.detector import Detector
from typing import *
from scipy.spatial.transform import Rotation as R

class CleanupDetector(Detector):
    def __init__(self, env, return_int=False):
        super().__init__(env, return_int)
        # predicate mappings
        self.predicates = {
            'small-enough-for-gripper-to-pick-up': {
                'func':self.small_enough_for_gripper_to_pick_up,
                'params':['tabletop_object','gripper']
                }, 
            'directly-on-table': {
                'func':self.directly_on_table,
                'params':['tabletop_object', 'table']
            }, 
            'large-enough-for-gripper-to-reach-inside': {
                'func':self.large_enough_for_gripper_to_reach_inside,
                'params':['container','gripper']
            },
            'occupying-gripper': {
                'func':self.exclusively_occupying_gripper,
                'params':['tabletop_object', 'gripper']
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
        self.object_types = {'tabletop_object':['cube', 'mug', 'drawer'], 'container':['drawer', 'mug'], 'gripper':['gripper'], 'mug':['mug'], 'drawer':['drawer'], 'table':['table']}

        # this is a hack to map the grounded object to their pddl format. This is needed because the grounded object is in the format e.g. 'mug' while the pddl object is in the format 'mug1'
        self.grounded_object_to_pddl_object = {'mug':'mug1', 'cube':'cube1', 'drawer':'drawer1', 'table':'table1', 'gripper':'gripper1'}
    
    def small_enough_for_gripper_to_pick_up(self, tabletop_obj:str, gripper:str) -> bool:
        """
        Returns True if the object is small enough for the gripper to pick up.
        """
        #hardcoding mug and pod to be small enough to pick up
        if tabletop_obj == 'mug' or tabletop_obj == 'cube':
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

    def large_enough_for_gripper_to_reach_inside(self, container:str, gripper:str) -> bool:
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
            if tabletop_obj in ('mug', 'cube'):
                return self.env.check_in_drawer(tabletop_obj)
            else:
                return False # only mugs and cubes can be inside the drawer
        elif container == 'mug':
            if tabletop_obj == 'cube':
                return self.env.check_in_mug(tabletop_obj)
            else:
                return False # only cubes can be inside the mug
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
            if self.exclusively_occupying_gripper(obj, gripper):
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
            if tabletop_obj in ('mug', 'cube'):
                return True
        elif container == 'mug':
            if tabletop_obj == 'cube':
                return True
        return False

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

        