from detection.detector import Detector

class NutAssemblyDetector(Detector):
    '''
    Class for detecting the states in the nut assembly task
    '''
    def __init__(self, env, return_int=False):
        super().__init__(env, return_int)
        
        # Planning domain predicates
        self.predicates = {
            'small-enough-for-gripper-to-pick-up': {
                'func':self.small_enough_for_gripper_to_pick_up,
                'params':['nut', 'gripper']
            },
            'directly-on-table': {
                'func':self.directly_on_table,
                'params':['nut', 'table']
            },
            'on-peg': {
                'func':self.on_peg,
                'params':['nut', 'peg']
            },
            'exclusively-occupying-gripper': {
                'func':self.exclusively_occupying_gripper,
                'params':['nut', 'gripper']
            },
            'shapes-match': {
                'func':self.shapes_match,
                'params':['nut', 'peg']
            },
            'free': {
                'func':self.free,
                'params':['gripper']
            }
        }

        # Planning domain object types
        self.object_types = {
            'tabletop-object': ['round-peg', 'square-peg', 'round-nut', 'square-nut'],
            'peg': ['round-peg', 'square-peg'],
            'nut': ['round-nut', 'square-nut'],
            'gripper': ['gripper'],
            'table': ['table'],
        }

        # Grounded objects
        self.grounded_objects = {
            'round-peg': ['round-peg1'],
            'square-peg': ['square-peg1'],
            'round-nut': ['round-nut1'],
            'square-nut': ['square-nut1'],
            'gripper': ['gripper1'],
            'table': ['table1'],
        }

    def small_enough_for_gripper_to_pick_up(self, tabletop_obj:str, gripper:str) -> bool:
        '''
        Check if the object is small enough for the gripper to pick up.
        
        Args:
            tabletop_obj (str): The tabletop object
            gripper (str): The gripper

        Returns:
            bool: True if the object is small enough for the gripper to pick
        '''
        # Hardcoding nuts to be small enough to pick up
        if tabletop_obj == 'round-nut' or tabletop_obj == 'square-nut':
            return True
        return False

    def directly_on_table(self, tabletop_obj:str, table:str='table') -> bool:
        '''
        Check if the object is directly on the table.
        
        Args:
            tabletop_obj (str): The tabletop object
            table (str): The table
            
        Returns:
            bool: True if the object is directly on the table                
        '''
        assert tabletop_obj in self.object_types['tabletop-object']
        return self.env.check_directly_on_table(tabletop_obj)

    def on_peg(self, nut:str, peg:str) -> bool:
        '''
        Check if the nut is on the peg

        Args:
            nut (str): The nut
            peg (str): The peg

        Returns:
            bool: True if the nut is on the peg
        '''
        assert nut in self.object_types['nut']
        assert peg in self.object_types['peg']
        return self.env.check_on_peg(nut, peg)

    def exclusively_occupying_gripper(self, nut:str, gripper:str) -> bool:
        '''
        Check if the object is exclusively occupying the gripper.

        Args:
            nut (str): The tabletop object
            gripper (str): The gripper

        Returns:
            bool: True if the nut is exclusively occupying the gripper
        '''
        assert nut in self.object_types['nut'] and gripper in self.object_types['gripper']
        gripper = self.env.robots[0].gripper
        if nut == 'round-nut':
            nut = self.env.nuts[1]
        elif nut == 'square-nut':
            nut = self.env.nuts[0]
        return self.env._check_grasp(gripper, nut.contact_geoms)

    def shapes_match(self, nut:str, peg:str) -> bool:
        '''
        Check if the shapes of the nut and peg match

        Args:
            nut (str): The nut
            peg (str): The peg

        Returns:
            bool: True if the shapes of the nut and peg match
        '''
        assert nut in self.object_types['nut']
        assert peg in self.object_types['peg']
        if nut == 'round-nut' and peg == 'round-peg':
            return True
        elif nut == 'square-nut' and peg == 'square-peg':
            return True
        return False

    def free(self, gripper:str) -> bool:
        '''
        Check if the gripper is free

        Args:
            gripper (str): The gripper

        Returns:
            bool: True if the gripper is free
        '''
        assert gripper in self.object_types['gripper']
        for obj in self.object_types['nut']:
            if self.exclusively_occupying_gripper(obj, gripper):
                return False
        return True
    
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