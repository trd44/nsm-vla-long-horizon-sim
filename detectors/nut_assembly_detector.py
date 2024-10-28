import itertools

class NutAssemblyDetector:
    '''
    Class for detecting the states in the nut assembly task
    '''
    def __init__(self, env, return_int=False):
        self.env = env
        if self.env.viewer_get_obs:
            self.obs = self.env.viewer._get_observations()
        else:
            self.obs = self.env._get_observations()
        self.return_int = return_int
        
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

    def update_obs(self, obs=None):
        """Update the observation

        Args:
            obs (OrderedDict, optional): The observation returned by `env.step(...)`. Defaults to None.
        """
        if obs is not None:
            self.obs = obs
        else:
            self.obs = self.env.viewer._get_observations() if self.env.viewer_get_obs else self.env._get_observations()

    def r_int(self, value):
        return int(value) if self.return_int else value

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

    def get_groundings(self) -> dict:
        '''
        Update the groundings by calling all the predicate functions

        Returns:
            dict: The groundings
        '''
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
                param_strs = [str(self.grounded_objects[param]) for param in comb]
                predicate_str = f'{predicate_name}({",".join(param_strs)})'
                # predicate_str = f'{predicate_name}({",".join([self.grounded_objects[param] for param in comb])})'
                groundings[predicate_str] = truth_value
        return groundings