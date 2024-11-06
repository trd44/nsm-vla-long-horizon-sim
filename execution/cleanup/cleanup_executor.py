from execution.executor import *

class PickUpFromTabletop(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="pick-up-from-tabletop")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action):
        """execute the pick up from tabletop operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing pick-up-from-tabletop operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the pick-up-from-tabletop loop
        
    
class FreeGripperFromLargeObject(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="free-gripper-from-large-object")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action):
        """execute the free gripper from large object operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing free-gripper-from-large-object operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the free-gripper-from-large-object loop
        

class OpenDrawer(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="open-drawer")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action):
        """execute the open drawer operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing open-drawer operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the open-drawer loop
    
class CloseDrawer(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="close-drawer")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action):
        """execute the close drawer operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing close-drawer operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the close-drawer loop

class PlaceInDrawerFromGripper(Executor): 
    def __init__(self):
        super().__init__("coded", operator_name="place-in-drawer-from-gripper")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action):
        """execute the place in drawer from gripper operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
        """
        print("Executing place-in-drawer-from-gripper operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the place-in-drawer-from-gripper loop
    