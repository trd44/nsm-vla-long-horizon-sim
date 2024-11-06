from execution.executor import *


class PickUpFromTabletopExecutor(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="pick-up-from-tabletop")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the pick up from tabletop operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing pick-up-from-tabletop operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()
        
        #TODO: code the execution of the pick-up-from-tabletop loop
        raise NotImplementedError

class OpenCoffeePodHolder(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="open-coffee-pod-holder")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the open coffee pod holder operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing open-coffee-pod-holder operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the open-coffee-pod-holder loop
        raise NotImplementedError
    
class CloseCoffeePodHolder(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="close-coffee-pod-holder")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the close coffee pod holder operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing close-coffee-pod-holder operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the close-coffee-pod-holder loop
        raise NotImplementedError 

class FreeGripperFromLargeObject(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="free-gripper-from-large-object")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the free gripper from large object operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing free-gripper-from-large-object operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the free-gripper-from-large-object loop
        raise NotImplementedError
    
class PlacePodInHolderFromGripper(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="place-pod-in-holder-from-gripper")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the place pod in holder from gripper operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing place-pod-in-holder-from-gripper operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the place-pod-in-holder-from-gripper loop
        raise NotImplementedError

class PlaceMugUnderHolderFromGripper(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="place-mug-under-holder-from-gripper")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the place mug under holder from gripper operator in the coffee simulation environment

        Args:
            detector (Detector): the detector for the coffee domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing place-mug-under-holder-from-gripper operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the place-mug-under-holder-from-gripper loop
        raise NotImplementedError
    


COFFEE_EXECUTORS = {
    "pick-up-from-tabletop": PickUpFromTabletopExecutor(),
    "open-coffee-pod-holder": OpenCoffeePodHolder(),
    "close-coffee-pod-holder": CloseCoffeePodHolder(),
    "free-gripper-from-large-object": FreeGripperFromLargeObject(),
    "place-pod-in-holder-from-gripper": PlacePodInHolderFromGripper(),
    "place-mug-under-holder-from-gripper": PlaceMugUnderHolderFromGripper()
}