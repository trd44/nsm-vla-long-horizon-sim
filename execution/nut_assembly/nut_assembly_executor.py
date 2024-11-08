from execution.executor import *

class PickUpNutFromTabletopExecutor(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="pick-up-nut-from-tabletop")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False) -> bool:
        """execute the pick up nut from tabletop operator in the nut assembly simulation environment

        Args:
            detector (Detector): the detector for the nut assembly domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        Returns:
            bool: True if the operator was executed successfully
        """
        print("Executing pick-up-nut-from-tabletop operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        precond_satisfied, unsatisfied_preconditions = self.check_precondition(detector)
        assert precond_satisfied, f"Unsatisfied preconditions: {unsatisfied_preconditions}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the pick-up-nut-from-tabletop loop
        return True
        

class PutNutOnPeg(Executor):
    def __init__(self):
        super().__init__("coded", operator_name="put-nut-on-peg")
    
    def execute(self, detector:Detector, grounded_operator:fs.Action, render=False):
        """execute the put nut on peg operator in the nut assembly simulation environment

        Args:
            detector (Detector): the detector for the nut assembly domain
            grounded_operator (fs.Action): the grounded operator to execute
            render (bool, optional): whether to render the environment. Defaults to False.
        """
        print("Executing pick-up-nut-from-tabletop operator")
        grounded_operator_name, _ = extract_name_params_from_grounded(grounded_operator)
        assert grounded_operator_name == self.name, f"Expected operator {self.name} but got {grounded_operator_name}"
        env = detector.get_env()
        obs = detector.get_obs()

        #TODO: code the execution of the pick-up-nut-from-tabletop loop

NUT_ASSEMBLY_EXECUTORS = {
    "pick-up-nut-from-tabletop": PickUpNutFromTabletopExecutor(),
    "put-nut-on-peg": PutNutOnPeg()
}
        
