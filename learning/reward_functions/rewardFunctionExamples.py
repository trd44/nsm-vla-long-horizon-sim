import tarski.fstrips as fs
from typing import *

def reward_shaping(effect:fs.SingleEffect, obs:dict) -> float:
    """This is an example of a reward shaping function for the open-drawer(drawer1) operator's effects. The function should return a value between 0 and 100% that represents the progress towards achieving a given effect.

    Args:
        effect (fs.SingleEffect): the effect (sub-goal) to be achieved
        obs (dict): a numeric observation with semantics whose keys are predicates and values are numeric values

    Returns:
        float: the reward shaping value for the effect between 0% and 100%
    """
    if effect.atom.pddl_repr() == 'exclusively-occupying-gripper' and isinstance(effect, fs.AddEffect):
        target_obj = effect.atom.subterms[0].name
        progress = obs[f'{target_obj}_to_robot0_eef_dist'] / 100 #TODO: replace this with the actual max possible distance
        progress = 1 - progress
        return progress
    
     
