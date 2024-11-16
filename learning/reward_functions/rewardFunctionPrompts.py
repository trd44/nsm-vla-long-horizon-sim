# placeholder prompt for the LLM to order a given operator's effects in the order they are expected to be achieved
order_effects_prompt = """Sort the effects of the grounded operator below in the order they are expected to be achieved. In order to manipulate an object， the object needs to be exclusively occupying the gripper first if it is not already. The operator:
```
{grounded_operator}
```
Output the sorted grounded effects in the following format where the effects are separated by new lines.
```
effects:
effect1
effect2
effect3
...
```
"""

# placeholder prompt for the LLM to generate a reward function. 
reward_shaping_prompt = """Generate a dense reward shaping function for the grounded operator below by filling in the blanks in the following reward shaping function template. The grounded operator:
```
{grounded_operator}
```
The observation_with_semantics dictionary:
```
{observation_with_semantics}
```
Output the reward shaping function by filling in the blanks in the following template:
```
from typing import *
import numpy as np

def reward_shaping_fn(observation_with_semantics:Dict[str, Union[bool, float, np.array]], grounded_effect:str) -> float:
    '''
    Args:
        observation_with_semantics (Dict[str, Union[bool, float, np.array]]): a dictionary containing the observation with semantics
        grounded_effect (str): a grounded effect of the operator
    Returns:
        float: the reward between 0 and 1 indicate percentage of completion towards the grounded effect
    '''
    ...
```
"""