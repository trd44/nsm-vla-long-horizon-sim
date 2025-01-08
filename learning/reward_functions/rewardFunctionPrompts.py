
# placeholder prompt for the LLM to generate a reward function. 
reward_shaping_prompt = """Generate a dense reward shaping function for the grounded operator below by filling in the blanks in the following reward shaping function template. The grounded operator:
```
{grounded_operator}
```
The observation_with_semantics dictionary:
```
{observation_with_semantics}
```
Output the reward shaping function by finishing the following reward function. To calculate the reward for the given grounded effect, you should extract the relevant information from the observation_with_semantics dictionary. You should then use the values to calculate progress towards the grounded effect. Do not make up any numbers.
The template:
```
# llm generated reward shaping function
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