# placeholder prompt for the LLM to order a given operator's effects in the order they are expected to be achieved
order_effects_prompt = """Sort the effects of the grounded operator below in the order they are expected to be achieved. The operator. In order to manipulate an object， your gripper needs to make contact with the object first if it is not already making contact:
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
reward_shaping_prompt = """generate a python reward function"""