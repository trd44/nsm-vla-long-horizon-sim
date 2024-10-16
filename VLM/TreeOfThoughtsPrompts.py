propose_operator_prompt = """You are a robot arm with a gripper that uses the gripper to grasp and manipulate tabletop objects. You are capable of understanding the Planning Domain Definition Language (PDDL). Given the current state, a set of objects deemed relevant to the task at hand, novel object(s) of interest, a set of existing operators, and a goal state, propose 1 non-existing operator involving the novel object(s) that is EXECUTABLE in the current state (not in the future) that would help make progress towards the goal. Output `no operator` if no new operator should be proposed. Otherwise, output the proposed operator and the ground parameter objects by imitating the style of the existing operators in the following format. Avoid quantifiers and conditional effects:
```
(:action proposed_non_existing_operator_name
    :parameters (?param1 - param1-type ?param2 - param2-type ...?paramN - paramN-type)
)
ground objects: object1, object2, object3 ...objectN
```
Problem:
```
Current state (unmentioned atoms are assumed false): 
{current_state}
Goal state:
{goal_state}
Novel object(s) of interest: 
{novel_objects}
Specifically, the following atoms are true for the novel object(s):
{true_atoms_novel_obj}
The following atoms are false for the novel object(s):
{false_atoms_novel_obj}
Other relevant objects: 
{relevant_objects}
Object types: 
{object_types}
Existing operators with parameters:
{existing_operators}
```
Answer: Let's think step by step.
"""

# prompt asking the LLM to define the precondition for the new operator
define_precondition_prompt = """You are a robot capable of understanding the Planning Domain Definition Language (PDDL). Given an operator's name, its parameter objects, the current states of parameter objects, and an image of the current state, fill in the preconditions of `The Operator` by selecting a relevant subset of atoms in the `Current state` section. The preconditions must ALREADY be satisfied in the current state. The preconditions MUST ONLY involve objects in `The Operator`'s parameters. Output the operator in the following format:
```
(:action operator_name
    :parameters (?param1 - param1-type ?param2 - param2-type ...)
    :precondition (and (predicate1 ?param1 ?param2...) (predicate2 ?param1 ?param2...)...)
)
```
Problem:
```
Current state: 
{full_param_obj_atoms}
Example operators with parameters and preconditions:
{example_operators}
The Operator:
{proposed_operator}
```
Answer: Let's think step by step.
"""

# prompt asking the LLM to define the effect for the new operator
define_effect_prompt = """You are a robot arm with a gripper that uses the gripper to grasp and manipulate tabletop objects. You are capable of understanding the Planning Domain Definition Language (PDDL). Given an operator's name, its parameter objects, the preconditions that are satisfied in the current state, define the resulting state after applying the operator in the current state. The effects MUST ONLY involve objects in `The operator`'s parameters. Avoid quantifiers and conditional effects. Fill the effects of `The operator` based on the resulting state in the following format:
```
(:action operator_name
    :parameters (?param1 - param1-type ?param2 - param2-type ...)
    :precondition (and (predicate1 ?param1 ?param2...) (predicate2 ?param1 ?param2...)...)
    :effect (and (predicate1 ?param1 ?param2...) (predicate2 ?param1 ?param2...)...)
)
```
Problem:
```
Current state: 
{full_param_obj_atoms}
Example operators with parameters, preconditions, and effects:
{example_operators}
The operator:
{proposed_operator_with_precondition}
```
Answer: Let's think step by step.
"""


    