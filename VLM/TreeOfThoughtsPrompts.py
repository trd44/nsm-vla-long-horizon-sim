propose_operator_prompt = """You are a robot capable of understanding the Planning Domain Definition Language (PDDL). Given the current state , a set of objects deemed relevant to the task at hand, novel object(s) of interest, a set of existing operators, and a goal state, propose 1 non-existing operator(s) involving the novel object(s) that can be executed in the current state (not in the future) that would help make progress towards the goal. Output `no operator` if no new operator should be proposed. Otherwise,  output the proposed operator's name(s) and parameters by imitating the style of the existing operators in the following format:
```
(:action proposed_non_existing_operator_name
    :parameters (?param1 - param1-type ?param2 - param2-type ...?paramN - paramN-type)
)
...
```
Output the ground parameter objects for each proposed operator in the following format:
```
ground objects for proposed_non_existing_operator_name: object1, object2, object3 ...objectN
...
```
Problem:
```
Current state (unmentioned atoms are assumed false): 
{current_state}
{true_atoms_novel_obj}
{false_atoms_novel_obj}
Goal state:
{goal_state}
Relevant objects: 
{relevant_objects}
Object types: 
{object_types}
Novel object(s) of interest: 
{novel_objects}
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
define_effect_prompt = """ You are a robot capable of understanding the Planning Domain Definition Language (PDDL). Given an operator's name, its parameter objects, the preconditions that are satisfied in the current state, and an image of the current state, define the resulting state after applying the operator in the current state. The effects MUST ONLY involve objects in `The operator`'s parameters. Fill the effects of `The operator` based on the resulting state in the following format:
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


vote_prompt = """Given a set of operators, several choices of world states and the goal state, decide which state is the most promising at reaching the goal state. Analyze each choice in detail, then conclude in the last line "The best choice is {id}", where id is the integer id of the choice.
{choices}
"""

    