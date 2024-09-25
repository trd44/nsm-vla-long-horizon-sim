propose_operator_prompt = """You are a robot capable of understanding the Planning Domain Definition Language (PDDL). Given the current state, a set of objects deemed relevant to the task at hand, novel object(s) of interest, a set of existing operators, and a goal state, propose 1 non-existing operator involving the novel object(s) that is executable in the current state (not in the future) and would help make progress towards the goal. Output `no operator` if no new operator should be proposed. Otherwise, output the operator's name and parameters in the following format:
```
(:action operator_name
    :parameters (param1 param2 ...)
)
```
Output the grounded objects that should be passed into the new operator as parameters in the following format:
```
ground objects: object1, object2, ...
```
Problem:
```
Current state (unmentioned atoms are assumed false): 
{current_state}
Goal state:
{goal_state}
Relevant objects: 
{relevant_objects}
Novel object(s) of interest: 
{novel_objects}
Specifically, here are things that are true about the novel object(s):
{true_atoms_novel_obj}
Here are things that are false about the novel object(s):
{false_atoms_novel_obj}
Object types: 
{object_types}
Existing operators:
{existing_operators}
```
Answer: Let's think step by step.
"""

# prompt asking the LLM to define the precondition for the new operator
define_precondition_prompt = """You are a robot capable of understanding the Planning Domain Definition Language (PDDL). Given an operator's name, its parameter objects, the current states of parameter objects, fill in the preconditions of the operator by selecting a relevant subset of atoms in the `Current state` section. The preconditions must ALREADY be satisfied in the current state. Output the operator in the following format:
```
(:action operator_name
    :parameters (?param1 - param1-type ?param2 - param2-type ...)
    :precondition (and (predicate1(param(s)...)) (predicate2(param(s)...))...)
)
```
Problem:
```
Current state: 
{full_current_state_atoms}
Operator:
{proposed_operator}
```
Answer: Let's think step by step.
"""

# prompt asking the LLM to define the effect for the new operator
define_effect_prompt = """You are a robot capable of understanding the Planning Domain Definition Language (PDDL). Given an operator's name, its parameter objects, the preconditions that are satisfied in the current state, output the resulting state after applying the operator in the current state. Fill the effects of the operator based on the resulting state in the following format:
```
(:action operator_name
    :parameters (?param1 - param1-type ?param2 - param2-type ...)
    :precondition (and (predicate1(param(s)...)) (predicate2(param(s)...))...)
    :effect (and (predicate1(param(s)...)) (predicate2(param(s)...))...)
)
```
Problem:
```
Current state: 
{full_current_state_atoms}
Operator:
{proposed_operator_with_precondition}
```
Answer: Let's think step by step.
"""


vote_prompt = """Given a set of operators, several choices of world states and the goal state, decide which state is the most promising at reaching the goal state. Analyze each choice in detail, then conclude in the last line "The best choice is {id}", where id is the integer id of the choice.
{choices}
"""


# current_state = "can-hold(coffee-pod1), can-hold(mug1), can-open(coffee-pod-holder1), can-contain(coffee-pod-holder1,coffee-pod1), can-contain(drawer1, coffee-pod1), in(coffee-pod1, drawer1), free(gripper1), on-table(mug1,table1), on-table(drawer1,table1), open(drawer1)"
# relevant_objects = "coffee-pod1, mug1, coffee-pod-holder1, gripper1, drawer1, table1"
# novel_objects = "drawer1"
# object_types = "coffee-pod mug coffee-pod-holder gripper drawer table"
# available_predicates = """(can-hold ?obj - object)
#     (can-open ?obj - object)
#     (can-contain ?container - object ?obj - object)
#     (on-table ?obj - object ?table - table)
#     (holding ?obj - object) ; whether the gripper is holding an object. If true, `free gripper` should be false.
#     (in ?obj - object ?container - object)
#     (open ?container - object)
#     (free ?gripper - gripper) ; whether the gripper is not holding anything. If true, `holding` should be false.
#     (under ?bottom - object ?top - object)"""
# existing_operators = """(:action pick-up-tabletop
#     :parameters (?obj - object ?table - table ?gripper - gripper) 
#     :precondition (and (on-table ?obj ?table) (can-hold ?obj) (free ?gripper)) 
#     :effect (and (holding ?obj) (not (on-table ?obj ?table)) (not (free ?gripper))) 
# )

# (:action open-coffee-pod-holder
#     :parameters (?holder - coffee-pod-holder ?lid - coffee-machine-lid ?gripper - gripper)
#     :precondition (and (not (open ?holder)) (can-open ?holder) (free ?gripper))
#     :effect (open ?holder)
# )

# (:action close-coffee-pod-holder
#     :parameters (?holder - coffee-pod-holder ?lid - coffee-machine-lid ?gripper - gripper)
#     :precondition (and (open ?holder) (free ?gripper))
#     :effect (not (open ?holder))
# )

# (:action place-pod-in-holder
#     :parameters (?pod - coffee-pod ?holder - coffee-pod-holder ?gripper - gripper)
#     :precondition (and (holding ?pod) (open ?holder) (can-contain ?holder ?pod) (not (free ?gripper)))
#     :effect (and (not (holding ?pod)) (in ?pod ?holder) (free ?gripper))
# )


# (:action place-mug-under-holder
#     :parameters (?mug - mug ?holder - coffee-pod-holder ?gripper - gripper)
#     :precondition (holding ?mug)
#     :effect (and (under ?mug ?holder) (not (holding ?mug)) (free ?gripper))
# )"""
# executed_operators = "open-drawer(drawer1)"
# propose_prompt = propose_prompt.format(n=2, current_state=current_state, relevant_objects=relevant_objects, novel_objects=novel_objects, object_type=object_types, available_predicates=available_predicates, existing_operators=existing_operators, executed_operators=executed_operators)
    