propose_prompt = """You are a robot capable of understanding the Planning Domain Definition Language (PDDL). Given the current state, a set of objects deemed relevant to the task at hand, novel object(s) of interest whose state can be changed, a set of available predicates, a set of existing operators, and a goal state, propose up to {n} new operators that would help you reach the goal state. The new operators must be executable in the current state i.e. check each operator's preconditions and make sure to only propose operators whose preconditions are satisfied in the current state. The operators' preconditions and effects must be defined using ONLY the available predicates. The operators' parameters must contain one or more of the novel object(s) of interest. An operator's preconditions and effects must ONLY involve objects in the operator's parameters list. Do not propose existing operators. If no new operator can be proposed, write nothing. Otherwise, separate each operator with a newline. Write each operator in the following PDDL format:
(:action operator_name
    :parameters (param1 param2 ...)
    :precondition (and (precondition1 precondition2 ...))
    :effect (and (effect1 effect2 ...))
)
Current state (unmentioned atoms are assumed false): 
{current_state}
Relevant objects: 
{relevant_objects}
Novel object(s) of interest: 
{novel_objects}
Object types: 
{object_types}
Available predicates: 
{available_predicates}
Existing operators:
{existing_operators}
Possible operators on novel objects:
"""

# prompt asking the LLM to parse text containing operators and add them to a FOL problem using the problem's add_operator method
parse_operators_prompt = """Given a text containing one or more operators separated with newlines in PDDL format, appropriately use the given tool to add each operator. The text is as follows:
{operators}"""

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
    