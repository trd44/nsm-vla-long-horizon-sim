propose_prompt = """You are a robot capable of understanding the Planning Domain Definition Language (PDDL). Given the current state, a set of objects deemed relevant to the task at hand, novel object(s) of interest whose state can be changed, a set of available predicates, a set of existing operators, and a goal state, propose {n} possible new operators that would change the state of the novel object(s) such that it is more likely to reach the goal state. The operators' preconditions and effects must be defined with ONLY the available predicates and the operators parameters must contain one or more of the novel object(s) of interest.
Current state: 
{current_state}
Relevant objects: 
{relevant_objects}
Novel object(s) of interest: 
{novel_objects}
Object type hierarchy: 
{object_type_hierarchy}
Available predicates: 
{available_predicates}
Existing operators:
{existing_operators}
Possible operators on novel objects:

Current state: 
{current_state}
Relevant objects: 
{relevant_objects}
Novel object(s) of interest: 
{novel_objects}
Object type hierarchy: 
{object_type_hierarchy}
Available predicates: 
{available_predicates}
Existing operators:
{existing_operators}
Possible operators on novel objects:
"""

vote_prompt = """Given a set of operators, several choices of world states and the goal state, decide which state is the most promising at reaching the goal state. Analyze each choice in detail, then conclude in the last line "The best choice is {id}", where id is the integer id of the choice.
{choices}
"""