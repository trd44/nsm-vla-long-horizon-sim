#%%
import yaml
from utils import *
from langchain_community.agent_toolkits import FileManagementToolkit

# class PDDLRefiner:
#     def __init__(self, config_file):
#         self.config = self.load_config(config_file)
        

#     def refine_pddl(self, pddl_file):
#         # Implement your PDDL refinement logic here
#         pass

# # Example usage
# refiner = PDDLRefiner('config.yaml')
# refiner.refine_pddl('/path/to/pddl_file.pddl')
#%%
# Set the PYTHONPATH to include the current directory
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
config = load_config("config.yaml")
planning_dir = config["planning_dir"]
read_tool, write_tool = FileManagementToolkit(
    root_dir=str(planning_dir),
    selected_tools=["read_file", "write_file"],
).get_tools() 
tools = [call_planner, verify_predicates_domain, verify_predicates_problem, read_tool, write_tool]
call_planner.invoke({"domain":config['init_planning_domain'], "problem":config['init_planning_problem']})
verify_predicates_problem.invoke({"domain":config['init_planning_domain'], "problem":config['init_planning_problem']})