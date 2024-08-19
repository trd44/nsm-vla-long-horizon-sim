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
config = load_config("config.yaml")
planning_dir = config["planning_dir"]
read_tool, write_tool = FileManagementToolkit(
    root_dir=str(planning_dir),
    selected_tools=["read_file", "write_file"],
).get_tools()  # If you don't provide a root_dir, operations will default to the current working directory
tools = [call_planner, verify_predicates_domain, verify_predicates_problem, read_tool, write_tool]
verify_predicates_problem.invoke({"domain":config['init_planning_domain'], "problem":config['init_planning_problem']})