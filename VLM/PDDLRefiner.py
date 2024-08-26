#%%
import base64
from utils import *
from VLM.PDDLprompts import *
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor

#%%
# Set the PYTHONPATH to include the current directory
config = load_config("config.yaml")
planning_dir = config["planning_dir"]
read_tool, write_tool = FileManagementToolkit(
    root_dir=str(planning_dir),
    selected_tools=["read_file", "write_file"],
).get_tools() 
tools = [call_planner, verify_predicates_domain, verify_predicates_problem, read_tool, write_tool]
model = ChatOpenAI(model=config['vlm_agent']['model'])
model_with_tools = model.bind_tools(tools)

# get the encoded agentview image
base64_image = encode_image(config['image_path'])
human
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_identify_goal_msg
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | model_with_tools
    | OpenAIToolsAgentOutputParser()
)

#agent_executor = create_react_agent(model, tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# call_planner.invoke({"domain":config['init_planning_domain'], "problem":config['init_planning_problem']})
# verify_predicates_problem.invoke({"domain":config['init_planning_domain'], "problem":config['init_planning_problem']})

if __name__ == "__main__":
    # prompt the user for input and start a conversation with the agent
    # while True:
    # user_input = input("Enter your input: ")
    res = list(agent_executor.stream({"input": "There is a drawer. There is a coffee pod inside the drawer. Install the coffee pod in the coffee dispenser and place the mug that is on the table under the dispenser"}))
    print(res)
            # print(chunk)
            # print("===")