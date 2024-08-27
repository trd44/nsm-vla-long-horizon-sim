
import os
from utils import *
from VLM.PDDLprompts import *
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor

config = load_config("config.yaml")
root_dir = os.getcwd()
planning_dir = root_dir + os.sep + config['planning_dir']
tools = [verify_predicates_domain, verify_predicates_problem, read_file, write_file]
model = ChatOpenAI(model=config['vlm_agent']['model'], temperature=config['vlm_agent']['temperature'])
model_with_tools = model.bind_tools(tools)

# get the encoded agentview image
base64_image = encode_image(config['image_path'])

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            [
                {
                    "type": "text",
                    "text": "{input}",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            ],
        ),
        ("system", system_image_describe_msg),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
novel_object_detector_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            [
                {
                    "type": "text",
                    "text": "{input}",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            ],
        ),
        ("system", system_novel_object_detection_msg),
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
    | novel_object_detector_prompt
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
    # res = list(agent_executor.stream())
    res = agent_executor.invoke({"input": "The novel object is the drawer. There is a coffee pod inside the drawer. Install the coffee pod in the coffee dispenser and place the mug under the coffee pod holder"})
    print(res)