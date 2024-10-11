import base64
from HybridSymbolicLLMPlanner import *
from utils import *
from langchain_core.messages import *
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents import AgentExecutor

thought_generator = ChatOpenAI(model=config['vlm_agent']['model'], temperature=config['vlm_agent']['temperature'])

# parser = ChatOpenAI(model=config['vlm_agent']['model'], temperature=config['vlm_agent']['temperature'])
# parser_with_tool = parser.bind_tools([planner.add_operator])
# parser_prompt = [
#     SystemMessage(content=parse_operators_prompt),
#     MessagesPlaceholder(variable_name='agent_scratchpad')
# ]
# parser_agent = (
#     {
#         "operators": lambda x: x["operators"],
#         "agent_scratchpad": lambda x: format_to_openai_tool_messages(
#             x["intermediate_steps"]
#         ),
#     }
#     | ChatPromptTemplate.from_messages(parser_prompt)
#     | parser_with_tool
#     | OpenAIToolsAgentOutputParser()
# )
# parser_agent_executor = AgentExecutor(agent=parser_agent, tools=[planner.add_operator], verbose=True)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             system_identify_goal_msg
#         ),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )
# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_to_openai_tool_messages(
#             x["intermediate_steps"]
#         ),
#     }
#     | prompt
#     | model_with_tools
#     | OpenAIToolsAgentOutputParser()
# )


state_evaluator = ChatOpenAI(model=config['vlm_agent']['model'], temperature=config['vlm_agent']['temperature'])
