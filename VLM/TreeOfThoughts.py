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

thought_generator = ChatOpenAI(model=config['vlm_agent']['model'], temperature=config['vlm_agent']['temperature'])
state_evaluator = ChatOpenAI(model=config['vlm_agent']['model'], temperature=config['vlm_agent']['temperature'])