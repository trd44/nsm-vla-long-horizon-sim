import base64
from planning.hybrid_symbolic_llm_planner import *
from utils import *
from langchain_openai import OpenAI
from langchain_core.messages import *
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents import AgentExecutor


from openai import OpenAI
client = OpenAI()

def generate_thought(prompt:str) -> str:
    """Generate a thought based on the given prompt.

    Args:
        prompt (str): prompt to the LLM
    """
    # gpt-o1 does not support setting the temperature parameter
    completion = client.chat.completions.create(
        model=config['vlm_agent']['model'],
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return completion.choices[0].message.content

#thought_generator = ChatOpenAI(model=config['vlm_agent']['model'], temperature=config['vlm_agent']['temperature'])
#thought_generator = OpenAI(model=config['vlm_agent']['model'], temperature=config['vlm_agent']['temperature'])
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
