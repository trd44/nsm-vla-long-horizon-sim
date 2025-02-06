import base64
from planning.planners import *
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

config = load_config('config.yaml')

def chat_completion(prompt:str, base64_image:bytes=None) -> str:
    """Get response from OpenAI based on the prompt.

    Args:
        prompt (str): prompt to the LLM or VLM
        base64_image (bytes, optional): image to be included in the prompt. Defaults to None.
    """
    if base64_image:
        completion = client.chat.completions.create(
            model=config['vlm_agent']['model'],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            
        )
    else:
        # gpt-o1 does not support setting the temperature parameter
        completion = client.chat.completions.create(
            model=config['llm_agent']['model'],
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
    return completion.choices[0].message.content

if __name__=="__main__":
    # test the VLM
    prompt = "You are the robot arm with a gripper performing tabletop manipulation tasks. Describe what you see in the image of the task."
    image_path = "images/agent_view.jpg"
    base64_image = encode_image(image_path)
    response = chat_completion(prompt)
    print(response)