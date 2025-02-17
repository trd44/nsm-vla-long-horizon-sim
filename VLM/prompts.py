# multimodal Chain-of-thought prompts for the VLM https://arxiv.org/pdf/2302.14045 
# the rationale generation prompt that asks the VLM to generate a description of the image
rationale_generation_prompt = """You are a robot arm with a gripper that can manipulate tabletop objects. You observe an image of the current state of the world. Describe what you see in the image of the task in detail:"""
eval_prompt = """You are a robot arm with a gripper that can manipulate tabletop objects. Based on the image of the current state of the world, you observe the following:
{rationale}
Question: You have been executing the following operator represented in the Planning Domain Definition Language (PDDL) as follows:
{grounded_operator}
Output `yes` if all effects are achieved, `no` if not all effects are achieved.
Answer:
"""

# a very simple prompt for planning
plan_prompt = """You are a robot arm with a gripper that can manipulate tabletop objects. You are capable of understanding the Planning Domain Definition Language (PDDL). Based on the image, generate a plan to achieve the goal of the task."""