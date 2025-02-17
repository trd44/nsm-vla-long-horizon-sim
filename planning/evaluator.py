from typing import *
from VLM.prompts import *
from VLM.openai_api import *
from utils import *
from planner import *

def vlm_evaluate_success(obs:dict, operator:str):
    """evaluate whether the operator has been successfully executed in the environment

    Args:
        obs (dict): the observation of the environment
        operator (str): the grounded pddl representation of the operator
    """
    image:np.array = obs['agentview_image']
    base64_image = numpy_to_base64(image)
    # multimodal CoT
    rationale = chat_completion(rationale_generation_prompt, base64_image)
    full_eval_prompt = eval_prompt.format(rationale=rationale, grounded_operator=operator)
    response = chat_completion(full_eval_prompt)
    # parse response
    if response.lower() == 'yes':
        return True
    elif response.lower() == 'no':
        return False
    else:
        raise ValueError("Invalid response from the VLM")

if __name__ == "__main__":
    # test the VLM evaluator
    plan = call_planner(pddl_dir="planning/PDDL/nut_assembly/")
    grounded_op = plan[0][0]
    image = load_image("images/agent_view.jpg")
    dummy_obs = {'agentview_image': image}
    success = vlm_evaluate_success(dummy_obs, grounded_op)