import os
import logging
import heapq
import copy
import itertools
import dill
from collections import deque

from tarski.search import GroundForwardSearchModel
from tarski.model import Model
from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
from tarski.io.fstrips import FstripsReader, FstripsWriter
from tarski.syntax import land, neg, CompoundFormula, Sort, Constant, Atom
from tarski.syntax.formulas import VariableBinding
from tarski.syntax.builtins import *
from tarski import fstrips as fs
from tarski.evaluators.simple import evaluate
from tarski.model import Model
from tarski.syntax.builtins import BuiltinPredicateSymbol
from VLM.openai_api import *
from utils import *

class LLMPlanner:
    """First iteration of a purely LLM planner"""
    #TODO: implement the LLM planner


