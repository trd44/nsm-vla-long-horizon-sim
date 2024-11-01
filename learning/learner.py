import os
import execution
import execution.executor

class Learner:
    def __init__(self, env, domain:str, operator_to_learn:str):
        self.env = env
        self.domain = domain
        self.operator = operator_to_learn
    
    def learn(self):
        #TODO: implement this
        pass
    
