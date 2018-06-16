# contains abstract class Agent
# we have two kinds of agents, in general, answerer / questioner
# author: satwik kottur

import torch
import torch.nn as nn


class Agent(nn.Module):
    # initialize
    def __init__(self):
        super(Agent, self).__init__()
