# COMP6714 Project
import numpy as np
import random
import torch

# you can change the random seed (or even disable it) when experimenting your implementation
# it is just for the course evaluating purpose
def apply_random_seed():
    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)