import os
import random

import torch
import torchvision.transforms as v2
import torchvision.transforms.functional as F
import yaml
from requests.packages import target
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import Dataset

class ProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, tensor):
        return self.processor(tensor, return_tensors="pt")["pixel_values"].squeeze(0)

