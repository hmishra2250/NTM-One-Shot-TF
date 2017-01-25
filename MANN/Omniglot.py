import tensorflow as tf
import numpy as np
import time

from .Model import memory_augmented_neural_network
from .Utils.Generator import OmniglotGenerator
from .Utils.Metrics import accuracy_instance

def omniglot():
    return