import tensorflow as tf
import numpy as np
import time

from MANN.Model import memory_augmented_neural_network
from MANN.Utils.Generator import OmniglotGenerator
from MANN.Utils.Metrics import accuracy_instance

def omniglot():
    return