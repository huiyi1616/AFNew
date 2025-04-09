# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:55:31 2025

@author: hw1616
"""

# config/train_config.py

import torch

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




