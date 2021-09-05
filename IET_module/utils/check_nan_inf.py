#!/usr/bin/env python3
import torch
from typing import List

def assert_nan_inf(input_tensors: List[torch.Tensor]) -> None:
    for tensor in input_tensors:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"nan or inf found in: {tensor}")
            raise Exception("found nan or inf")