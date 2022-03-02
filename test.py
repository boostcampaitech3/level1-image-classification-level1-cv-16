import pandas as pd
import torch
import numpy as np

first_model_path="../output/Eb3_mask_only_lr*2_best_acc.csv"
second_model_path="../output/Eb3_GenderAge__best_acc.csv"

first_model = pd.read_csv(first_model_path)
second_model = pd.read_csv(second_model_path)
result = second_model["ans"] + first_model["ans"]*6
first_model["ans"] = result

first_model.to_csv("../output/버전이름/output.csv")