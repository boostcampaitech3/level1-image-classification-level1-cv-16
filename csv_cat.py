from bdb import Breakpoint
import pandas as pd
import torch
import numpy as np

first_model_path="../output/final/Eb3_mask_only_offaug/Eb3_mask_only_offaug_best_acc.csv"
second_model_path_mask="../output/final/Eb3_genderage_only_mask_offaug/Eb3_genderage_only_mask_offaug_best_acc.csv"
second_model_path_inc="../output/final/Eb3_genderage_only_inc_offaug/Eb3_genderage_only_inc_offaug_best_acc.csv"
second_model_path_not="../output/final/Eb3_genderage_only_not_offaug/Eb3_genderage_only_not_offaug_best_acc.csv"

first_model = pd.read_csv(first_model_path)
second_model_mask = pd.read_csv(second_model_path_mask)
second_model_inc = pd.read_csv(second_model_path_inc)
second_model_not = pd.read_csv(second_model_path_not)

for i,id in enumerate(first_model['ImageID']):
    if first_model["ans"].iloc[i]==0:
        [second_i]=second_model_mask.index[second_model_mask["ImageID"]==id].tolist()
        first_model["ans"].iloc[i] = second_model_mask["ans"].iloc[second_i] + first_model["ans"].iloc[i]*6
    elif first_model["ans"].iloc[i]==1:
        [second_i]=second_model_inc.index[second_model_inc["ImageID"]==id].tolist()
        first_model["ans"].iloc[i] = second_model_inc["ans"].iloc[second_i] + first_model["ans"].iloc[i]*6
    elif first_model["ans"].iloc[i]==2:
        [second_i]=second_model_not.index[second_model_not["ImageID"]==id].tolist()
        first_model["ans"].iloc[i] = second_model_not["ans"].iloc[second_i] + first_model["ans"].iloc[i]*6
 


first_model.to_csv("../output/final/final_output.csv")
