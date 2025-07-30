import os
import subprocess
import numpy as np
from Bio.PDB.PDBParser import PDBParser
import warnings
import yaml
import glob
from rdkit import Chem
from rdkit.Chem.rdMolAlign import CalcRMS
from easydict import EasyDict
import json
import re

# json_file='./dock_file_save/crossdocked/2024_12_27_21_702043448/dock_result/dock_dict.json'
# json_file='./dock_file_save/pdbbind/2025_01_09_08_1393007167/dock_result/dock_dict.json'
json_file='./dock_file_save/crossdocked/2025_01_06_09_1410992238/dock_result/dock_dict.json'


dict_path='./smiles_pdb/crossdocked/2024_12_01_21_2231507340/dock_protein_center/dock_'

try:
    with open(json_file, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: File {json_file} not found.")
    data = {}

affinity_values = {}
for key, values in data.items():
    
    for record in values:
        if record.get('mode_id') == 0:
            affinity_values[key] = record.get('affinity', None)  
            break

affinity_list = [value for value in affinity_values.values() if value is not None]

if affinity_list:
    average_affinity = sum(affinity_list) / len(affinity_list)
    print(f"Average Affinity: {average_affinity:.4f}")
else:
    print("No valid affinity values found.")



