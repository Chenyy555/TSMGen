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
import csv

json_file='./data_pdbbind/dock/dock_protein_center/dock_dict.json'

output_csv_file='./data_pdbbind/dock/dock_protein_center/pocket_vina.csv'

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

with open(output_csv_file, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['ligand_name', 'affinity'])
    for key, affinity in affinity_values.items():
        csv_writer.writerow([key, affinity])