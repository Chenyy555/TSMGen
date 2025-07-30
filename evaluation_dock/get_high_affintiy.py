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
from collections import defaultdict
import statistics

config_dir= './evaluation_dock/high_affinity.yaml'
with open(config_dir, 'r') as f:
    config = yaml.full_load(f)

json_file = config['dock_dict_json']
dataset = config['dataset']

pocket_vina_path = config['pocket_vina_path']
final_result_path = config['final_result_path']


if dataset == 'pdbbind':
    pocket_path = './data_pdbbind/test_64.yaml'
    ori_vina_path = '/home/nic/Code/HGNN-GPT/GPT-last-new-2/pdbbind/dock_result/pocket_vina.csv'
else:
    pocket_path = './data_crossdocked/test.yaml'           # './data_crossdocked/test.yaml'
    ori_vina_path = '/home/nic/Code/HGNN-GPT/GPT-last-new-2/crossdocked/dock_result2/pocket_vina.csv'

with open(pocket_path, 'r') as f:
    pocket_dict = yaml.full_load(f)
pocket_names=list(pocket_dict.keys())


ori_vina = {}
with open(ori_vina_path, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        ligand_name = row['pocket_name']
        affinity = float(row['affinity'])
        ori_vina[ligand_name] = affinity


with open(json_file, 'r') as f:
    dock_data = json.load(f)

affinity_values = {}
for key, values in dock_data.items():
    for record in values:
        if record.get('mode_id') == 0:
            affinity_values[key] = record.get('affinity', None)
            break

affinity_list = [value for value in affinity_values.values() if value is not None]

print(affinity_list)

if affinity_list:
    average_affinity = sum(affinity_list) / len(affinity_list)
    print(f"Average Affinity: {average_affinity:.4f}")
else:
    print("No valid affinity values found.")


high_num =0
for key, affinity in affinity_values.items():
    pocket_name = re.sub(r'_\d$', '', key) + '.pdb'
    for key, value in ori_vina.items():
        if pocket_name in key:
            ori_affinity = value
            break
    if affinity <= ori_affinity:
        high_num = high_num + 1

print(high_num / len(affinity_list))


