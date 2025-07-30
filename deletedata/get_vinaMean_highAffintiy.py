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
    pocket_path = './data_pdbbind/test_87.yaml'
    # pocket_path = './data_pdbbind/test_64.yaml'
    ori_vina_path = './data_pdbbind/dock/dock_protein_center/pocket_vina.csv'
else:
    pocket_path = './data_crossdocked/test.yaml'           # './data_crossdocked/test.yaml'
    ori_vina_path = '/home/nic/Code/HGNN-GPT/GPT-last-new-2/crossdocked/dock_result/pocket_vina.csv'


ori_vina = {}
with open(ori_vina_path, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        ligand_name = row['pocket_name']
        affinity = float(row['affinity'])
        ori_vina[ligand_name] = affinity

with open(json_file, 'r') as f:
    dock_data = json.load(f)

with open(pocket_path, 'r') as f:
    pocket_dict = yaml.full_load(f)

affinity_values = {}
for key, values in dock_data.items():
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


pocket_dock_values = defaultdict(list)
for key, affinity in affinity_values.items():
    pocket_key = re.sub(r'_\d+$', '', key) 
    pocket_dock_values[pocket_key].append((key, affinity))

all_molecules_affinities = {}
for pocket, molecules in pocket_dock_values.items():
    sorted_molecules = sorted(molecules, key=lambda x: x[1])[:10]
    for molecule, affinity in molecules:
        all_molecules_affinities[molecule] = affinity

print(sum(all_molecules_affinities.values()) / len(all_molecules_affinities))

high_num =0
for molecule, affinity in all_molecules_affinities.items():
    pocket_name = re.sub(r'_\d+$', '', molecule)
    ori_ligand = pocket_name[:-9]
    ori_ligand_vina = ori_vina[ori_ligand]
    if affinity < ori_ligand_vina:
        high_num = high_num + 1

print(high_num)
print(high_num / len(all_molecules_affinities))

pocket_27 = []
for key,value in pocket_dict.items():
    pocket_item = os.path.splitext(os.path.basename(key))[0]
    pocket_27.append(pocket_item)



all_molecules_affinities_27 = {}
for molecule, affinity in all_molecules_affinities.items():
    pocket_name = re.sub(r'_\d+$', '', molecule)
    if pocket_name in pocket_27:
        all_molecules_affinities_27[molecule] = affinity

high_num2=0
for molecule, affinity in all_molecules_affinities_27.items():
    pocket_name = re.sub(r'_\d+$', '', molecule)
    ori_ligand = pocket_name[:-9]
    ori_ligand_vina = ori_vina[ori_ligand]
    if affinity < ori_ligand_vina:
        high_num2 = high_num2 + 1

print(sum(all_molecules_affinities_27.values()) / len(all_molecules_affinities_27))
print(high_num2)
print(high_num2 / len(all_molecules_affinities_27))

