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
import pandas as pd
import shutil

pocket_path = './data_crossdocked/test.yaml'           # './data_crossdocked/test.yaml'
ori_vina_path = '/home/nic/Code/HGNN-GPT/GPT-last-new-2/crossdocked/dock_result2/pocket_vina.csv'
json_file4='./dock_file_save/crossdocked/2025_01_12_16_1743065493/dock_result/dock_dict.json'
json_file3='./dock_file_save/crossdocked/2025_01_05_20_1741635931/dock_result/dock_dict.json'

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

with open(json_file3, 'r') as f:
    dock_data3 = json.load(f)

with open(json_file4, 'r') as f:
    dock_data4 = json.load(f)


# def get_pocket_results(dock_data,pocket_names,ori_vina):

#     affinity_values = {}
#     for key, values in dock_data.items():
#         for record in values:
#             if record.get('mode_id') == 0:
#                 affinity_values[key] = record.get('affinity', None)
#                 break
    
#     pocket_dock_values={}
#     for key,value in affinity_values.items():
#         pocket_name = "_".join(key.split("_")[:-1])
#         if pocket_name not in pocket_dock_values:
#             pocket_dock_values[pocket_name]=[]
#         pocket_dock_values[pocket_name].append((key,value))

#     pocket_results = {}
#     for item in pocket_names:
#         high_num = 0
#         affinity_all = 0
#         pocket_name = re.search(r'([^/]+_pocket\d+)\.pdb$', item).group(1)
#         dock_values_all = pocket_dock_values[pocket_name]
#         for key, value in dock_values_all:
#             if value <= ori_vina[item]:
#                 high_num = high_num + 1
#             affinity_all = affinity_all + value

#         avg_affinity=float(affinity_all/len(dock_values_all))
#         lower_ratio=float(high_num/len(dock_values_all))
#         pocket_results[item] = {
#                 'avg_affinity': avg_affinity,
#                 'lower_ratio': lower_ratio,
#                 'original_affinity': ori_vina[item]
#             }
        
#     df_results = pd.DataFrame.from_dict(pocket_results, orient='index')

#     return df_results

# data_3=get_pocket_results(dock_data3,pocket_names,ori_vina)
# data_4=get_pocket_results(dock_data4,pocket_names,ori_vina)

# data_3.to_csv('pocket_results3.csv', index_label='pocket_name')
# data_4.to_csv('pocket_results4.csv', index_label='pocket_name')


# one_pocket_name='1phk_A_rec_1phk_atp_lig_tt_min_0_pocket10'
# one_pocket_name='3tym_A_rec_3n5v_xfh_lig_tt_min_0_pocket10'
# one_pocket_name='5liu_X_rec_4gq0_qap_lig_tt_min_0_pocket10'
# one_pocket_name='3nfb_A_rec_3nfb_oae_lig_tt_docked_2_pocket10'
# one_pocket_name='4azf_A_rec_5lxc_7aa_lig_tt_min_0_pocket10'   
# one_pocket_name='14gs_A_rec_20gs_cbd_lig_tt_min_0_pocket10'
one_pocket_name='1ai4_A_rec_1ai5_mnp_lig_tt_docked_0_pocket10' # 


def get_top_10(one_pocket_name,dock_data):
    affinity_values = {}
    for key, values in dock_data.items():
        for record in values:
            if record.get('mode_id') == 0:
                affinity_values[key] = record.get('affinity', None)
                break
    
    pocket_dock_values={}
    for key,value in affinity_values.items():
        pocket_name = "_".join(key.split("_")[:-1])
        if pocket_name not in pocket_dock_values:
            pocket_dock_values[pocket_name]=[]
        pocket_dock_values[pocket_name].append((key,value))

    one_pocket_affinity = pocket_dock_values[one_pocket_name]
    top_10_values = sorted(one_pocket_affinity, key=lambda x: x[1])[:10]
    # top_10_affinities = [value for _, value in top_10_values]
    # return top_10_affinities
    return top_10_values


top_10_values_3=get_top_10(one_pocket_name,dock_data3)
print(top_10_values_3)
top_10_values_4=get_top_10(one_pocket_name,dock_data4)
print(top_10_values_4)

gen_smiles_3={}
gen_smiles_4={}

def get_smi(config_dir):
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)
    return list(config.keys())

pdbqt_path_3='./dock_file_save/crossdocked/2025_01_05_20_1741635931/smiles_pdbqt/'
smiles_yaml_3='./save/pre/crossdocked/char/hgnn/2025_01_05_20/sample_300_30_True_1_1_1741635931/'
save_path_3='./best_pocket/chapter_3/'
pocket_smiles_path = os.path.join(smiles_yaml_3, one_pocket_name)+ '_sampled_temp1.yaml'
smiles_all = get_smi(pocket_smiles_path)


for key,value in top_10_values_3:
    one_pdbqt_path = os.path.join(pdbqt_path_3,key) + '.pdbqt'
    destination_file=os.path.join(save_path_3,key) + '.pdbqt'
    shutil.copy(one_pdbqt_path, destination_file)
    
    parts = key.rsplit('_', 1)
    smiles_index = int(parts[1])
    one_smiles=smiles_all[smiles_index]

    gen_smiles_3[key] = {
        'smiles': one_smiles,
        'affinity_value': value
    }
    
df_results = pd.DataFrame.from_dict(gen_smiles_3, orient='key')

csv_file_path = os.path.join(save_path_3,'gen_smiles_3.csv')
df_results.to_csv(csv_file_path)



pdbqt_path_4='./dock_file_save/crossdocked/2025_01_12_16_1743065493/smiles_pdbqt/'