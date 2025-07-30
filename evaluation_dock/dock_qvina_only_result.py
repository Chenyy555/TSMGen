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



def get_result(docked_sdf, ref_mol=None):
    suppl = Chem.SDMolSupplier(docked_sdf,sanitize=False)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        try:
            rmsd = CalcRMS(ref_mol, mol)
        except:
            rmsd = np.nan
        results.append(EasyDict({
            # 'rdmol': mol,
            'mode_id': i,
            'affinity': float(line[0]),
            'rmsd_lb': float(line[1]),
            'rmsd_ub': float(line[2]),
            # 'rmsd_ref': rmsd
        }))

    return results


dataset = 'crossdocked'

if dataset == 'pdbbind':
    pocket_path = './data_pdbbind/test_87.yaml'
    # pocket_path = './data_pdbbind/test_64.yaml'
    pocket_pdbqt = './data_pdbbind/dock/pocket_pdbqt/'
else:
    pocket_path = './data_crossdocked/final_test.yaml'
    pocket_pdbqt = '/home/nic/Code/HGNN-GPT/GPT-last-new-2/crossdocked/pocket_pdbqt/'


out_sdf='./dock_file_save/crossdocked/2024_12_27_21_702043448/out_sdf/'
save_prop_path = './dock_file_save/crossdocked/2024_12_01_21_2231507340/dock_result/'
os.makedirs(save_prop_path, exist_ok=True)


with open(pocket_path, 'r') as f:
    pocket_dict = yaml.full_load(f)


pocket_names=list(pocket_dict.keys())


dock_dict={}
dock_scoring_dict={}
error_dock=[]

for file_name in os.listdir(out_sdf):
    file_path = os.path.join(out_sdf, file_name)

    pocket_ligand_name = file_name[:-8]
    pocket_name = re.sub(r'_\d$', '', pocket_ligand_name) + '.pdb'
    exists = any(pocket_name in item for item in pocket_names)
    if exists:
        result = get_result(file_path)
        dock_dict[pocket_ligand_name] = result


with open(save_prop_path + 'dock_dict2.json', 'w') as f:
    json.dump(dock_dict, f, indent=4)  


