import os
import sys
import glob
import pandas as pd
import yaml
import subprocess

def pdb_to_pdbqt(pdb_file, pdbqt_file):
    try:
        venv_python = "/home/nic/.conda/envs/adt/bin/python"

        command = [
            venv_python,
            "/home/nic/.conda/envs/adt/bin/prepare_receptor4.py",
            "-r", pdb_file,
            "-o", pdbqt_file
        ]
        subprocess.run(command, check=True)
        return True
    except:
        print(f"Tranformation of {pdb_file} failed! ")
        return False
        


receptor_dir='./data_pdbbind/dock/test_set.yaml'
pocket_pdb_dir = './data_pdbbind/dock/pocket_pdb'
pocket_pdbqt_dir = './data_pdbbind/dock/pocket_pdbqt/'

if not os.path.exists(pocket_pdbqt_dir):
    os.makedirs(pocket_pdbqt_dir)
error_pocket2pdbqt =[]

with open(receptor_dir, 'r') as f:
    pocket_dict = yaml.full_load(f)

pocket_name = list(pocket_dict.keys())

for index, pocket_item in enumerate(pocket_name):
    
    pocket_item_dir = os.path.join(pocket_pdb_dir, pocket_item)+'_pocket.pdb'

    pocket_item_pdbqt = os.path.join(pocket_pdbqt_dir, pocket_item)+'_pocket.pdbqt'

    result = pdb_to_pdbqt(pocket_item_dir,pocket_item_pdbqt)

    if not result:
        error_pocket2pdbqt.append(pocket_item)


with open('./data_pdbbind/dock/' +'error_pocket2pdbqt.yaml', 'w') as f:
    yaml.dump(error_pocket2pdbqt, f)
