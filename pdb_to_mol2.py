from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import openbabel as obabel
import os
import subprocess
import yaml

def convertmol2(input_file,output_file):

    try:
        command = ['obabel',input_file,'-O',output_file]
        subprocess.run(command, check=True)
        return True

    except:
        print("error")
        return False


error_list=[]

train_dir='./data_crossdocked/test.yaml'
train_dir='data_pdbbind/pocket-smiles.yaml'
with open(train_dir, 'r') as f:
    train_config = yaml.full_load(f)

pockets = list(train_config.keys())

pocket_dir = '/home/nic/Data/v2020-other-PL/'
# pocket_dir = '/home/nic/Data/crossdock2020/crossdocked_pocket10/'

for pocket in pockets:
    input_dir = pocket_dir + pocket + '/' + pocket + '_pocket.pdb'
    out_dir = os.path.splitext(input_dir)[0] + ".mol2"

    flag = convertmol2(input_dir,out_dir)
    if flag is False:
        error_list.append(pocket)


print(len(error_list))

# obabel -ipdb /home/nic/Data/crossdock2020/crossdocked_pocket10/YUIC_BACSU_56_218_0/4wlk_B_rec_4wlk_3ql_lig_tt_min_0_pocket10.pdb -omol2 ligand
# obabel -ipdb 7d5u_remove_mol.pdb -O 7d5u_remove_mol.mol2