import os.path as osp
import os
from rdkit import Chem
import numpy as np
from easydict import EasyDict
from rdkit import Chem
import subprocess
from rdkit.Chem.rdMolAlign import CalcRMS
import shutil
import re


def prepare_ligand_obabel(work_dir, ligand, out_lig_name=None, mode='allH'):

    lig_pdbqt = work_dir.replace('sdf','pdbqt')
    # lig_pdbqt = work_dir.replace('sdf','pdb')
    if mode == 'ph':
        command = 'obabel {lig_sdf} -O {lig_pdbqt} -p 7.4'.format(lig_sdf=ligand, lig_pdbqt=lig_pdbqt)
        #command = 'prepare_ligand -l {lig_sdf} -o {lig_pdbqt} -A hydrogens'.format(lig_sdf=lig_sdf, lig_pdbqt=lig_pdbqt)
    elif mode == 'noh':
        command = 'obabel {lig_sdf} -O {lig_pdbqt}'.format(lig_sdf=ligand, lig_pdbqt=lig_pdbqt)
    elif mode == 'allH':
        # mol = Chem.SDMolSupplier(ligand)[0]
        # ligand_allH = ligand[:-4]+'_allH.sdf'
        # # write_sdf(mol, ligand_allH)
        # ligand = mol
        command = 'obabel {lig_sdf} -O {lig_pdbqt}'.format(lig_sdf=ligand, lig_pdbqt=lig_pdbqt)
    
    proc = subprocess.Popen(
                command, 
                shell=True, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
                )
    proc.communicate() 
    return out_lig_name


ori_path='./data_pdbbind/dock/ori_ligand/'
save_path ='./data_pdbbind/dock/ligand_pdbqt/'


for file_name in os.listdir(ori_path):

    file_path = os.path.join(ori_path, file_name)
    each_path = os.path.join(save_path, file_name)
    prepare_ligand_obabel(each_path,file_path)


