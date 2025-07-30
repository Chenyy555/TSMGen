import os
import sys
import glob
import pandas as pd
import yaml
import subprocess

def pdb_to_pdbqt(pdb_file, pdbqt_file):
    try:
        # os.popen(f'/home/nic/.conda/envs/adt/bin/python /home/nic/.conda/envs/adt/bin/prepare_ligand4.py -l {input_pdb} -O {output_pdbqt}')
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


pocket_item_dir = './case_study/7DPU/7dpu.pdb'

pocket_item_pdbqt = './case_study/7DPU/7dpu_remove_mol.pdbqt'

pdb_to_pdbqt(pocket_item_dir,pocket_item_pdbqt)