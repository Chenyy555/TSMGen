from openbabel import pybel
import os
import subprocess
import yaml
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

def pdb_to_pdbqt(input_pdb, output_pdbqt):
    try:
        # os.popen(f'/home/nic/.conda/envs/adt/bin/python /home/nic/.conda/envs/adt/bin/prepare_ligand4.py -l {input_pdb} -O {output_pdbqt}')
        venv_python = "/home/nic/.conda/envs/adt/bin/python"

        command = [
            venv_python,
            "/home/nic/.conda/envs/adt/bin/prepare_ligand4.py",
            "-l", input_pdb,
            "-o", output_pdbqt
        ]

        subprocess.run(command, check=True)
        return True
    except:
        print(f"Tranformation of {input_pdb} failed! ")
        return False

def docking_with_sdf(protein_pdbqt, lig_pdbqt, centroid, verbose=1, out_lig_sdf=None, save_pdbqt=False):
    '''
    work_dir: is same as the prepare_target
    protein_pdbqt: .pdbqt file
    lig_sdf: ligand .sdf format file
    '''

    os.makedirs(save_pdbqt, exist_ok=True)
    os.makedirs(out_lig_sdf, exist_ok=True)

    cx, cy, cz = centroid

    out_lig_pdbqt = os.path.splitext(os.path.basename(lig_pdbqt))[0] + '_out.pdbqt'
    out_lig_pdbqt = os.path.join(save_pdbqt, out_lig_pdbqt)

    out_sdf_name = os.path.splitext(os.path.basename(lig_pdbqt))[0] + '_out.sdf'
    out_lig_sdf = os.path.join(out_lig_sdf, out_sdf_name)


    command = '''/home/nic/qvina/qvina2.1 \
        --receptor {receptor_pre} \
        --ligand {ligand_pre} \
        --center_x {centroid_x:.4f} \
        --center_y {centroid_y:.4f} \
        --center_z {centroid_z:.4f} \
        --size_x 50 --size_y 50 --size_z 50 \
        --out {out_lig_pdbqt} \
        --exhaustiveness {exhaust}
        obabel {out_lig_pdbqt} -O {out_lig_sdf} -h'''.format(receptor_pre = protein_pdbqt,
                                            ligand_pre = lig_pdbqt,
                                            centroid_x = cx,
                                            centroid_y = cy,
                                            centroid_z = cz,
                                            out_lig_pdbqt = out_lig_pdbqt,
                                            exhaust = 24,
                                            out_lig_sdf = out_lig_sdf)
    
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
        )
    proc.communicate()

    if not save_pdbqt:
        os.remove(out_lig_pdbqt)
    
    if verbose: 
        if os.path.exists(out_lig_sdf):
            print('searchable docking is finished successfully')
        else:
            print('docing failed')

    return out_lig_sdf

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

def calculate_center(pdbqt_file):
    parser = PDBParser()
    structure = parser.get_structure("pdbqt", pdbqt_file)

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
    coords = np.array(coords)
    center_of_mass = np.mean(coords, axis=0)
    center_of_mass = center_of_mass.astype(float)
    return center_of_mass




# 7d5u
config_dir= './case_study/ligand_pdbqt_dock.yaml'
with open(config_dir, 'r') as f:
    config = yaml.full_load(f)

receptor_path = config['receptor_path']
ligand_pdb = config['ligand_pdb']
ligand_pdbqt = config['ligand_pdbqt']
save_dir = config['save_dir']
out_path_sdf = config['save_dir']
out_path_pdbqt = config['save_dir']

result_pdbqt = pdb_to_pdbqt(ligand_pdb,ligand_pdbqt)
centroid = calculate_center(receptor_path)
docked_sdf = docking_with_sdf(receptor_path,ligand_pdbqt,centroid,out_lig_sdf=out_path_sdf,save_pdbqt=out_path_pdbqt)

result = get_result(docked_sdf)


print(result)

