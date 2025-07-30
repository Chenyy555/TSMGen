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

warnings.filterwarnings("ignore", message="Unused variable")

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


dock_batch_dir= './case_study/dock_batch.yaml'
with open(dock_batch_dir, 'r') as f:
    config = yaml.full_load(f)

receptor_path = config['receptor_path']
ligand_dir = config['ligand_dir']
save_prop_path = config['save_prop_path']
out_path_pdbqt = config['out_path_pdbqt']
out_path_sdf= config['out_path_sdf']

# receptor_path = "./case_study/7DPU/7dpuQU.pdbqt"

# ligand_dir = "./case_study/7DPU/dock_file_save/2258692459/smiles_pdbqt/"
# save_prop_path='./case_study/7DPU/dock_file_save/2258692459/'

centroid = calculate_center(receptor_path)
# out_path_pdbqt='./case_study/7DPU/dock_file_save/2258692459/out_pdbqt'
# out_path_sdf='./case_study/7DPU/dock_file_save/2258692459/out_sdf'
dock_dict={}
error_dock=[]

for ligand_file in os.listdir(ligand_dir):
    # each_save_pdbqt=''
    ligand_file_all=os.path.join(ligand_dir, ligand_file)
    docked_sdf = docking_with_sdf(receptor_path,ligand_file_all,centroid,out_lig_sdf=out_path_sdf,save_pdbqt=out_path_pdbqt)
    try:
        result = get_result(docked_sdf)
        dock_dict[ligand_file]=result
    except:
        error_dock.append(ligand_file)
print(error_dock)

with open(save_prop_path + 'dock_dict.json', 'w') as f:
    json.dump(dock_dict, f, indent=4)




