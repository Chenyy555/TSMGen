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



def scoring_with_sdf(protein_pdbqt, lig_pdbqt, centroid, out_lig_sdf=None):
    '''
    work_dir: is same as the prepare_target
    protein_pdbqt: .pdbqt file
    lig_sdf: ligand .sdf format file
    '''

    cx, cy, cz = centroid

    command = '''/home/nic/qvina/qvina2.1 \
        --receptor {receptor_pre} \
        --ligand {ligand_pre} \
        --center_x {centroid_x:.4f} \
        --center_y {centroid_y:.4f} \
        --center_z {centroid_z:.4f} \
        --size_x 40 --size_y 40 --size_z 40 \
        --exhaustiveness {exhaust} \
        --score_only'''.format(receptor_pre = protein_pdbqt,
                                            ligand_pre = lig_pdbqt,
                                            centroid_x = cx,
                                            centroid_y = cy,
                                            centroid_z = cz,
                                            exhaust = 32)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    p = proc.communicate()[0]
    c = p.decode("gbk").strip()
    score = re.search("\nAffinity:(.*)\n", c).group().strip().split()[1]

    return float(score)


config_dir='./evaluation_dock/dock.yaml'
with open(config_dir, 'r') as f:
    config = yaml.full_load(f)

dataset = config['dataset']
ligand_pdbqt=config['smiles_pdbqt']
out_path_sdf=config['out_path_sdf']
out_path_pdbqt=config['out_path_pdbqt']

if dataset == 'pdbbind':
    pocket_path = './data_pdbbind/test_87.yaml'
    # pocket_path = './data_pdbbind/test_64.yaml'
    pocket_pdbqt = '/home/nic/Code/HGNN-GPT/GPT-last-new-2/pdbbind/pocket_pdbqt/'
else:
    pocket_path = './data_crossdocked/test.yaml'
    pocket_pdbqt = '/home/nic/Code/HGNN-GPT/GPT-last-new-2/crossdocked/pocket_pdbqt/'


with open(pocket_path, 'r') as f:
    pocket_dict = yaml.full_load(f)

pocket_names=list(pocket_dict.keys())


dock_dict={}
dock_scoring_dict={}
error_dock=[]

save_prop_path = config['save_prop_path']
os.makedirs(save_prop_path, exist_ok=True)


for item in pocket_names:
    
    if dataset == 'crossdocked':
        receptor_name = os.path.splitext(os.path.basename(item))[0]
        receptor_dir = os.path.join(pocket_pdbqt, receptor_name) + '.pdbqt'
        # ligand_name = receptor_name[:-9]
    else:
        receptor_dir = os.path.join(pocket_pdbqt, item) + '.pdbqt'
        receptor_name = item
        # ligand_name = item

    matching_ligands = []
    num = 0
    for ligand_file in os.listdir(ligand_pdbqt):
        if num > 100:
            break
        if receptor_name in ligand_file:
            matching_ligands.append(ligand_file)
            num = num + 1

    centroid=calculate_center(receptor_dir)

    for ligand in matching_ligands:

        ligand_dir = os.path.join(ligand_pdbqt, ligand)
        ligand_name= os.path.splitext(ligand)[0]
        docked_sdf = docking_with_sdf(receptor_dir,ligand_dir,centroid,out_lig_sdf=out_path_sdf,save_pdbqt=out_path_pdbqt)

        try:
            result = get_result(docked_sdf)
            dock_dict[ligand_name] = result
            score = scoring_with_sdf(receptor_dir,ligand_dir,centroid)
            dock_scoring_dict[ligand_name] = score
        except:
            error_dock.append(ligand_name)
    
    with open(save_prop_path + 'dock_dict.json', 'w') as f:
        json.dump(dock_dict, f, indent=4)


with open(save_prop_path + 'dock_dict.json', 'w') as f:
    json.dump(dock_dict, f, indent=4)

with open(save_prop_path + 'dock_scoring_dict.json', 'w') as f:
    json.dump(dock_scoring_dict, f, indent=4)  

with open(save_prop_path + 'error_dock.json', 'w') as f:
    json.dump(error_dock, f, indent=4)  

with open(config['save_config'] +'dock.yaml', 'w') as f:
    yaml.dump(config, f)