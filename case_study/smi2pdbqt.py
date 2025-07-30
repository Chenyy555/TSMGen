from openbabel import pybel
import os
import subprocess
import yaml

def smi_pdb(smi,save_path):
    try:
        mol = pybel.readstring("smi", smi)
        mol.OBMol.StripSalts(10)
        mols = mol.OBMol.Separate()

        # print(pybel.Molecule(mols))

        mol = pybel.Molecule(mols[0])
        for imol in mols:
            imol = pybel.Molecule(imol)
            if len(imol.atoms) > len(mol.atoms):
                mol = imol

        mol.addh()
        mol.make3D(forcefield='mmff94', steps=100)
        mol.localopt()
        mol.write(format='pdb', filename=str(save_path), overwrite=True)
        return 1
    except:
        print(f"Tranformation of {smi} failed! ")
        return 0


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

def get_smi(config_dir):
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)
    return list(config.keys())



openbable_dir= './case_study/smi2pdbqt.yaml'
with open(openbable_dir, 'r') as f:
    config = yaml.full_load(f)


pocket_names = [config['pocket_name']]
smiles_yaml = config['smiles_yaml']
save_smiles_pdb = config['smiles_pdb']
save_smiles_pdbqt = config['smiles_pdbqt']

if not os.path.exists(save_smiles_pdb):
    os.makedirs(save_smiles_pdb)

if not os.path.exists(save_smiles_pdbqt):
    os.makedirs(save_smiles_pdbqt)

list_error=[]
error_2pdbqt = []

for index, pocket_item in enumerate(pocket_names):

    pocket_smiles_path = os.path.join(smiles_yaml, pocket_item)+ '_sampled_temp1.yaml'
    smiles = get_smi(pocket_smiles_path)

    for index, smile in enumerate(smiles):

        each_save_pdb = os.path.join(save_smiles_pdb, pocket_item) +'_' +str(index) + '.pdb'
        each_save_pdbqt = os.path.join(save_smiles_pdbqt, pocket_item) +'_' +str(index) + '.pdbqt'
        result = smi_pdb(smile, each_save_pdb)
        if result==0:
            error_item = pocket_item + '_' + str(index)
            list_error.append(error_item)
        else:
            result_pdbqt = pdb_to_pdbqt(each_save_pdb,each_save_pdbqt)
            if not result:
                error_item = pocket_item + '_' + str(index)
                error_2pdbqt.append(error_item)

save_error_list = config['error_list']
with open(save_error_list +'error_2pdb.yaml', 'w') as f:
    yaml.dump(list_error, f)

with open(save_error_list +'error_2pdbqt.yaml', 'w') as f:
    yaml.dump(error_2pdbqt, f)

with open(save_error_list +'smi2pdbqt.yaml', 'w') as f:
    yaml.dump(config, f)
