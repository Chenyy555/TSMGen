from Bio.PDB import PDBParser, Polypeptide
from Bio import PDB
import json
import yaml
import os


def extract_sequence():
    pocket_path="/home/nic/Data/crossdock2020/crossdocked_pocket10/"
    path="../final_filtered_train.yaml"

    with open(path, 'r') as f:
        pocket_yaml = yaml.full_load(f)

    pockets = list(pocket_yaml.keys())

    pocket_sequence={}

    for item in pockets:
        pdb_file = item  

        pdb_path = pocket_path + pdb_file

        try:
            full_sequence=extract_amino_acids_sequence(pdb_path)
            pocket_sequence[pdb_file]=full_sequence
        except Exception as e:
            print("error")

    with open('pocket_sequence.yaml', 'w') as file:
        yaml.dump(pocket_sequence, file)

def extract_amino_acids_sequence(pocket_path):
    structure = PDB.PDBParser().get_structure("protein_structure", pocket_path)
    amino_acids = {}
    for model in structure:
        for chain in model:
            sequence = []
            for residue in chain:
                if Polypeptide.is_aa(residue):
                    if residue.id[0] == ' ':
                        try:
                            sequence.append(Polypeptide.three_to_one(residue.get_resname()))
                        except KeyError:
                            print(f"Unknown residue: {residue.get_resname()}")
            
            amino_acids[chain.id] = ''.join(sequence)  
    
    full_sequence = ''

    for chain, sequence in amino_acids.items():
        if sequence:  
            full_sequence += sequence

    return full_sequence

extract_sequence()