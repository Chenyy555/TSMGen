from Bio import PDB
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
import random
import xgi
import utils.hypergraph_util as hgut

def convert_to_edge_index(hyperedges):
    edge_index = []

    for i in range(hyperedges.shape[0]):
        connected_nodes = np.where(hyperedges[i] > 0)[0]
        
        for node in connected_nodes:
            edge_index.append((i, node)) 

    return torch.tensor(edge_index, dtype=torch.long).t()  


def extract_amino_acids(structure,pocket_path):
    amino_acids = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.Polypeptide.is_aa(residue):
                    if residue.id[0] == ' ':
                        try:
                            amino_acids[(residue.parent.id, residue.id)] = [residue, residue['CA'].get_coord()] 
                        except:
                            print(pocket_path)
    return amino_acids

def build_vector_features(amino_acids,pdb_file_path):
   
    model = list(amino_acids.items())[0][1][0].parent.parent
   
    dssp = PDB.DSSP(model, pdb_file_path, dssp='/tools/bin/mkdssp')

    vertex_feature_vectors = {}
    second_structure_labels = "HBEGITS-"

    amino_acid_type_labels = "ACDEFGHIKLMNPQRSTVWY"
    

    chemical_element_statistics_labels = ["benzene_ring", "hydroxyl", "thiol", "phosphate"]

    for amino_acid_id, amino_acid in amino_acids.items():
        amino_acid_features = dssp[amino_acid_id]
        second_structure = np.zeros((1, 8))
        second_structure[0, second_structure_labels.index(amino_acid_features[2])] = 1
        spatial_features = np.array([[amino_acid_features[3], amino_acid_features[4], amino_acid_features[5]]])
        amino_acid_type = np.zeros((1, 20))
        amino_acid_type[0, amino_acid_type_labels.index(PDB.Polypeptide.three_to_one(amino_acid[0].resname))] = 1
        vertex_feature_vectors[amino_acid_id] = np.hstack((second_structure, spatial_features, amino_acid_type))
    
    return vertex_feature_vectors


def build_first_hyperedges(amino_acids, vertex_feature_vectors, k_neig, threshold):
    n = len(amino_acids)
    adjacency_matrix = None


    space_structure_hyperedges = np.zeros((n, n))
    distances = np.zeros((n, n))   
    sequence_structure_hyperedges = np.zeros((n, n))
    for i, amino_acid_i in enumerate(amino_acids):
        for j, amino_acid_j in enumerate(amino_acids):  
            distances[i][j] = np.linalg.norm(np.array(amino_acids[amino_acid_i][1]) - np.array(amino_acids[amino_acid_j][1]))
            if distances[i][j] < threshold:  
                space_structure_hyperedges[j][i] =1
    adjacency_matrix = space_structure_hyperedges
    

    for center_idx in range(n):
        distances[center_idx, center_idx] = 0
        dis_vec = distances[center_idx]                
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze() 
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx   

        for node_idx in nearest_idx[:k_neig]:
            #sequence_structure_hyperedges[node_idx, center_idx] = list(vertex_feature_vectors[(amino_acids[amino_acid_j][0].parent.id, amino_acids[amino_acid_j][0].id)][0]) 
            sequence_structure_hyperedges[node_idx, center_idx] = 1

    adjacency_matrix = np.hstack((adjacency_matrix, sequence_structure_hyperedges))

    return space_structure_hyperedges, sequence_structure_hyperedges, adjacency_matrix

def obtian_edge(adjacency_matrix):
    adjacency_matrix = torch.from_numpy(adjacency_matrix)
    row_indices, col_indices = torch.nonzero(adjacency_matrix, as_tuple=True)
    edge_index = torch.stack([row_indices, col_indices], dim=0) 
    edge_index = torch.tensor(edge_index,dtype=torch.long)
    return edge_index


def pocket_hypergraph(pocket_path, threshold):
    protein_structure = PDB.PDBParser().get_structure("protein_structure", pocket_path)
    amino_acids=extract_amino_acids(protein_structure,pocket_path)

    vertex_feature_vectors = build_vector_features(amino_acids, pocket_path)
    node_features=torch.tensor(list(vertex_feature_vectors.values()),dtype=torch.float32).squeeze()
    k_neig=5
    space_edge, sequence_edge, first_edge= build_first_hyperedges(amino_acids, vertex_feature_vectors, k_neig, threshold)
    # edge_index_space=space_edge.nonzero().t()
    # edge_index_sequence=sequence_edge.nonzero().t()
    # edge_index_first=first_edge.nonzero().t()

    edge_index_space=obtian_edge(space_edge)
    # edge_index_space=space_edge
    edge_index_sequence=obtian_edge(sequence_edge)
    edge_index_first=obtian_edge(first_edge)
    # G,hyper_edge = hgut.generate_G_from_H(space_edge)

    return node_features, edge_index_space, edge_index_sequence, edge_index_first
