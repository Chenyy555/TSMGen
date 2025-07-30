import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from rdkit import Chem
import ipdb
from rdkit import DataStructs
import numpy as np
import threading
import selfies as sf
import yaml
from copy import deepcopy
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def check_novelty(gen_smiles, train_smiles): 
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]  
        novel = len(gen_smiles) - sum(duplicates)  
        novel_ratio = novel*100./len(gen_smiles)  
    print("novelty: {:.3f}%".format(novel_ratio))
    return novel_ratio

def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp:=Crippen.MolLogP(mol)>=-2) & (logp<=5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def calculate_diversity(pocket_mols):
    if len(pocket_mols) < 2:
        return 0.0

    div = 0
    total = 0
    for i in range(len(pocket_mols)):
        for j in range(i + 1, len(pocket_mols)):
            div += 1 - tanimoto_sim(pocket_mols[i], pocket_mols[j])
            total += 1
    return div / total

def tanimoto_sim(mol, ref):
    # fp1 = Chem.RDKFingerprint(ref)
    # fp2 = Chem.RDKFingerprint(mol)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=1024) 
    fp2 = AllChem.GetMorganFingerprintAsBitVect(ref, 1, nBits=1024)
    return DataStructs.TanimotoSimilarity(fp1, fp2)



def convert_smiles(vocab, vocab_dict,smiles):
    # 
    result = [vocab_dict[key] for key in smiles]
    smile=''
    for value in result:
        if value == '<eos>':
            break  
        elif value == '<pad>':
            smile += ''  
        else:
            smile += value
    
    smile = smile.replace("<sos>", "")
    smile = smile.replace("<eos>", "")
    if vocab == 'selfies':
        # convert SELFIES back to SMILES
        try:
            smile = sf.decoder(smile)
        except:
            print("error")
    return smile

def inverted_dict(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = yaml.full_load(f)
    inverted_vocab = {value: key for key, value in vocab.items()}

    return inverted_vocab


@torch.no_grad()
def sample(model, gnn_data, hgnn_data, x, steps, device, vocab, molecule_num, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in   
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.    
    """ 
    model.eval()

    batch_size = molecule_num   
    finish = torch.zeros(batch_size, dtype=torch.bool).to(device)
    eos_int = vocab.vocab['<eos>']  # Get the index for <eos>

    for k in range(steps):


        lengths=""
        logits= model.sample_from_pocket(gnn_data, hgnn_data, x, lengths)


        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature

        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)


        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)                      


        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)      
        else:
            _, ix = torch.topk(probs, k=1, dim=-1) 


        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)


        eos_sampled = (ix.squeeze() == eos_int)  # Shape: [batch_size]
        finish = torch.logical_or(finish, eos_sampled)


        # If all sequences have finished, stop sampling
        if torch.all(finish):
            break

    return x    
