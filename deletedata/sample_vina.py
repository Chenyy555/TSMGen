from generate.sample_util import check_novelty, sample, canonic_smiles,inverted_dict,convert_smiles,obey_lipinski,calculate_diversity,set_seed
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import RDConfig
import yaml
import os
import sys
sys.path.append('..')
from model.model import Pocket_GNN
from model.decoder import GPTDecoder, GPTConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
from collections import Counter
import time

import metrics.SA_Score.sascorer as sascorer
import metrics.NP_Score.npscorer as npscorer
from dataloaders.dataloader_protein_atom import DatapairDataset
from torch_geometric.loader import DataLoader
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from dataloaders.dataloader_protein_atom import pocket_sequence_gen
import warnings
warnings.filterwarnings("ignore")

def read_config(out_dir, config_name):
    config_dir = out_dir + config_name+".yaml"
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)
    return config


def data_dict(smiles_dir,val_dir,test_dir):
    with open(val_dir + 'val_dict.yaml', "r") as file_b:
        data_val = yaml.safe_load(file_b)
        dict_val = data_val

    with open(smiles_dir, "r") as file_a:
        data_a = yaml.safe_load(file_a)
        dict_train = {k: v for k, v in data_a.items() if k not in dict_val}
    
    with open(test_dir, "r") as file_c:
        data_a = yaml.safe_load(file_c)
        dict_test = data_a

    return dict_val, dict_train, dict_test


def safe_apply(func, molecule):
    try:
        return func(molecule)
    except Exception:
        return np.nan

if __name__ == '__main__':

    sample_config_dir="./config/sample.yaml"
    with open(sample_config_dir, 'r') as f:
        sample_config = yaml.full_load(f)

    seed = int(time.time() * 1000) % (2**32 - 1)

    set_seed(seed)


    top_k=sample_config['top_k']
    result_dir=sample_config['result_dir']
    molecule_nums=sample_config['molecule_nums']   
    temperature=sample_config['temperature']
    sample_num=sample_config['sample_num']
    sample_type= sample_config['sample_type']


    sampled_mols_dir = result_dir + f"sample_{molecule_nums}_{top_k}_{sample_type}_{temperature}_{sample_num}_{seed}/"
    if not os.path.exists(sampled_mols_dir):
        os.makedirs(sampled_mols_dir)



    train_config=read_config(result_dir,"train")
    encoder_config=read_config(result_dir,"encoder")
    decoder_config=read_config(result_dir,"decoder")
    block_size=decoder_config['block_size']


    device = torch.device(
        'cuda:1' if torch.cuda.is_available() else 'cpu'
    )
    print('device: ', device)



    vocab, vocab_path =  train_config['vocab'], train_config['vocab_path']


    model_path = result_dir + "val_model.pt"
    config=GPTConfig(decoder_config['vocab_size'], block_size, num_props=int(decoder_config['num_props']),n_layer=decoder_config['n_layer'],
                     n_embd=decoder_config['n_embd'],n_head=decoder_config['n_head'],att_num=int(decoder_config['att_num']),
                     alpha_use=decoder_config['use_alpha'],use_gate=decoder_config['use_gate'],use_encoder_norm=decoder_config['use_encoder_norm'],
                     sample=True,pretain=decoder_config['pretrain'],use_alpha=decoder_config['use_alpha'])
    model = Pocket_GNN(train_config,encoder_config, config, device).to(device)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device(device)
        )
    )

    smiles_dir = train_config['smiles_dir']
    val_dir = result_dir
    test_dir=sample_config['test_dir']
    val_dict, train_dict, test_dict=data_dict(smiles_dir,val_dir,test_dir)
    val_pockets = list(test_dict.keys())
    pocket_dir=train_config['pocket_dir']
    dataset_type=train_config['dataset']
    num_val_pockets = len(val_pockets)
    features_to_use = encoder_config['features_to_use']
    hgnn_train=encoder_config['encoder_Train']['HGNN']
    gnn_train=encoder_config['encoder_Train']['GNN']

    qed_threshold=0.35
    sa_threshold=0
    if decoder_config['use_gate'] ==True:
        qed_threshold=0.42
    if (hgnn_train and gnn_train) or decoder_config['att_num'] == 2:
        qed_threshold=0.5

    if hgnn_train and gnn_train and decoder_config['att_num'] == 2:
        sa_threshold=0.58
    
    pocket_sequence=pocket_sequence_gen(dataset_type)

    # create a valset of the pockets
    valset = DatapairDataset(
        dataset_type,
        pocket_sequence,
        pockets=val_pockets,
        pocket_dir=pocket_dir,
        smiles_dict=None,
        features_to_use=features_to_use,
        vocab=vocab,
        vocab_path=vocab_path,
        hgnn_train=hgnn_train,
        gnn_train=gnn_train,
    )

    val_loader = DataLoader(
        valset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False
    )

    model.eval()

    inverted_vocab=inverted_dict(vocab_path)   

    valid_list=[]        
    unique_list=[]       
    unique_list_1=[]   
    novelty_list=[]      
    novelty_list_1=[] 
    pocket_DiV=[]        
    SA_list=[]
    QED_list=[]
    logP_list=[]

    SanitizeSmiles_list=[] 
    pocket_metrics = []    
    all_dfs=[]             
    time_list=[]
    pocket_num=0
    t_pocket_start = time.time()
    for i,(gnn_data, hgnn_data) in enumerate(val_loader):
        if pocket_num>100:
                break
        if pocket_num<110:
            gnn_data = gnn_data.to(device)
            hgnn_data=hgnn_data.to(device)
            pocket_name = gnn_data.pocket_name[0]

            print('sampling SMILES for pocket {}...'.format(pocket_name))

            start_int = [key for key, value in valset.vocab.int2tocken.items() if value == '<sos>'][0]

            # create a tensor of shape [batch_size, seq_step=1]
            sos = torch.ones(
                [molecule_nums, 1],
                dtype=torch.long,
                device=device
            )
            sos = sos * start_int
            x = torch.tensor(start_int, dtype=torch.long, device=device)[None,...].repeat(molecule_nums, 1)

            molecules= sample(model, gnn_data, hgnn_data, x, block_size, device, valset.vocab, molecule_nums, temperature=temperature, sample=sample_type, top_k=top_k)

            # a dictionary that stores the frequency of each valid SMILES
            mol_dict = {}
            SanitizeSmiles=[]
            num_invalid=0
            num_valid=0
            mol_list=[]

            for smiles in molecules:
                if num_valid == 100:
                    break
                smiles=convert_smiles(vocab, inverted_vocab, smiles.tolist())
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print('SMILES of None value in sample',smiles)
                    num_invalid += 1
                    continue
                else:
                    # mol_list.append(mol)
                    # Chem.SanitizeMol(mol)
                    # SanitizeSmiles.append(mol)
                    # num_valid += 1
                    # if smiles in mol_dict:
                    #     mol_dict[smiles] += 1
                    # else:
                    #     mol_dict[smiles] = 1

                    if QED.qed(mol)<qed_threshold or round((10-sascorer.calculateScore(mol))/9,2)<sa_threshold:
                        num_invalid += 1
                        continue
                    else:
                        mol_list.append(mol)
                        # Chem.SanitizeMol(mol)
                        SanitizeSmiles.append(mol)
                        num_valid += 1
                        if smiles in mol_dict:
                            mol_dict[smiles] += 1
                        else:
                            mol_dict[smiles] = 1
                    

            time_list.append(float((time.time() - t_pocket_start)/num_valid)*100)

            valid_rate = float(num_valid / (num_valid + num_invalid))
            valid_list.append(valid_rate)
            print("valid rate: {}%".format(valid_rate * 100))


            num_unique = len(list(mol_dict.keys()))
            unique_rate_1 = float(num_unique / (num_valid + num_invalid))
            unique_rate = float(num_unique / num_valid)
            unique_list_1.append(unique_rate_1)
            unique_list.append(unique_rate)
        
            gen_smiles = mol_dict.keys()
            train_smiles=train_dict.values()    
            novelty_rate=len(list(set(gen_smiles) -set(train_smiles))) / len(gen_smiles)
            novelty_list.append(novelty_rate)

            count1 = Counter(gen_smiles)
            count2 = Counter(train_smiles)
            result = sum((count1 - count2).values())
            novelty_list_1.append(result/ len(gen_smiles))

            pocket_DiV.append(calculate_diversity(mol_list))

            pocket_name = os.path.splitext(os.path.basename(pocket_name))[0]
            out_path = sampled_mols_dir + pocket_name + \
                '_sampled_temp{}.yaml'.format(temperature)
            with open(out_path, 'w') as f:
                yaml.dump(mol_dict, f)

            pocket_metrics.append({'pocket_name':pocket_name,'valid': valid_list[i],'unique': unique_list[i],
                                'unique_1': unique_list_1[i],'novelty': novelty_list[i],'novelty_1': novelty_list_1[i],'Diversity ':pocket_DiV[i],'time':time_list[i]})
          

            for i in SanitizeSmiles:
                SanitizeSmiles_list.append({'molecule' : i, 'pocket_name':pocket_name,'smiles': Chem.MolToSmiles(i)})

            pocket_num=pocket_num+1

    results = pd.DataFrame(SanitizeSmiles_list)
    canon_smiles = [canonic_smiles(s) for s in results['smiles']]  
    unique_smiles = list(set(canon_smiles))    
    novel_ratio = check_novelty(unique_smiles, train_smiles)  

    # results['Ro5'] = results['molecule'].apply(lambda x: obey_lipinski(x))
    # results['qed'] = results['molecule'].apply(lambda x: QED.qed(x) )
    # results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
    # results['sas0-1'] = results['molecule'].apply(lambda x: round((10-sascorer.calculateScore(x))/9,2))
    # results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x) )
    # results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x) )
    # results['np'] = results['molecule'].apply(lambda x: npscorer.scoreMol(x) )
    # results['weight'] = results['molecule'].apply(lambda x: Descriptors.MolWt(x) )

    results['Ro5'] = results['molecule'].apply(lambda x: obey_lipinski(x))
    results['qed'] = results['molecule'].apply(lambda x: safe_apply(QED.qed, x))
    results['sas'] = results['molecule'].apply(lambda x: safe_apply(sascorer.calculateScore, x))
    results['sas0-1'] = results['molecule'].apply(lambda x: safe_apply(
        lambda mol: round((10 - sascorer.calculateScore(mol)) / 9, 2), x
    ))
    results['logp'] = results['molecule'].apply(lambda x: safe_apply(Crippen.MolLogP, x))
    results['tpsa'] = results['molecule'].apply(lambda x: safe_apply(CalcTPSA, x))
    results['np'] = results['molecule'].apply(lambda x: safe_apply(npscorer.scoreMol, x))
    results['weight'] = results['molecule'].apply(lambda x: Descriptors.MolWt(x) )
    # results['temperature'] = temp


    nan_rows = results[results.isna().any(axis=1)]
    if not nan_rows.empty:
        nan_rows.to_csv(sampled_mols_dir + 'nan_rows.csv', index=False)
    

    all_lens=num_val_pockets  
    results['validity'] = np.round(len(results)/(molecule_nums*all_lens), 5)
    results['unique'] = np.round(len(unique_smiles)/len(results), 5)   
    results['novelty'] = np.round(novel_ratio/100, 5)
    all_dfs.append(results)
    results = pd.concat(all_dfs)
    results.to_csv(sampled_mols_dir + 'metrics_each' + '.csv', index = False)


    results.fillna(0, inplace=True)
    pd_all_metrics = pd.DataFrame(pocket_metrics)
    avg_scores = results.groupby('pocket_name')[['Ro5','qed', 'sas','sas0-1','logp','tpsa','np','weight']].mean().reset_index()
    pd_all_metrics = pd.merge(pd_all_metrics, avg_scores, on='pocket_name', how='left')
    col_means = pd_all_metrics.iloc[:, 1:].mean().round(5)
    avg_row = pd.DataFrame([['avg'] + col_means.tolist()], columns=pd_all_metrics.columns)
    pd_all_metrics = pd.concat([pd_all_metrics, avg_row], ignore_index=True)
    pd_all_metrics.to_csv(sampled_mols_dir + 'pocket_metrics' + '.csv', index = False)


    print('Valid ratio: ', np.round(len(results)/(molecule_nums*all_lens), 3)) 
    print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))      
    print('Novelty ratio: ', np.round(novel_ratio/100, 3))                   