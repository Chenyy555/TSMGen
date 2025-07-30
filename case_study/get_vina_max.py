import json
import yaml
import os
import csv
import pandas as pd
import shutil

def get_top_10(dock_data):
    affinity_values = {}
    for key, values in dock_data.items():
        for record in values:
            if record.get('mode_id') == 0:
                affinity_values[key] = record.get('affinity', None)
                break

    top_10_values = sorted(affinity_values.items(), key=lambda x: x[1])[:10]
  
    return top_10_values

def get_pdbqt_smiles(pdbqt_path, save_path, smiles_all,top_10_values):
    save_path = os.path.join(save_path,'smiles_pdbqt')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gen_smiles_top={}
    for key,value in top_10_values:
        key = key.rstrip('.pdbqt')
        one_pdbqt_path = os.path.join(pdbqt_path,key) + '_out.pdbqt'
        destination_file=os.path.join(save_path,key) + '_out.pdbqt'
        shutil.copy(one_pdbqt_path, destination_file)
        
        parts = key.rsplit('_', 1)
        smiles_index = int(parts[1])
        one_smiles=smiles_all[smiles_index]

        gen_smiles_top[key] = {
            'smiles': one_smiles,
            'affinity_value': value
        }
    
    return gen_smiles_top


def get_smi(smiles_yaml):
    with open(smiles_yaml, 'r') as f:
        config = yaml.full_load(f)
    return list(config.keys())



json_file_7D5U = './case_study/7D5U/dock_file_save/158585049/dock_dict.json'
smiles_7D5U='./case_study/7D5U/sample_300_30_True_1_1_158585049/7d5u_sampled_temp1.yaml'
save_path_7D5U = './case_study/7D5U/best_10/'
pdbqt_path_7D5U = './case_study/7D5U/dock_file_save/158585049/out_pdbqt/'
with open(json_file_7D5U, 'r') as f:
    data_7D5U = json.load(f)

top_10_7D5U = get_top_10(data_7D5U)
smiles_all_7D5U = get_smi(smiles_7D5U)
gen_smiles_7D5U = get_pdbqt_smiles(pdbqt_path_7D5U,save_path_7D5U,smiles_all_7D5U,top_10_7D5U)

df_results_7D5U = pd.DataFrame.from_dict(gen_smiles_7D5U, orient='index')
csv_file_path = os.path.join(save_path_7D5U,'gen_smiles.csv')
df_results_7D5U.to_csv(csv_file_path)




json_file_7DPU = './case_study/7DPU/dock_file_save/2258692459/dock_dict.json'
smiles_7DPU='./case_study/7DPU/sample_300_30_True_1_1_2258692459/7dpu_sampled_temp1.yaml'
save_path_7DPU = './case_study/7DPU/best_10/'
pdbqt_path_7DPU='./case_study/7DPU/dock_file_save/2258692459/out_pdbqt/'
with open(json_file_7DPU, 'r') as f:
    data_7DPU = json.load(f)

top_10_7DPU = get_top_10(data_7DPU)
smiles_all_7DPU = get_smi(smiles_7DPU)
gen_smiles_7DPU = get_pdbqt_smiles(pdbqt_path_7DPU,save_path_7DPU,smiles_all_7DPU,top_10_7DPU)

df_results_7DPU = pd.DataFrame.from_dict(gen_smiles_7DPU, orient='index')
csv_file_path = os.path.join(save_path_7DPU,'gen_smiles.csv')
df_results_7DPU.to_csv(csv_file_path)



# lowest_affinity = float('inf')
# lowest_key = None
# lowest_data = None

# for key, modes in data.items():
#     for mode in modes:
#         if mode['mode_id'] == 0:
#             if mode['affinity'] < lowest_affinity:
#                 lowest_affinity = mode['affinity']
#                 lowest_key = key
#                 lowest_data = mode

# print("Key:", lowest_key)
# print("Affinity:", lowest_data['affinity'])


