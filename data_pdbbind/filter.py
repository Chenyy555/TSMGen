import yaml
import json


file1_path = "./data_split/test_set.yaml" 
file2_path = "./data_pdbbind/dock/error_pocket2pdbqt.yaml"
file3_path ='./data_pdbbind/dock/dock_protein_center/dock_error_list.json'
output_path = "./data_pdbbind/test_87.yaml"
output_path2 = "./data_pdbbind/test_64.yaml" 


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def write_yaml(data, file_path):
    with open(file_path, 'w') as f:
        yaml.safe_dump(data, f)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

file1_data = load_yaml(file1_path)
file2_data = load_yaml(file2_path)
file3_data = load_json(file3_path)

filtered_data = {k: v for k, v in file1_data.items() if k not in file2_data}
filtered_data2 = {k: v for k, v in file1_data.items() if k not in file3_data}

write_yaml(filtered_data, output_path)
write_yaml(filtered_data2, output_path2)
