import pandas as pd
import json

last_prop_mean={}


df = pd.read_csv("./data_pdbbind/dock/dock_protein_center/pocket_vina.csv")
column_name = "affinity"
average_value = df[column_name].mean()
print(f"The average value of the '{column_name}' column is: {average_value}")

last_prop_mean['affinity']=average_value


data_json='./data_pdbbind/dock/dock_protein_center/dock_scoring_dict.json'
with open(data_json, 'r') as file:
    data = json.load(file)
values = list(data.values())
average_value_2 = sum(values) / len(values)
print("Average Value:", average_value_2)

last_prop_mean['score']=average_value_2



output_file = "./data_pdbbind/dock/dock_protein_center/last_mean.json"

with open(output_file, "w") as file:
    json.dump(last_prop_mean, file, indent=4)

print(f"Dictionary saved to {output_file}")

