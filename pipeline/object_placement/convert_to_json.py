import pandas as pd
import json
import os

def convert_to_json(csv_file_path, json_file_path):
    with open(csv_file_path, "r") as f:
        lines = f.readlines()
    records = {}
    for line in lines[1:]:
        record = (line.strip('\n').split(";"))
        n = len(record)
        records[record[0]] = {
            'description': ', '.join(record[1:n-1]),
            'objects': [obj.strip() for obj in record[n-1].split(',')]
        }
    return records
# 


if __name__ == "__main__":
    
    sub_dictionaries = {}
    
    for file in os.listdir("taxonomy"):
        if file.endswith(".csv"):
            sub_dictionaries[file.replace('.csv', '')] = convert_to_json(f"taxonomy/{file}", f"taxonomy/{file.replace('.csv', '.json')}")
    
    with open("taxonomy/taxonomy.json", "w") as f:
        json.dump(sub_dictionaries, f, indent=4)
        
    all_objects = []
    for sub_dictionary in sub_dictionaries.values():
        for attributes in sub_dictionary:
            all_objects.extend(sub_dictionary[attributes]['objects'])
    all_objects = list(set(all_objects))
    all_objects.sort()
    with open("taxonomy/all_objects.json", "w") as f:
        json.dump(all_objects, f, indent=4)