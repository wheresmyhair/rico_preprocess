import json

path_file = './data/json/0.json'

with open(path_file, 'r') as f:
    data = json.load(f)

data['activity']['root'].keys()