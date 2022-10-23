import pickle
import torch
from torch import nn


target_id = '4'

weight = {'name': 0, 'description': 1, 'class_1': 0, 'class_2': 0, 'abilities': 0.5, 'objective': 0.5}


with open('data/data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('data/skills_dict.pkl', 'rb') as f:
    dict_data = pickle.load(f)


weighted_embeddings = []
id_list = []

for idx, instance in data.items():
    weighted_embedding = weight['name'] * instance['name'] + weight['description'] * instance['description'] + \
        weight['class_1'] * instance['class_1'] + weight['class_2'] * instance['class_2'] + \
        weight['abilities'] * instance['abilities'] + weight['objective'] * instance['objective']

    weighted_embeddings.append(weighted_embedding)
    id_list.append(idx)

    if idx == target_id:
        target_embeddings = weighted_embedding

weighted_embeddings = torch.stack(weighted_embeddings)

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
similarity = cos(target_embeddings, weighted_embeddings)

topk = torch.topk(similarity, 5)

topk = [id_list[idx] for idx in topk.indices]

print('-------------------------')
print(dict_data[target_id]['name'], dict_data[target_id]['q_digit'])
print(dict_data[target_id]['description'])

print('-------------------------')
for k in topk:
    print(dict_data[k]['name'], dict_data[k]['q_digit'])
    print(dict_data[k]['description'])
    print()
