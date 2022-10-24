from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoTokenizer
from kobert_tokenizer import KoBERTTokenizer
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import pickle

# dataset = DatasetDict()
dataset = load_dataset('json', data_files='data/skills.jsonl')['train']


model = AutoModel.from_pretrained('skt/kobert-base-v1')
# tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

inputs = tokenizer(dataset[0]['description'], return_tensors='pt')

embeded_data = {}

for instance in tqdm(dataset):
    description = tokenizer(instance['description'], return_tensors='pt')
    description = model(**description).pooler_output[0]

    name = tokenizer(instance['name'], return_tensors='pt')
    name = model(**name).pooler_output[0]

    class_1 = tokenizer(instance['class_1'], return_tensors='pt')
    class_1 = model(**class_1).pooler_output[0]

    class_2 = tokenizer(instance['class_2'], return_tensors='pt')
    class_2 = model(**class_2).pooler_output[0]

    objective = tokenizer(instance['objective'], return_tensors='pt')
    objective = model(**objective).pooler_output[0]

    abilities = tokenizer(' '.join(instance['abilities']), return_tensors='pt')
    abilities = model(**abilities).pooler_output[0]

    embeded_data[instance['id']] = {'description': description, 'name': name, 
        'class_1': class_1, 'class_2': class_2, 'objective':objective, 'abilities': abilities}
    

with open('data/data.pkl', 'wb') as f:
    pickle.dump(embeded_data, f, pickle.HIGHEST_PROTOCOL)


