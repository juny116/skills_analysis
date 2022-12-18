from django.shortcuts import render
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.template import loader

from sentence_transformers import SentenceTransformer, util
from transformers import AutoConfig, AutoModel, AutoTokenizer
from kobert_tokenizer import KoBERTTokenizer
from datasets import load_dataset, DatasetDict
import pickle
from .forms import SearchForm, WeightForm, MaskStringForm
from .models import Job, WeightSetup, MaskString
from tqdm import tqdm
import numpy as np

import torch
from torch import nn


# weighted_embeddings = []
# idx_to_id = []
# id_to_idx = {}

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

attr_list = ['name', 'description', 'class_1', 'class_2', 'objective', 'abilities_concat', 'abilities_avg']
index_to_attr = {i: attr_list[i] for i in range(len(attr_list))}
sub_attr_list = ['name', 'description', 'class_1', 'class_2', 'objective']

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS').to('cuda:0')

dataset = load_dataset('json', data_files='data/skills.jsonl')['train']
# model = AutoModel.from_pretrained('skt/kobert-base-v1').to('cuda:0')
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

def build(masks):
    global attr_list, sub_attr_list, data
    print('Run Build')
    embeded_data = {}
    for instance in tqdm(dataset):
        batch_inputs = [instance[attr] for attr in sub_attr_list]
        batch_inputs.append(' '.join(instance['abilities']))
        batch_inputs = tokenizer(batch_inputs, return_tensors='pt', padding=True)
        batch_inputs = {k: v.to('cuda:0') for k, v in batch_inputs.items()}
        outputs = model(**batch_inputs).pooler_output.detach().cpu()

        batch_inputs = tokenizer(instance['abilities'], return_tensors='pt', padding=True)
        batch_inputs = {k: v.to('cuda:0') for k, v in batch_inputs.items()}
        ablilites_outputs = model(**batch_inputs).pooler_output.detach().cpu()
        ablilites_outputs = ablilites_outputs.mean(dim=0).unsqueeze(0)
        outputs = torch.cat([outputs, ablilites_outputs], dim=0)
        embeded_data[instance['id']] = outputs.detach().numpy()

    with open('data/data.pkl', 'wb') as f:
        pickle.dump(embeded_data, f, pickle.HIGHEST_PROTOCOL)

    data = embeded_data


def build2(masks):
    global attr_list, sub_attr_list, idx_to_id, id_to_idx, data
    id_to_idx = {}
    idx_to_id = []
    print('Run Build2')
    embeded_data = []
    for idx, instance in tqdm(enumerate(dataset)):

        batch_inputs = [instance[attr] for attr in sub_attr_list]
        batch_inputs.append(' '.join(instance['abilities']))

        outputs = model.encode(batch_inputs)
        ablilites_outputs = model.encode(instance['abilities'])
        ablilites_outputs = np.mean(ablilites_outputs, axis=0, keepdims=True)
        outputs = np.concatenate([outputs, ablilites_outputs])
        embeded_data.append(outputs)
        idx_to_id.append(instance['id'])
        id_to_idx[instance['id']] = idx * len(attr_list)

    embeded_data = np.concatenate(embeded_data)
    with open('data/data.pkl', 'wb') as f:
        pickle.dump(embeded_data, f, pickle.HIGHEST_PROTOCOL)

    data = embeded_data
    print('Done Build2')
    print(id_to_idx)


try:
    with open('data/data.pkl', 'rb') as f:
        data = pickle.load(f)
except:
    build2(None)


def index(request):
    return render(request, 'web/index.html')


def preprocess(request):
    masks = MaskString.objects.all()
    build_done = False
    if request.method == "POST":
        form = MaskStringForm(request.POST)
        item_to_delete = request.POST.get('delete_items')
        build_embeddings = request.POST.get('build_embeddings')

        if build_embeddings:
            build2(masks)
            build_done = True

        elif item_to_delete:
            MaskString.objects.get(pk=item_to_delete).delete()

        elif form.is_valid():
            form.save()

    else:
        form = MaskStringForm()

    return render(request, 'web/preprocess.html', {'form': form, 'masks': masks, 'build_done': build_done})

def weight_update(request):
    global weighted_embeddings
    global idx_to_id
    global id_to_idx

    # try:
    item, created = WeightSetup.objects.get_or_create(id=1, defaults={'name': 0.2, 'description': 0.2, 
                                                                'class_1': 0.2, 'class_2': 0, 'objective': 0.2, 'abilities_avg': 0.2, 'abilities_concat': 0})

    weighted_embeddings = []
    idx_to_id = []
    id_to_idx = {}
    success = False
    if request.method == 'POST':
        form = WeightForm(request.POST, instance=item)
        if form.is_valid():
            weight = form.cleaned_data
            cnt = 0
            for id, instance in data.items():
                weights = [[float(weight[attr])] for attr in attr_list]
                weights = torch.Tensor(weights)
                weighted_embedding = torch.mul(weights, instance)
                weighted_embedding = torch.sum(weighted_embedding, 0)
                weighted_embeddings.append(weighted_embedding)
                idx_to_id.append(id)
                id_to_idx[id] = cnt
                cnt += 1

            weighted_embeddings = torch.stack(weighted_embeddings)
            success = True
            form.save()
    else:
        form = WeightForm(instance=item)

    return render(request, 'web/weight_update.html', {'form': form, 'success': success})


def test(request, test_id):
    # question = get_object_or_404(Question, pk=question_id)
    test = test_id
    return render(request, 'web/test.html', {'test': test})


def retrieve(request):
    # question = get_object_or_404(Question, pk=question_id)
    target = None
    results = None

    if request.method == 'POST':
        form = SearchForm(request.POST)
        if not torch.is_tensor(weighted_embeddings):
            error=True
            return render(request, 'web/retrieve.html', {'form': form, 'target': target, 'results': results, 'error': error})

        if form.is_valid():
            job_id = form.cleaned_data['job_id']
            num = int(form.cleaned_data['topk'])

            target_embeddings = weighted_embeddings[id_to_idx[job_id]]
            similarity = cos(target_embeddings, weighted_embeddings)

            topk = torch.topk(similarity, num)
            scores = topk.values.tolist()
            topk = [idx_to_id[idx] for idx in topk.indices]

            target = Job.objects.get(id=job_id)
            jobs = [Job.objects.get(id=k) for k in topk]
            results = [result for result in zip(jobs, scores)]
            # results = Job.objects.filter(id__in=topk)
    else:
        form = SearchForm()

    return render(request, 'web/retrieve.html', {'form': form, 'target': target, 'results': results})


def weight_update2(request):
    global weighted_embeddings
    global idx_to_id
    global id_to_idx
    weighted_embeddings = []
    # try:
    item, created = WeightSetup.objects.get_or_create(id=1, defaults={'name': 0.2, 'description': 0.2, 
                                                                'class_1': 0.2, 'class_2': 0, 'objective': 0.2, 'abilities_avg': 0.2, 'abilities_concat': 0})

    success = False
    if request.method == 'POST':
        form = WeightForm(request.POST, instance=item)
        if form.is_valid():
            weight = form.cleaned_data
            form.save()
    else:
        form = WeightForm(instance=item)

    return render(request, 'web/weight_update.html', {'form': form, 'success': success})


def retrieve2(request):
    target = None
    results = None
    global attr_list
    global idx_to_id
    global id_to_idx
    # print(id_to_idx)
    if request.method == 'POST':
        form = SearchForm(request.POST)

        weight, created = WeightSetup.objects.get_or_create(id=1, defaults={'name': 0.2, 'description': 0.2, 
                                                                    'class_1': 0.2, 'class_2': 0, 'objective': 0.2, 'abilities_avg': 0.2, 'abilities_concat': 0})

        if form.is_valid():
            job_id = form.cleaned_data['job_id']
            num = int(form.cleaned_data['topk'])

            jobs = Job.objects.all()

            # calc sim for all field (include similarity between different fields)
            similarity = util.cos_sim(data[id_to_idx[job_id]:id_to_idx[job_id]+7], data)
            torch.set_printoptions(profile="full")

            # construct indices for similarity between same field
            indices = [[i * len(attr_list) + j for i in range(len(jobs))] for j in range(len(attr_list))]
            temp = []
            for i in range(len(attr_list)):
                temp.append(similarity[i][indices[i]])
            similarity = torch.stack(temp)

            weights = [[float(getattr(weight,attr))] for attr in attr_list]
            weights = torch.tensor(weights).view(len(attr_list), 1)
            similarity = weights * similarity
            similarity = torch.sum(similarity, dim=0)

            topk = torch.topk(similarity, num)
            scores = topk.values.tolist()
            topk = [idx_to_id[idx] for idx in topk.indices]

            target = jobs.get(id=job_id)
            retreived = [jobs.get(id=k) for k in topk]
            results = [result for result in zip(retreived, scores)]

    else:
        form = SearchForm()

    return render(request, 'web/retrieve.html', {'form': form, 'target': target, 'results': results})