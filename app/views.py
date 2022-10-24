from django.shortcuts import render
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.template import loader

from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoTokenizer
from kobert_tokenizer import KoBERTTokenizer
from datasets import load_dataset, DatasetDict
import pickle
from .forms import SearchForm, WeightForm, MaskStringForm
from .models import Job, WeightSetup, MaskString
from tqdm import tqdm

import torch
from torch import nn


with open('data/data.pkl', 'rb') as f:
    data = pickle.load(f)

weighted_embeddings = []
idx_to_id = []
id_to_idx = {}

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

dataset = load_dataset('json', data_files='data/skills.jsonl')['train']
model = AutoModel.from_pretrained('skt/kobert-base-v1').to('cuda:0')
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

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
            build()
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
                                                                'class_1': 0.2, 'class_2': 0, 'abilities': 0.2, 'objective': 0.2, 'type':'CONCAT'})

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
                weighted_embedding = float(weight['name']) * instance['name'] + float(weight['description']) * instance['description'] + \
                    float(weight['class_1']) * instance['class_1'] + float(weight['class_2']) * instance['class_2'] + \
                    float(weight['abilities']) * instance['abilities'] + float(weight['objective']) * instance['objective']
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
        
            target_embeddings = weighted_embeddings[id_to_idx[job_id]]
            similarity = cos(target_embeddings, weighted_embeddings)

            topk = torch.topk(similarity, 5)

            topk = [idx_to_id[idx] for idx in topk.indices]

            target = Job.objects.get(id=job_id)
            results = [Job.objects.get(id=k) for k in topk]
            # results = Job.objects.filter(id__in=topk)
    else:
        form = SearchForm()

    return render(request, 'web/retrieve.html', {'form': form, 'target': target, 'results': results})


def build():
    embeded_data = {}
    for instance in tqdm(dataset):
        description = tokenizer(instance['description'], return_tensors='pt')
        description = {k: v.to('cuda:0') for k, v in description.items()}
        description = model(**description).pooler_output[0].detach().cpu()

        name = tokenizer(instance['name'], return_tensors='pt')
        name = {k: v.to('cuda:0') for k, v in name.items()}
        name = model(**name).pooler_output[0].detach().cpu()

        class_1 = tokenizer(instance['class_1'], return_tensors='pt')
        class_1 = {k: v.to('cuda:0') for k, v in class_1.items()}
        class_1 = model(**class_1).pooler_output[0].detach().cpu()

        class_2 = tokenizer(instance['class_2'], return_tensors='pt')
        class_2 = {k: v.to('cuda:0') for k, v in class_2.items()}
        class_2 = model(**class_2).pooler_output[0].detach().cpu()

        objective = tokenizer(instance['objective'], return_tensors='pt')
        objective = {k: v.to('cuda:0') for k, v in objective.items()}
        objective = model(**objective).pooler_output[0].detach().cpu()

        abilities = tokenizer(' '.join(instance['abilities']), return_tensors='pt')
        abilities = {k: v.to('cuda:0') for k, v in abilities.items()}
        abilities = model(**abilities).pooler_output[0].detach().cpu()

        embeded_data[instance['id']] = {'description': description, 'name': name, 
            'class_1': class_1, 'class_2': class_2, 'objective':objective, 'abilities': abilities}
    

    with open('data/data.pkl', 'wb') as f:
        pickle.dump(embeded_data, f, pickle.HIGHEST_PROTOCOL)

    data = embeded_data
