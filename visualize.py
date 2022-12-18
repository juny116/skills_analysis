import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "simple_ui.settings")
django.setup()
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import torch

from app.models import Job, WeightSetup

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

tab20 = plt.cm.get_cmap('tab20').colors
tab20b = plt.cm.get_cmap('tab20b').colors
colors = (tab20 + tab20b)[:22]

attr_list = ['name', 'description', 'class_1', 'class_2', 'objective', 'abilities_concat', 'abilities_avg']

iris = load_iris()
df = pd.DataFrame(data= np.c_[iris.data, iris.target] , 
                  columns= ['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

with open('data/data.pkl', 'rb') as f:
    embeded_data = pickle.load(f)

jobs = Job.objects.all()
weight, created = WeightSetup.objects.get_or_create(id=1, defaults={'name': 0.2, 'description': 0.2, 
                                                        'class_1': 0.2, 'class_2': 0, 'objective': 0.2, 'abilities_avg': 0.2, 'abilities_concat': 0})

reshpaed_data = embeded_data.reshape(len(jobs), len(attr_list), -1)
weights = [[float(getattr(weight, attr))] for attr in attr_list]
weights = torch.tensor(weights).view(1, len(attr_list), 1)
weighted_sum = weights * reshpaed_data
weighted_sum = torch.sum(weighted_sum, dim=1)

# class target 정보 제외
train_df = df[['sepal length', 'sepal width', 'petal length', 'petal width']]

# 2차원 t-SNE 임베딩
tsne_np = TSNE(n_components = 2).fit_transform(weighted_sum)

df = pd.DataFrame(data= [q[0] for q in Job.objects.values_list('class_1')], columns= ['class'])

# numpy array -> DataFrame 변환
tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])
# print(tsne_df)
classes = ([q[0] for q in Job.objects.values_list('class_1').distinct()])
print(len(classes))

# class target 정보 불러오기 
tsne_df['class'] = df['class']

temp = []
for i in classes:
    temp.append(tsne_df[tsne_df['class'] == i])
cnt = 0
# target 별 시각화
plt.rcParams["font.family"] = "nanum"
plt.figure(figsize=(10, 7))
for sub_df, color in zip(temp, colors):
    plt.scatter(sub_df['component 0'], sub_df['component 1'], color = color, label = sub_df['class'])
    cnt += 1
# plt.scatter(tsne_df_0['component 0'], tsne_df_0['component 1'], color = 'pink', label = 'setosa')
# plt.scatter(tsne_df_1['component 0'], tsne_df_1['component 1'], color = 'purple', label = 'versicolor')
# plt.scatter(tsne_df_2['component 0'], tsne_df_2['component 1'], color = 'yellow', label = 'virginica')

# plt.xlabel('component 0')
# plt.ylabel('component 1')
plt.xticks([]),plt.yticks([])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=11)
# plt.show()
plt.savefig('savefig_default.png')