
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "simple_ui.settings")
django.setup()

from app.models import Job


from datasets import load_dataset
dataset = load_dataset('json', data_files='data/skills.jsonl')['train']


for instance in dataset:
    job = Job.objects.create(id=instance['id'], name=instance['name'], description=instance['description'], class_1=instance['class_1'], 
        class_2=instance['class_2'], objective=instance['objective'], abilities=', '.join(instance['abilities']), 
        q_digit=instance['q_digit'])

