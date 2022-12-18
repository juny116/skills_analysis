from email.policy import default
from django import forms
from .models import CHOICES, WeightSetup, MaskString, Job

Q_DIGIT_CHOICES = sorted([(q[0],q[0]) for q in Job.objects.values_list('q_digit').distinct()])
SEARCH_METHOD = [('emb','임베딩 가중합'),('sim', '유사도 가중합')]

class WeightForm(forms.ModelForm):
    class Meta:
        model = WeightSetup
        fields = ("name", "description", "class_1", "class_2", "abilities_concat", "objective", "abilities_avg")


class SearchForm(forms.Form):
    job_name = forms.CharField(max_length=20, widget=forms.TextInput(attrs={'style': 'width: 200px', 'id':'tags'}))
    q_digit = forms.ChoiceField(widget=forms.Select, choices=Q_DIGIT_CHOICES)
    topk = forms.DecimalField(max_digits=5, widget=forms.NumberInput(attrs={'style': 'width: 50px'}))
    method = forms.ChoiceField(widget=forms.Select, choices=SEARCH_METHOD)

class MaskStringForm(forms.ModelForm):
    
    class Meta:
        model = MaskString
        fields = ("mask", )