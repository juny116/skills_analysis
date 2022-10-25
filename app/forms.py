from email.policy import default
from django import forms
from .models import CHOICES, WeightSetup, MaskString


class WeightForm(forms.ModelForm):
    class Meta:
        model = WeightSetup
        fields = ("name", "description", "class_1", "class_2", "abilities_concat", "objective", "abilities_avg")


class SearchForm(forms.Form):
    job_id = forms.CharField(max_length=20)


class MaskStringForm(forms.ModelForm):
    
    class Meta:
        model = MaskString
        fields = ("mask", )