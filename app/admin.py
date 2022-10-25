from django.contrib import admin

# Register your models here.
from .models import Job, WeightSetup, MaskString


class JobAdmin(admin.ModelAdmin):
    fields = ['id', 'name', 'description', 'class_1', 'class_2', 'abilities', 'objective', 'q_digit']

admin.site.register(Job, JobAdmin)


class WeightSetupAdmin(admin.ModelAdmin):
    fields = ['name', 'description', 'class_1', 'class_2', 'abilities_concat', 'abilities_avg', 'objective']
    
admin.site.register(WeightSetup, WeightSetupAdmin)


class MaskStringAdmin(admin.ModelAdmin):
    fields = ['mask',]
    
admin.site.register(MaskString, MaskStringAdmin)

