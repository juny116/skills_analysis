from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('error/', views.test, name='error'),
    path('preprocess/', views.preprocess, name='preprocess'),
    path('weight_update/', views.weight_update, name='weight_update'),
    path('retrieve/', views.retrieve, name='retrieve'),
]