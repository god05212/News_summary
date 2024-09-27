from django.urls import path
from . import views

app_name = 'news_webapp'

urlpatterns = [
    path('', views.first_view, name='first_view'),
]
