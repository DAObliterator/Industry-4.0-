from django.urls import path
from . import views

urlpatterns = [
    # Path is empty string '' because it's the home page of this app
    path('', views.dashboard, name='dashboard'),
]
