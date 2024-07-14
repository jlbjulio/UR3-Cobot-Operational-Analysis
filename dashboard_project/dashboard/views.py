from django.shortcuts import render
from .dash_app import dash_app

def dashboard(request):
    return render(request, 'dashboard/dashboard.html')