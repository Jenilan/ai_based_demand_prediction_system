
from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('process/', views.process, name='process'),
    path('results/', views.results, name='results'),
    path('api/chart-data/', views.api_chart_data, name='api_chart_data'),
    path('results/download/', views.download_results, name='download_results'),
    path('outputs/download/', views.download_output, name='download_output'),
]
