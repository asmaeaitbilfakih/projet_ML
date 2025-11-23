from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('reglog_details/', views.regLog_details, name='reglog_details'),
    path('reglog_atelier/', views.regLog_atelier , name='reglog_atelier'),
    path('reglog_tester/', views.regLog_tester, name='reglog_tester'),
    path('reglog_prediction', views.linear_prediction, name='reglog_prediction'),

    path('linreg_details/', views.linreg_details, name='linreg_details'),
    path('linreg_atelier/', views.linreg_atelier, name='linreg_atelier'),
    path('prediction-or/', views.prediction_or, name='prediction_or'),
    path('linreg_atelier/', views.linreg_atelier, name='linreg_atelier'),
    path('linear_prediction/', views.linear_prediction, name='linear_prediction'),
    path('linear_results/', views.linear_results, name='linear_results'),

    path('decision_tree_prediction/', views.decision_tree_prediction, name='decision_tree_prediction'),
    path('decision_tree_details/', views.decision_tree_details, name='decision_tree_details'),
    path('decision_tree_atelier/', views.decision_tree_atelier, name='decision_tree_atelier'),
    path('prediction_tree_c/', views.prediction_tree_c, name='prediction_tree_c'),
    path('decision_tree_results/', views.decision_tree_results, name='decision_tree_results')


]