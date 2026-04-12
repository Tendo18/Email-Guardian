from django.urls import path
from classifier import views

urlpatterns = [
    path('classify/', views.ClassifyEmailView.as_view(), name='classify'),
    path('feedback/', views.FeedbackView.as_view(), name='feedback'),
    path('classifications/', views.ClassificationListView.as_view(), name='classifications'),
    path('model-status/', views.ModelStatusView.as_view(), name='model-status'),
]