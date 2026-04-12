from django.contrib import admin
from django.urls import path, include
from classifier.views import home

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('classifier.urls')),
    path('', home, name='home'),
]
