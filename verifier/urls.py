from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("api/health", views.health, name="health"),
    path("api/chatgpt", views.chatgpt, name="chatgpt"),
    path("api/verify", views.verify, name="verify"),
]
