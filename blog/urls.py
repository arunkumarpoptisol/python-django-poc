from django.urls import path
from .views import BlogPostView, BlogPostDetailView, CreatePostView, UpdatePostView
from . import views

urlpatterns = [
    path("", BlogPostView.as_view(), name="blog-home"),
    path("<int:pk>/", BlogPostDetailView.as_view(), name="blog-detail"),
    path("add/", CreatePostView.as_view(), name="blog-add"),
    path("<int:pk>/update/", UpdatePostView.as_view(), name="blog-update"),
    path("about/", views.about, name="blog-about"),
]
