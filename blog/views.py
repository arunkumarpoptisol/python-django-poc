from django.shortcuts import render
from django.views.generic import ListView, DetailView, CreateView,UpdateView
from .models import Post


def home(request):
    context = {
        'posts': Post.objects.order_by('date_posted')
    }
    return render(request, 'blog/home.html', context)


class BlogPostView(ListView):
    model = Post
    template_name = 'blog/home.html'
    context_object_name = 'posts'
    ordering = ['-date_posted']


class BlogPostDetailView(DetailView):
    model = Post
    template_name = 'blog/blog-detail.html'


class CreatePostView(CreateView):
    model = Post
    fields = ['title', 'content']

class UpdatePostView(UpdateView):
    model = Post
    fields = ['title', 'content']

def about(request):
    return render(request, 'blog/about.html', {'title': 'About'})
