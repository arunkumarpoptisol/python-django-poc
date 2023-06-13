from django.shortcuts import render,get_object_or_404

from django.template import loader
# Create your views here.
from django.http import HttpResponse,Http404
from .models import Question

def index(request):
    Question_Result=Question.objects.order_by('date')
    context = {"latest_question_list": Question_Result}
    return render(request, "index.html", context)

#Show details page
def detail(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    return render(request, "detail.html", {"question": question})