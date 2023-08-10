from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("chatgpt/", views.CHATGPT, name="chatgpt"),
    path("langchain/", views.langChainFunc, name="langchain"),
    path("PDFChat/", views.PDFChat, name="PDFChat"),
    path("csv/", views.csv, name="csv"),
    path("AskPDFQuestion/", views.AskPDFQuestion, name="AskPDFQuestion"),
    path("AskCSVQuestion/", views.AskCSVQuestion, name="AskCSVQuestion"),

]
