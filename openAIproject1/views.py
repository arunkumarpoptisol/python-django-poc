from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.getenv('OPENAI_API_KEY')

def index(req):
    return HttpResponse('<h2>OPENAI HOME</h2>')


def CHATGPT(req):
    try:
        text = f"""
        You should express what you want a model to do by \ 
        providing instructions that are as clear and \ 
        specific as you can possibly make them. \ 
        This will guide the model towards the desired output, \ 
        and reduce the chances of receiving irrelevant \ 
        or incorrect responses. Don't confuse writing a \ 
        clear prompt with writing a short prompt. \ 
        In many cases, longer prompts provide more clarity \ 
        and context for the model, which can lead to \ 
        more detailed and relevant outputs.
        """
        prompt = f"""
        Summarize the text delimited by triple backticks \ 
        into a single sentence.
        ```{text}```
        """
        response = get_completion(prompt)
        print(response)
        return HttpResponse(response)

    except Exception as e:
        return HttpResponse(e)


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]
