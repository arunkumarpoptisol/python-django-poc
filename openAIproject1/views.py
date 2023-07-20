from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
import openai
import os

import textwrap
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
# from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader

from dotenv import load_dotenv


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

# load_dotenv()
# model = OpenAI(model_name="text-davinci-003", temperature=0.0)

openai.api_key = os.getenv('OPENAI_API_KEY')


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


def print_response(response: str):
    print("\n".join(textwrap.wrap(response, width=100)))


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


def langChainFunc(request):

    # prompt-templates
    constructor_prompt = PromptTemplate(
        input_variables=["adjective", "content"],
        template="Tell me a {adjective} joke about {content}.",
    )

    promptTemplates = f"Constructor: \
        {constructor_prompt.format(adjective='funny', content='chickens')}"

    # few shot
    examples = [
        {"word": "happy", "antonym": "sad"},
        {"word": "tall", "antonym": "short"},
    ]

    example_formatter_template = """Word: {word}
    Antonym: {antonym}
    """

    example_prompt = PromptTemplate(
        input_variables=["word", "antonym"],
        template=example_formatter_template,
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input\n",
        suffix="Word: {input}\nAntonym: ",
        input_variables=["input"],
        example_separator="\n",
    )

    # output parser

    parser = PydanticOutputParser(pydantic_object=Joke)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()},
    )

    joke_query = "Tell me a joke."
    formatted_prompt = prompt.format_prompt(query=joke_query)

    # openai
    # output = model(formatted_prompt.to_string())
    # parsed_joke = parser.parse(output)
    # print(parsed_joke)

    # document loader 👌
    url = "C:/Users/arunkumar.p/Desktop/Optisol Files/python/python-django-poc/openAIproject1/doc/layout-parser-paper.pdf"
    loader = PyPDFLoader(url)
    pages = loader.load_and_split()

    # text Splitter
    with open(url,  errors="ignore") as f:
        state_of_the_union = f.read()

    text_splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    texts = text_splitter.create_documents([state_of_the_union])

    # Vector Store
    # embeddings = OpenAIEmbeddings()
    # db = FAISS.from_documents(texts, embeddings)

    # query = "What did the president say about Ketanji Brown Jackson"
    # docs = db.similarity_search(query)
    # print(docs[0].page_content)

    loader = CSVLoader(
        file_path="C:/Users/arunkumar.p/Desktop/Optisol Files/python/python-django-poc/openAIproject1/doc/addresses.csv")

    data = loader.load()

    # loader = UnstructuredCSVLoader(
    #     file_path="C:/Users/arunkumar.p/Desktop/Optisol Files/python/python-django-poc/openAIproject1/doc/addresses.csv", mode="elements"
    # )
    # docs = loader.load()
    # loader = CSVLoader(
    #     file_path="C:/Users/arunkumar.p/Desktop/Optisol Files/python/python-django-poc/openAIproject1/doc/addresses.csv",
    #     # csv_args={
    #     #     "delimiter": ",",
    #     #     "quotechar": '"',
    #     #     "fieldnames": ["Name", "Points"],
    #     # },
    # )

    # data = loader.load()
    return render(request, "output.html",
                  {"promptTemplates": promptTemplates,
                   "FewShot": few_shot_prompt.format(input="big"),
                   "outputparse": formatted_prompt.to_string(),
                   "Numberofpages": len(pages),
                   "firstpage": pages[0],
                   "FirstChunk": texts[0],
                   "SecondChunk": texts[1],
                   #   "VectorStore": docs[0].page_content,
                   #    'CSVLoader': data
                   })
