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

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from django import forms
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader

from dotenv import load_dotenv


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

# load_dotenv()
# model = OpenAI(model_name="text-davinci-003", temperature=0.0)

openai.api_key = os.getenv('OPENAI_API_KEY')


def index(request):
    context = {}
    context['content'] = 'OPENAI HOME'
    return render(request, "base.html", context)


def CHATGPT(request):
    try:
       # given the sentiment from the lesson on "inferring",
        # and the original customer message, customize the email
        sentiment = "negative"

        # review for a blender
        review = f"""
        So, they still had the 17 piece system on seasonal \
        sale for around $49 in the month of November, about \
        half off, but for some reason (call it price gouging) \
        around the second week of December the prices all went \
        up to about anywhere from between $70-$89 for the same \
        system. And the 11 piece system went up around $10 or \
        so in price also from the earlier sale price of $29. \
        So it looks okay, but if you look at the base, the part \
        where the blade locks into place doesn’t look as good \
        as in previous editions from a few years ago, but I \
        plan to be very gentle with it (example, I crush \
        very hard items like beans, ice, rice, etc. in the \ 
        blender first then pulverize them in the serving size \
        I want in the blender then switch to the whipping \
        blade for a finer flour, and use the cross cutting blade \
        first when making smoothies, then use the flat blade \
        if I need them finer/less pulpy). Special tip when making \
        smoothies, finely cut and freeze the fruits and \
        vegetables (if using spinach-lightly stew soften the \ 
        spinach then freeze until ready for use-and if making \
        sorbet, use a small to medium sized food processor) \ 
        that you plan to use that way you can avoid adding so \
        much ice if at all-when making your smoothie. \
        After about a year, the motor was making a funny noise. \
        I called customer service but the warranty expired \
        already, so I had to buy another one. FYI: The overall \
        quality has gone done in these types of products, so \
        they are kind of counting on brand recognition and \
        consumer loyalty to maintain sales. Got it in about \
        two days.
        """
        prompt = f"""
        You are a customer service AI assistant.
        Your task is to send an email reply to a valued customer.
        Given the customer email delimited by ```, \
        Generate a reply to thank the customer for their review.
        If the sentiment is positive or neutral, thank them for \
        their review.
        If the sentiment is negative, apologize and suggest that \
        they can reach out to customer service. 
        Make sure to use specific details from the review.
        Write in a concise and professional tone.
        Sign the email as `AI customer agent`.
        Customer review: ```{review}```
        Review sentiment: {sentiment}
        """
        response = get_completion(prompt)
        # print(response)
        # return HttpResponse(response)
        context = {}
        context['sentiment'] = sentiment
        context['review'] = review
        context['prompt'] = prompt
        context['content'] = response
        # return HttpResponse('<h2>OPENAI HOME</h2>')
        return render(request, "base.html", context)

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
    try:
        # prompt-templates  👌
        constructor_prompt = PromptTemplate(
            input_variables=["adjective", "content"],
            template="Tell me a {adjective} joke about {content}.",
        )

        promptTemplates = f"Constructor: \
            {constructor_prompt.format(adjective='funny', content='chickens')}"

        # few shot 👌
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

        # output parser 👌

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
        url = "C:/Users/arunkumar.p/Desktop/Optisol Files/python/python-django-poc/openAIproject1/doc/history of computers.pdf"
        # url = "C:/Users/arunkumar.p/Desktop/Optisol Files/python/python-django-poc/openAIproject1/doc/layout-parser-paper.pdf"
        loader = PyPDFLoader(url)
        pages = loader.load_and_split()

        # text Splitter
        with open(url,  errors="ignore") as f:
            state_of_the_union = f.read()

        text_splitter = CharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=20,
        )
        texts = text_splitter.create_documents([state_of_the_union])

        # Vector Store
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(texts, embeddings)

        query = "Computer"
        # 👎
        docs = db.similarity_search(query)
        # print(docs[0].page_content)

        # loader = UnstructuredCSVLoader(
        #     file_path="C:/Users/arunkumar.p/Desktop/Optisol Files/python/python-django-poc/openAIproject1/doc/addresses.csv", mode="elements"
        # )
        # docs = loader.load()
        loader = CSVLoader(
            file_path="C:/Users/arunkumar.p/Desktop/Optisol Files/python/python-django-poc/openAIproject1/doc/addresses.csv",
            # csv_args={
            #     "delimiter": ",",
            #     "quotechar": '"',
            #     "fieldnames": ["Name", "Points"],
            # },
        )

        data = loader.load()
        return render(request, "output.html",
                      {"promptTemplates": promptTemplates,
                       "FewShot": few_shot_prompt.format(input="big"),
                       "outputparse": formatted_prompt.to_string(),
                       "Numberofpages": len(pages),
                       "firstpage": pages[0],
                       "FirstChunk": texts[0],
                       "SecondChunk": texts[1],
                       "VectorStore": docs[0].page_content,
                       'CSVLoader': data
                       })
    except Exception as e:
        return HttpResponse(e)


# creating a form
class InputForm(forms.Form):
    question = forms.CharField(max_length=200)


def AskPDFQuestion(request):
    print(request)
    context = {}
    context['form'] = InputForm()
    return render(request, "pdfInput.html", context)


def PDFChat(request):
    if request.method == "POST":
        data = request.POST
        action = data.get("question")

        try:
            url = "C:/Users/arunkumar.p/Desktop/Optisol Files/python/python-django-poc/openAIproject1/doc/history of computers.pdf"
            # load the pdf with PyPDF2 package
            pdf_reader = PdfReader(url)

            text = ""
            # concat all the pages as text
            for page in pdf_reader.pages:
                text += page.extract_text()

            # text Splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Vector Store
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            query = action

            docs = VectorStore.similarity_search(query)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
                print(response)
                # st.write(response)

            context = {}
            context['input'] = query
            context['PDF'] = response
            # context['form'] = InputForm()
            return render(request, "pdfOutput.html", context)
        except Exception as e:
            return HttpResponse(e)


def AskCSVQuestion(request):
    print(request)
    context = {}
    context['form'] = InputForm()
    context['action'] = InputForm()
    return render(request, "csvInput.html", context)


def csv(request):
    if request.method == "POST":
        data = request.POST
        action = data.get("question")
        loader = CSVLoader(
            file_path="C:/Users/arunkumar.p/Desktop/Optisol Files/python/python-django-poc/openAIproject1/doc/addresses.csv",
            encoding="utf-8", csv_args={
                'delimiter': ','}
        )
        data = loader.load()
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_documents(data, embedding=embeddings)

        query = action

        docs = VectorStore.similarity_search(query)
        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
            print(response)
            # st.write(response)

        context = {}
        context['input'] = query
        context['PDF'] = response
        # context['form'] = InputForm()
        return render(request, "pdfOutput.html", context)
