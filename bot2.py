import re
import os
import google.generativeai as genai

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

data = CSVLoader(file_path='./diabetes_food.csv').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(data)
embedding_model = HuggingFaceHubEmbeddings(huggingfacehub_api_token=os.getenv("HF_TOKEN"))
db = FAISS.from_documents(documents, embedding_model)  # default_embedding_model = all-mpnet-base-v2 (in sentence transformers repo)
my_retriever = db.as_retriever(search_kwargs={"k": 5})

from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

def get_result(query_typed):

    genai.configure(api_key=os.getenv("GEMINI_API"))
    # Set up the model
    generation_config = {
    "temperature": 0.1,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    ]

    model = genai.GenerativeModel(model_name = "gemini-1.0-pro", #stable version
                                generation_config=generation_config,
                                safety_settings=safety_settings)
    user_typed = query_typed

    
    docs = my_retriever.invoke(user_typed)
    temp_docs = "" #relevant info as combined string
    for each in docs:
        temp_docs += (str(each) + ". ")

    ques = user_typed
    context1 = temp_docs
    prompt1 = f"Read this thoroughly:- {context1}.  Now, you are supposed to give nutritional information (if mentioned in context give those values, else give all information you find) of the food mentioned:- {ques}. When asked to compare food items, search the food items carefully. Assume that this is the only source of info you have. Get me the best possible answer from here.If you are unable to get answer, simply return \"did_not_find_answer\" "


    response = model.generate_content(prompt1)
    #use try, catch before response.text
    op_data = response.text

    p1 = "did_not_find_answer"
    p2 = "does not"
    p3 = "not available"
    p4 = "not provided"
    p5 = "Not mentioned"
    if(bool(re.search("{}|{}|{}|{}|{}".format(p1,p2,p3,p4,p5),op_data))):
        print("just inside if condition")
        ddg_result = search.run(user_typed + " in food and nutrient domain") #result from duckduckgo search
        context2 = ddg_result
        prompt2 = f"Read this thoroughly:- {context2}.  Using this context, you are supposed to give answer for this query:- {ques}. "
        response = model.generate_content(prompt2)
        print("resp of prompt 2")
        try:
            op_data = response.text
        except ValueError:
            op_data = "some error occured"

    return op_data