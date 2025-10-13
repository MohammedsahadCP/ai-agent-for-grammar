import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
from langchain.schema.runnable import RunnableLambda,RunnableSequence

load_dotenv()

groq_api_key=os.getenv('GROQ_API_KEY')

def webscrap(url:str):
    data=WebBaseLoader(web_paths=("https://www.luminartechnolab.com/course-detail/python-training-kochi",),
                   bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                       class_=("course-details-content")
                   )
                                            ))
    data=data.load()        
    return data[0].page_content
# print(data[0].page_content)

prompt=ChatPromptTemplate(
    [
        ("system","for any content convert it into concise passage containing important points"),
        ("user","{data}")
    ]
)

model=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

parser=StrOutputParser()

chain = RunnableSequence([
    RunnableLambda(lambda x: {"data": webscrap(x["url"])}),
    prompt,
    model,
    parser
])

app=FastAPI(title='Langchain Server',
            version="1.0",
            description="A simple API server using Langchain runnable interfaces"
            )


add_routes(
    app,
    chain,
    path="/chain"
)

@app.get("/")
def home():
    return {"message": "Langchain Server is running. Visit /docs for Swagger UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)