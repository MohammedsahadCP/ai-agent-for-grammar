from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langserve import add_routes
load_dotenv()

groq_api_key=os.getenv("groq_api_key")

prompt=ChatPromptTemplate(
    [
        ("system","for any message convert the message to minimal sentence possible"),
        ("user","{message}")
    ]
)

model=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

parser=StrOutputParser()

chain=prompt|model|parser

## app definition


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





if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8001)

    
# from fastapi import FastAPI

# app=FastAPI(
#     title='Langchain Server',
#     version="1.0",
#     description="A simple API server using Langchain runnable interfaces"   
# )

# @app.get("/hello")
# def home():
#     return {"message": "Hello, World!"}     

# from fastapi import FastAPI
# from pydantic import BaseModel
# from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatGroq
# from langchain.output_parsers import StrOutputParser
# from langserve import add_routes
# from dotenv import load_dotenv
# import os

# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # FastAPI app with custom Swagger path
# app = FastAPI(
#     title="LangChain + Groq Server",
#     version="1.0",
#     description="FastAPI + LangChain + Groq API server with Swagger UI",
#     docs_url="/swagger",       # custom Swagger path
#     redoc_url="/redoc",
#     openapi_url="/openapi.json"
# )

# # Define input model for chain
# class ChainInput(BaseModel):
#     message: str

# # Define chain components
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "For any query, just say the number of alphabets in the message."),
#     ("user", "{message}")
# ])

# model = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
# parser = StrOutputParser()

# # Wrap chain in a callable function
# def chain_callable(inputs: dict):
#     message = inputs.get("message")
#     text = prompt.format({"message": message})
#     output = model.predict(text)
#     parsed = parser.parse(output)
#     return {"result": parsed}

# # Add LangServe route
# add_routes(app, chain_callable, path="/chain")

# # Home route
# @app.get("/")
# def home():
#     return {"message": "LangChain + Groq server running. Visit /swagger for Swagger UI."}

# # Run server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
