import os
from dotenv import load_dotenv
import time
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage


load_dotenv()

## In this program we will see how to invoke a model using the init_chat_model function and also how to stream the response from the model.

def get_weather(city:str) ->str:
    """get weather for a given city"""
    return f"Its always sunny in {city}"

model=init_chat_model(model="llama-3.3-70b-versatile",model_provider="groq",max_tokens=1024,timeout=30,temperature=0.7)

response=model.invoke("what is the weather in delhi?")
print(response.content)

chunks=[]
for chunk in model.stream("what is the weather in delhi?"):
    print(chunk.text,end="",flush=True)
    time.sleep(0.2)
    chunks.append(chunk)

full_message=sum(chunks[1:],chunks[0])
print("\n\nFull message:",full_message.text)

## the Model is bind with the tool and then invoked with the tool. the tool is not called directly.
model_tools=model.bind_tools([get_weather])
response_with_tools=model_tools.invoke("what is the weather in delhi?")
print(response_with_tools.tool_calls)




