import os
from dotenv import load_dotenv
import time
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage,ToolMessage
from pydantic import BaseModel,Field
import json

load_dotenv()

model=init_chat_model(model="llama-3.3-70b-versatile",model_provider="groq",max_tokens=1024,timeout=30,temperature=0.7)

print(json.dumps(model.profile,indent=2))
print(model.profile.get("structured_output"))

def get_weather(location: str)-> str:
    '''Get weather at a location'''
    return f"sunny,35° C in {location}"

model_with_tool=model.bind_tools([get_weather])

messages=[

    SystemMessage(content="you are a concise weather assistant. "),
    HumanMessage(content="what is the weather in Delhi?"),
]

ai_msg=model_with_tool.invoke(messages)
messages.append(ai_msg)

print(messages)

if ai_msg.tool_calls:
    for call in ai_msg.tool_calls:
        print(call["name"])
        result=get_weather(**call["args"])

        messages.append(ToolMessage(content=result,tool_call_id=call["id"]))

    final=model_with_tool.invoke(messages)
    print(final.content)