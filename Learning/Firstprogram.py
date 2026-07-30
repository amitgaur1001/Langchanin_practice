import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core import tools
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage


load_dotenv()

# print(os.getenv("GROQ_API_KEY")[:])



def get_weather(city:str) ->str:
    """get weather for a given city"""
    return f"Its always sunny in {city}"


llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0  # Best practice for precise tool calling
)

agent= create_agent(model=llm,
                    tools=[get_weather],
                    system_prompt="you are a helpful assistant"
                    )


response =agent.invoke(
    {
        "messages":[{"role": "user", "content": "what is the weather in delhi?"}]
    }
)

print(response['messages'][-1])

# groq_model=ChatGroq(model="llama-3.3-70b-versatile")

# result=groq_model.invoke("tell me about langchain")
# print(result.content)

# init_groq_model=init_chat_model("groq:llama-3.3-70b-versatile")
init_groq_model=init_chat_model(model="llama-3.3-70b-versatile",model_provider="groq",max_tokens=1024,timeout=30)
result1=init_groq_model.invoke("tell me about langchain")
print(result1.tool_calls)

messages=[SystemMessage(content="you are a pirate, answer everything in pirate Language"),
          HumanMessage(content="what is the capital of india?")
          ]

result2=init_groq_model.invoke(messages)
print(result2.content)

import pprint
pprint.pprint(result2)