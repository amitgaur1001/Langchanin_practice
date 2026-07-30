import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core import tools

load_dotenv()

print(os.getenv("GROQ_API_KEY")[:])



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

print(response)