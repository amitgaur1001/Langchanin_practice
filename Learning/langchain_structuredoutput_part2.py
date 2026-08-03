from typing import Union
from pydantic import BaseModel,Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,AIMessage,ToolMessage

class NewBooking(BaseModel):
    '''A request to book a new movie ticket'''
    customer_Name:str=Field(default="",description="Name of the customer")
    movie:str=Field(default="",description="Name of the Movie")
    ticket_count: int=Field(default=1,ge=1,le=10)

class CancelBooking(BaseModel):
    '''A request to cancel an existing booking'''
    customer_name:str=Field(default="",description="Name of the customer")
    movie_title:str=Field(default="", description="Name of the Movie")

model=init_chat_model(model="llama-3.3-70b-versatile",model_provider="groq",max_tokens=1024,timeout=30,temperature=0.7)
union_agent=create_agent(model=model,
                         tools=[],
                         response_format=ToolStrategy(Union[NewBooking,CancelBooking]),
                         system_prompt="extract the booking details exactly as stated.do not invent information"
                    
                         )



result=union_agent.invoke(HumanMessage(content="Hi I am amit, Strictly book 15 tickets, I want to give a party to my students"))

for m in result["messages"]:
    print(f"-------{m.type}---------")
    print(m.content)
