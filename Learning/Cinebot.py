import os
from dotenv import load_dotenv
import time
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from pydantic import BaseModel,Field
from typing import Literal,Union
from langchain.agents.structured_output import ProviderStrategy,ToolStrategy
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

model=init_chat_model(model="llama-3.3-70b-versatile",
                      model_provider="groq",
                      max_tokens=1024,
                      timeout=30,
                      temperature=0.7
                      )


class booking_request(BaseModel):
    customer_name:str=Field(description="name of the customer")
    movie:str=Field(description="name of the movie")
    action:Literal["book", "cancel"]=Field(description="action to be performed, either book or cancel")
    ticket_count :int=Field(description="Number of tickets",gt=1,lt=10)

class NewBooking(BaseModel):
    '''A request to BOOK new tickets'''
    customer_name: str
    Movie_title: str
    ticket_count: int

class CancelBooking(BaseModel):
    '''A request to CANCEL and existing booking '''
    customer_name: str
    Movie_title: str


''' Once you're working at the Agent level (§3.7 onward), 
the equivalent, documented pattern is create_agent(model, response_format=BookingRequest), 
and the result is read from result["structured_response"] rather than the bare return value '''

structured_model=model.with_structured_output(booking_request)
result=structured_model.invoke("i would like to book intersteller at the 7 pm show")

print(result)
print(result.action)

@tool
def peek_show(movie_title:str)->str :
    '''check show time for the movies'''
    print("I am called")
    return "7PM and 10:15 PM"

incomplete_model=model.bind_tools([peek_show]).with_structured_output(booking_request)
result=incomplete_model.invoke("Is Interstellar showing tonight? Book 2 seats for Rohan")
result

union_agent=create_agent(model=model,tools=[], response_format=(Union[NewBooking,CancelBooking]))

result=union_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "I want to cancel my movie Oppenheimer, I am Mayank"
        }
    ]
})

result['structured_response']


class MovieShows(BaseModel):
  name : str
  timing:str

response = model.with_structured_output(MovieShows).invoke("Is Interstellar showing tonight at 7pm at the Downtown cinema ?")

print(response)