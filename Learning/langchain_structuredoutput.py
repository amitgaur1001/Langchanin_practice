import os
from dotenv import load_dotenv
import time
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from pydantic import BaseModel,Field

load_dotenv()

model=init_chat_model(model="llama-3.3-70b-versatile",model_provider="groq",max_tokens=1024,timeout=30,temperature=0.7)

class email(BaseModel):
    subject:str=Field(description="subject of the email")
    body:str=Field(description="body of the email")
    sender:str=Field(description="sender of the email")
    recipient:str=Field(description="recipient of the email")

model_with_structure=model.with_structured_output(email)
response=model_with_structure.invoke("write a leave request email to my manager.")

print(response)
print(type(response))

booking_requests = [
    "Hi, I'd like 2 tickets for Interstellar at the 7pm show tonight, name is Priya.",
    "can u book me a seat for the 9:30 showing of dune part two? im rohan",
    "URGENT - need to CANCEL my booking for Oppenheimer, confirmation was under Aisha",
]

for request in booking_requests:
    response_unstructured=model.invoke(f"Extract the customer's name, movie, and what they want (book or cancel) from: {request}")
    print(response_unstructured.content)


