import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Initialize the Groq model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# Example tool defined in modelintegration notebook
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

# Create the agent
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful agent.",
)

if __name__ == "__main__":
    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "What is the weather like in New York?",
            }
        ]
    })
    print(response)
