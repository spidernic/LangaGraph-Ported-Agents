from langchain_openai import ChatOpenAI

# Initialize the model client (example with OpenAI)
model_client = ChatOpenAI(model="gpt-4o", api_key="your-api-key")

# Create the agent
agent = MagenticOneCoderAgent(name="CodeAssistant", model_client=model_client)

# Example messages in AutoGen format
messages = [
    {"role": "user", "content": "Write a Python function to calculate factorial."}
]

# Generate a reply
reply = agent.generate_reply(messages, thread_id="session_1")
print(reply)

# Continue the conversation
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "Can you optimize it?"})
reply = agent.generate_reply(messages, thread_id="session_1")
print(reply)
