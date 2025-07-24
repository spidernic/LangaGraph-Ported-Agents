from typing import Any, ClassVar, Dict, List, Annotated, TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

MAGENTIC_ONE_CODER_DESCRIPTION = "A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills."

MAGENTIC_ONE_CODER_SYSTEM_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use the 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible."""

class MagenticOneCoderAgent:
    """An agent, used by MagenticOne that provides coding assistance using an LLM model client.

    The prompts and description are sealed, to replicate the original MagenticOne configuration. 
    See AssistantAgent if you wish to modify these values.
    
    Ported to LangGraph.
    """

    component_provider_override: ClassVar[str | None] = 'autogen_ext.agents.magentic_one.MagenticOneCoderAgent'

    def __init__(self, name: str, model_client: BaseChatModel, **kwargs: Any):
        self.name = name
        self.model_client = model_client

        # Use the sealed description and system prompt from AutoGen
        self.description = MAGENTIC_ONE_CODER_DESCRIPTION
        self.system_prompt = MAGENTIC_ONE_CODER_SYSTEM_MESSAGE

        # Define the state
        class AgentState(TypedDict):
            messages: Annotated[List[AnyMessage], add_messages]

        # Node function: call the model
        def call_model(state: AgentState) -> Dict[str, List[AnyMessage]]:
            messages = state["messages"]
            # Ensure system prompt is included
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=self.system_prompt)] + messages
            response = self.model_client.invoke(messages)
            return {"messages": [response]}

        # Build the graph
        workflow = StateGraph(state_schema=AgentState)
        workflow.add_node("agent", call_model)
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)

        # Compile the graph with memory
        self.graph = workflow.compile(checkpointer=MemorySaver())

    def generate_reply(self, messages: List[Dict[str, str]], thread_id: str = "default") -> str:
        """Generate a reply based on the input messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            thread_id: Thread ID for stateful conversation.
        
        Returns:
            The generated reply content.
        """
        # Convert to LangChain messages
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        # Invoke the graph
        result = self.graph.invoke(
            {"messages": lc_messages},
            config={"configurable": {"thread_id": thread_id}}
        )

        # Return the last message content
        return result["messages"][-1].content
