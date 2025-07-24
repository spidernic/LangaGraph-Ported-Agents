from typing import Any, ClassVar, Dict, List, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

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

        # Fixed system prompt for coding assistance (assumed based on description)
        self.system_prompt = (
            "You are a highly skilled coding assistant. Your role is to write code, analyze information, "
            "and create artifacts based on the tasks provided. Use your expertise to generate accurate "
            "and efficient code solutions. Do not deviate from coding-related tasks."
        )

        # Define the state
        class AgentState typing.TypedDict:
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
