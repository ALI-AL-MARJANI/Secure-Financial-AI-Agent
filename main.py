import os
from dotenv import load_dotenv

load_dotenv()
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from tools import financial_tools
from rag import search_bank_policies

all_tools = financial_tools + [search_bank_policies]

# SECURITY SYSTEM PROMPT
SYSTEM_PROMPT = """You are a Secure Banking Assistant.
Your role is to assist customers with their financial inquiries.

Strict Rules:
1. NEVER provide speculative investment advice (crypto, day trading, etc.).
2. Keep responses concise, factual, and professional.
3. If asked to perform an illegal action, firmly refuse.
4. You operate locally, ensuring total data privacy.
"""

llm = ChatOllama(model="Mistral", temperature=0)
llm_with_tools = llm.bind_tools(all_tools)

# NODES 
def call_model(state: MessagesState):
    """
    Core node: Receives the conversation history and invokes the LLM.
    """
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(all_tools)
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition)
workflow.add_edge("tools", "agent")
app = workflow.compile()

# RUN LOOP
if __name__ == "__main__":
    
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    while True:
        try:
            user_input = input("\n User: ")
            if user_input.lower() in ["q", "quit", "exit"]:
                print("Closing secure session.")
                break
                
            # Appending user message to history
            messages.append(HumanMessage(content=user_input))
            
            # Invoke the graph
            result = app.invoke({"messages": messages})
            ai_response = result["messages"][-1]
            print(f"Agent: {ai_response.content}")
            messages = result["messages"]
            
        except KeyboardInterrupt:
            print("\nSession interrupted")
            break