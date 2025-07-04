import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langgraph.graph import MessagesState , StateGraph,END,START
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage

load_dotenv()

llm = AzureChatOpenAI(
     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    # openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key= os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4o",
    temperature=0.7,
)

def search(query: str) -> str:
    """Search the web for a query"""
    return DuckDuckGoSearchRun().invoke({"query": query})

search = DuckDuckGoSearchRun()

tools = [search]

llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(
    content="You are a financial advisory assistant that can perform web searches(to get history and current news) to answer questions and also analysises the results of those searches then provide the infromation about stocks, bond etc and advise also to buy or not "
)


def reasoner(state: MessagesState) -> AnyMessage:
     return {"messages": [llm_with_tools.invoke([sys_msg]+ state["messages"])]}

#making graph

builder = StateGraph(MessagesState)

#adding nodes
builder.add_node(
    "reasoner",
    reasoner,
    
)
builder.add_node("tools",ToolNode(tools))

#add edges
builder.add_edge(START, "reasoner")
builder.add_conditional_edges(
    "reasoner",
    tools_condition,
)

builder.add_edge("tools", "reasoner")
finanacial_graph = builder.compile()


# user_input = st.text_input("Enter the stock or financial instrument you want advice on:")
# messages = [HumanMessage(
#     content = "advise me on the current stock market trends and whether I should buy or sell stocks of {user_input}"
# )]

# messages = finanacial_graph.invoke({"messages": messages})
st.set_page_config("Financial Advisor", page_icon="💹", layout="centered")
st.title("💹 AI‑Powered Financial Advisor")
st.caption("Get real‑time, research‑backed buy/sell guidance for any stock or instrument.")

with st.container(border=True):
    user_input = st.text_input(
        "Ticker or Instrument",
        placeholder="e.g. AAPL, BTC‑USD, 10‑year T‑bond…",
        label_visibility="visible",
    )
    submit = st.button("🔍 Analyze")

if submit and user_input:
    with st.spinner("Crunching numbers and scouring the web…"):
        prompt = (
            f"Advise me on the current stock market trends and whether I should buy or sell "
            f"stocks of {user_input}"
        )
        messages = [HumanMessage(content=prompt)]
        result = finanacial_graph.invoke({"messages": messages})
        assistant_resp = (
            result["messages"][-1].content
            if result and "messages" in result
            else "Sorry—no response from the advisor."
        )

    st.divider()
    st.subheader("📊 Advisor’s Insight")
    st.write(assistant_resp)

elif submit:
    st.warning("Please enter a ticker or instrument first.")