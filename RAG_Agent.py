from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model='gpt-5-nano', temperature=0)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

pdf_file = "Trilha-de-Engenharia-de-IA-Agentica.pdf"
name_file_without_ext = os.path.splitext(os.path.basename(pdf_file))[0]
persist_directory = os.path.join(os.getcwd(), "chroma_db")

if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
    print("Vector Store found. Loading...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=name_file_without_ext
    )
else:
    print("Vector Store not found. Creating...")
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")
        
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=name_file_without_ext
    )

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

@tool
def retriever_tool(query: str) -> str:
    """Searches for information within the PDF document."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    return "\n\n".join([f"Doc {i+1}: {d.page_content}" for i, d in enumerate(docs)])

tools = [retriever_tool]
llm_with_tools = llm.bind_tools(tools)

def call_model(state: MessagesState):
    messages = state['messages']
    if not isinstance(messages[0], SystemMessage):
        sys_msg = SystemMessage(content=f"""
You are an intelligent AI assistant who answers questions about {name_file_without_ext} based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about {name_file_without_ext}. You can make multiple calls if needed.
If you need to look up information before asking a follow-up question, you are allowed to do that!
Please always cite the specific part of the documents you use in your answers.
""")
        messages = [sys_msg] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

def running_agent():
    thread_id = "user_session_1"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n--- Agent Started ---")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        events = app.stream(
            {"messages": [HumanMessage(content=user_input)]}, 
            config, 
            stream_mode="values"
        )

        for event in events:
            if "messages" in event:
                current_message = event["messages"][-1]

                if isinstance(current_message, AIMessage) and current_message.tool_calls:
                    for tool_call in current_message.tool_calls:
                        print(f"\n  REASONING: The Agent decided to call '{tool_call['name']}'")
                        print(f"    Arguments: {tool_call['args']}")

                elif isinstance(current_message, ToolMessage):
                    print(f"\n DATA: The tool '{current_message.name}' returned results.")
                    preview = str(current_message.content)[:150].replace('\n', ' ')
                    print(f"    Content: {preview}...")

                elif isinstance(current_message, AIMessage) and not current_message.tool_calls:
                    print(f"\n--- ANSWER ---:\n{current_message.content}")

if __name__ == "__main__":
    running_agent()