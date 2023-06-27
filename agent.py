from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.utilities import SerpAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
import faiss
from langchain.agents import Tool
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI
# import docun

search = SerpAPIWrapper()

tools = [
    # Tool(
    #     name="Doc Understanding",
    #     func=docun.docun,
    #     description="Useful when you want information from documents"
    # ),
    Tool(
        name="Search",
        func=search.run,
        description="Useful when you want information about current events"
    ),
    WriteFileTool(),
    ReadFileTool(),
]

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})

agent = AutoGPT.from_llm_and_tools(
    ai_name="LawyerAI",
    ai_role="Lawyer",
    tools=tools,
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    memory=vectorstore.as_retriever()
)

# Set verbose to be true
agent.chain.verbose = True

prompt = """
You are LawyerAI, an AI agent that wants to write a statement of facts based on the evidence you're provided. 
Your goals: 
1. Read template.txt to understand the format of the statement of facts you're going to write. 
2. Read evidence.txt to understand the facts of the case
3. Search online for additional information to find any more information you still need to write the statement of facts.
4. Write the statement of facts.
"""
agent.run([prompt])
