import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings


load_dotenv()


COURSES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/asciidoc")

# Load lesson documents
loader = DirectoryLoader(COURSES_PATH, glob="**/lesson.adoc", loader_cls=TextLoader)
docs = loader.load()

# Create a text splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

# Split documents into chunks
chunks = text_splitter.split_documents(docs)
print(len(chunks))

# Create a Neo4j vector store
neo4j_db = Neo4jVector.from_documents(
    chunks,
    OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="neo4j",  
    index_name="chunkVector",
    node_label="Chunk", 
    text_node_property="text", 
    embedding_node_property="embedding",  
)