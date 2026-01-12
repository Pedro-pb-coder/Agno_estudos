from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.youtube_reader import YouTubeReader
from agno.vectordb.pgvector import PgVector

from agno.models.openai import OpenAIChat
from agno.tools.openai import OpenAITools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
import os
# Verificação de segurança da API Key
load_dotenv()
# Verificação de segurança da API Key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Erro: OPENAI_API_KEY não encontrada no arquivo .env")
    exit()


db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Create Knowledge Instance
knowledge = Knowledge(
    name="YouTube Knowledge Base",
    description="Knowledge base from YouTube video transcripts",
    vector_db=PgVector(
        table_name="youtube_vectors", 
        db_url=db_url
    ),
)


# Add YouTube video content synchronously
knowledge.add_content(
    metadata={"source": "youtube", "type": "educational"},
    url="https://www.youtube.com/watch?v=6pJ4o3jJ2cU&t=4s",

    reader=YouTubeReader(),
)

# Create an agent with the knowledge
agent = Agent(
    model= OpenAIChat(id="gpt-5-nano",
                      instructions=["fale primeiro quais são os dados que existem em knowkedge"]
                      ),
    tools=[OpenAITools(all),DuckDuckGoTools()],
    knowledge=knowledge,
    search_knowledge=True,
    markdown=True,
    debug_mode=True,
)

"""""
# Query the knowledge base
agent.print_response(
    "O que esta salvo em knowledge",
    markdown=True
)
"""""
while True:
    pergunta = input("\nVocê: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        break
    
    # O agente processa a pergunta e busca no DuckDB
    agent.print_response(pergunta, stream=True)