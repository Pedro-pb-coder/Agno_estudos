import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
# Ajuste no import do Embedder
#from agno.embedders.openai import OpenAIEmbedder
#from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.embedder.openai import OpenAIEmbedder


from agno.knowledge.reader.youtube_reader import YouTubeReader
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector

from agno.tools.duckdb import DuckDbTools

#from agno.vectordb.duckdb.duckdb import DuckDb
#Sfrom agno.vectordb.duckdb import DuckDb


from agno.tools.youtube import YouTubeTools

# Carrega chaves de API do .env para usar agno . 
load_dotenv()

# Verificação de segurança da API Key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Erro: OPENAI_API_KEY não encontrada no arquivo .env")
    exit()

# 1- 

# Caminho do banco de dados local
DATABASE_FILE = "repositorio_youtube.duckdb"

# --- CONFIGURAÇÃO DO BANCO DE DADOS LOCAL ---
# O DuckDB criará um arquivo chamado 'biblioteca_videos.duckdb' na sua pasta.
vector_db = DuckDbTools(
    table_name="meus_videos",
    db_url=DATABASE_FILE,
    embedder=OpenAIEmbedder(model="text-embedding-3-small")
)

# 2. Configuração do Repositório de Vídeos (Knowledge Base)
knowledge_base = YouTubeReader(
    urls=[
        # Exemplo: videos longos do canal  efap sobre stranger things.
        "https://www.youtube.com/watch?v=9H8EJLN9qXU", 
        #"https://www.youtube.com/watch?v=6pJ4o3jJ2cU", 
    ],
    vector_db=vector_db,
)

knowledge = Knowledge(
    vector_db=PgVector(table_name="youtube_vectors", db_url=DATABASE_FILE),
)
knowledge.add_content(
    urls=["https://www.youtube.com/watch?v=9H8EJLN9qXU"],
    reader=YouTubeReader(),
)

# 3. Definição do Agente
agent = Agent(
    name="Local_YouTube_Bot",
    model=OpenAIChat(id="gpt-5-nano"),
    tools=[YouTubeTools()],
    knowledge=knowledge,
    search_knowledge=True, # Habilita o RAG (Busca no banco)
    read_chat_history=True,
    instructions=[
        "Você é um assistente técnico especializado em análise de conteúdo de vídeo.",
        "Utilize o DuckDB para buscar informações no repositório de vídeos indexados.",
        "Sempre cite o vídeo específico de onde a informação foi extraída.",
        "Caso o usuário peça uma análise comparativa, use os dados disponíveis no conhecimento.",
        "Responda em português brasileiro."
    ],
    markdown=True,
    debug_mode=True
)

# 4. Inicialização: Carrega os vídeos para o DuckDB
# recreate=False garante que ele não tente baixar tudo de novo se o arquivo já existir
print("📚 Indexando vídeos no DuckDB... Aguarde.")
knowledge.load(recreate=False)


# 2. Loop de interação via terminal
print("\n✅ Agente Pronto! Digite sua pergunta sobre os vídeos (ou 'sair'):")

while True:
    pergunta = input("\nVocê: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        break
    
    # O agente processa a pergunta e busca no DuckDB
    agent.print_response(pergunta, stream=True)

"""""

while True:
    pergunta = input("\nVocê: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        break
    
    # O agente processa a pergunta e busca no DuckDB
    agent.print_response(pergunta, stream=True)


if __name__ == "__main__":
    # Exemplo de pergunta para testar o RAG
    agent.print_response(
        "Explique o que foi discutido nos vídeos .", 
        stream=True
    )

"""""