import os
from agno.agent import Agent
from agno.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.agentic import AgenticChunking
from agno.vectordb.pgvector import PgVector
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.models.openai import OpenAIChat
from agno.db.postgres import PostgresDb
from agno.tools.duckduckgo import DuckDuckGoTools

# Configurações Globais
DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai" 

# --- 1. Configurar Banco de Vetores ---
def get_vector_db():
    """Configura a conexão com o PgVector."""
    return PgVector(
        table_name="pdf_documents",
        db_url=DB_URL,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"), 
    )

# --- 2. Configurar Memória (Histórico de Chat) ---
def get_storage():
    return PostgresDb(
        db_url=DB_URL,
        memory_table="user_memories_chat",
    )

# --- 3. Processar PDF (Lógica Pura) ---
def process_pdf(file_path: str):
    """Lê o PDF e salva no PgVector"""
    try:
        reader = PDFReader(chunk=True)
        # agentic chunking habilitado
        reader2 =PDFReader(
             name="Agentic Chunking Reader",
             chunking_strategy=AgenticChunking(),
              )
                
        documents = reader.read(file_path)
        
        vector_db = get_vector_db()
        knowledge_base = Knowledge(vector_db=vector_db,readers=PDFReader(chunk=True))
        
        # Carrega os documentos
        knowledge_base.add_content(path=file_path,skip_if_exists= True,  reader=reader2)
        return True, "Processamento concluído com sucesso."
    except Exception as e:
        return False, str(e)

# --- 4. Criar o Agente ---
def get_agent(session_id: str = "default_session"):
    """
    Cria o agente. O session_id é crucial para a API saber 
    quem é o usuário e buscar o histórico correto.
    """
    vector_db = get_vector_db()
    knowledge_base = Knowledge(vector_db=vector_db,readers= PDFReader(chunk=True))
    storage = get_storage()
    
    return Agent(
        model=OpenAIChat(id="gpt-5-nano"), # Ajustado para modelo existente
        knowledge=knowledge_base,
        search_knowledge=True,
        
        # Ferramentas
        tools=[DuckDuckGoTools()],
        
        # Memória Persistente no Postgres

        # acesso as conversas anteriores. para informar a resposta
        db=storage,         
        #storage=storage, 
        # Give the Agent the ability to update memories
        enable_agentic_memory=True,
        # OR - Run the MemoryManager automatically after each response
        enable_user_memories=True,
        # -------
        session_id=session_id, # Importante para API
        add_history_to_context=True,
        num_history_runs=5,
        
        # Configs
        markdown=True,
        debug_mode=True,
    )