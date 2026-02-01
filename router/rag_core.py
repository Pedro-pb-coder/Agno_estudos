import os
import re
from agno.agent import Agent
from agno.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.agentic import AgenticChunking
from agno.vectordb.pgvector import PgVector
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.models.openai import OpenAIChat
from agno.db.postgres import PostgresDb
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.learn import LearningMachine, SessionContextConfig
from dotenv import load_dotenv
load_dotenv()
# Configurações Globais
DB_URL = os.environ["db_url"] 


# --- Função Auxiliar para Nomes de Tabela Seguros ---
def sanitize_table_name(prefix: str, user_id: str, session_id: str) -> str:
    """
    Cria um nome de tabela seguro para SQL:
    Ex: user="Pedro", session="Carros 2024" -> "vec_pedro_carros_2024"
    """
    raw_name = f"{prefix}_{user_id}_{session_id}"
    # Substitui qualquer coisa que não seja letra ou número por underline
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', raw_name).lower()
    return clean_name

# --- Função Auxiliar para Nomes de Tabela Seguros ---
def sanitize_table_memory_name(prefix: str, user_id: str) -> str:
    """
    Cria um nome de tabela seguro para SQL:
    Ex: user="Pedro", session="Carros 2024" -> "vec_pedro_carros_2024"
    """
    raw_name = f"{prefix}_{user_id}"
    # Substitui qualquer coisa que não seja letra ou número por underline
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', raw_name).lower()
    return clean_name


# --- 1. Configurar Banco de Vetores ---
def get_vector_db(user_id: str, session_id: str):
    """Configura a conexão com o PgVector específica para o user/session.."""
    # Gera nome: ex: vectors_pedro_sessao1
    dynamic_table = sanitize_table_name("vec", user_id, session_id)

    return PgVector(
        table_name=dynamic_table, # adicionar user_id e session_id
        db_url=DB_URL,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"), 
    )

# --- 2. Configurar Memória (Histórico de Chat) ---
def get_storage(user_id: str):
    # Gera nome: ex: mem_pedro_sessao1
    dynamic_table = sanitize_table_memory_name("mem", user_id)

    return PostgresDb(
        db_url=DB_URL,
        #adicionar tabela de session_ids para o escopo das sessões serem separados 
        memory_table=dynamic_table,
    )

# --- 3. Processar PDF (Lógica Pura) ---
def process_pdf(file_path: str, user_id: str, session_id: str):
    """Lê o PDF e salva no PgVector"""
    try:
        reader = PDFReader(chunk=True)
        # agentic chunking habilitado
        reader2 =PDFReader(
             name="Agentic Chunking Reader",
             chunking_strategy=AgenticChunking(),
              )
                
        documents = reader.read(file_path)
        
        vector_db = get_vector_db(user_id, session_id)
        knowledge_base = Knowledge(vector_db=vector_db,readers=PDFReader(chunk=True))
        
        # Carrega os documentos
        knowledge_base.add_content(path=file_path,skip_if_exists= True,  reader=reader2)
        return True, f"Processado na tabela: {vector_db.table_name}"
    except Exception as e:
        return False, str(e)

# --- 4. Criar o Agente ---
def get_agent(api_key,user_id: str, session_id: str):
    """
    Cria o agente. O session_id é crucial para a API saber 
    quem é o usuário e buscar o histórico correto.
    """
    vector_db = get_vector_db(user_id, session_id)
    knowledge_base = Knowledge(vector_db=vector_db,readers= PDFReader(chunk=True))
    storage = get_storage(user_id)
    
    return Agent(
        model=OpenAIChat(id="gpt-5-nano",api_key=api_key), # Ajustado para modelo existente
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