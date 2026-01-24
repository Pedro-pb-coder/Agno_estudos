import os
import shutil
import streamlit as st
from agno.agent import Agent
#
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.agentic import AgenticChunking
#from agno.knowledge.pdf import PDFKnowledgeBase


from agno.vectordb.pgvector import PgVector
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.models.openai import OpenAIChat

# memory
from agno.db.postgres import PostgresDb

# --- Config da Página ---
st.set_page_config(page_title="Agente RAG com Agno & PgVector", layout="wide")
st.title("🤖 Agente de Conhecimento (PDF + PgVector)")

# --- Sidebar:  ---
with st.sidebar:
    st.header("Configurações")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    # URL de conexão com o db
    # 
    db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"  #"postgresql+psycopg://admin:adminpassword@localhost:5432/vector_db"

    st.info("Certifique-se que o container Docker do Postgres está rodando.")

if not openai_api_key:
    st.warning("Por favor, insira sua OpenAI API Key na barra lateral para continuar.")
    st.stop()

# variável de ambiente para o Agno/OpenAI
os.environ["OPENAI_API_KEY"] = openai_api_key

# Configura o armazenamento das conversas no mesmo banco do Docker
db = PostgresDb(
  db_url=db_url,
  memory_table="user_memories_chat",  # Optionally specify a table name for the memories
)


# --- Lógica do Banco de Vetores (PgVector) ---
def get_vector_db():
    """Configura a conexão com o PgVector."""
    return PgVector(
        table_name="pdf_documents",
        db_url=db_url,
        # O embedder define como o texto vira vetor. O padrão é text-embedding-3-small (1536 dimensoes)
        embedder=OpenAIEmbedder(id="text-embedding-3-small"), 
    )

# --- Upload e Processamento de PDF ---
uploaded_file = st.file_uploader("Faça upload de um arquivo PDF", type="pdf")

if uploaded_file:
    # Salvar o arquivo temporariamente
    with st.spinner("Salvando arquivo..."):
        temp_dir = "temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    st.success(f"Arquivo '{uploaded_file.name}' carregado!")

    # Botão para processar (Criar Embeddings)
    if st.button("🧠 Alimentar Knowledge Base (Criar Embeddings)"):
        with st.spinner("Lendo PDF e salvando vetores no Postgres... Isso pode levar um momento."):
            try:
                # ---- adicionar por etapas 
                # 1. Ler o PDF explicitamente usando o Reader
                reader = PDFReader(chunk=True)
                documents = reader.read(file_path)
                # knowledge_base.add_content(path=file_path,  readers=PDFReader(chunk=True))


                # 1. Configurar Knowledge Base apontando para o arquivo local
                knowledge_base = Knowledge( # PDFKnowledgeBase
                    
                    #path=file_path,
                    vector_db=get_vector_db(),
                    readers=PDFReader(chunk=True)
                )
                
                # 2. Carregar: Lê o PDF, quebra em chunks, cria embeddings e salva no Postgres


                #knowledge_base._load_content(path=file_path, recreate=False) # recreate=False evita apagar dados anteriores
                
                knowledge_base.add_content(path=file_path,skip_if_exists= True,  reader=PDFReader(chunk=True))

                # agentic chunking 
                
                knowledge_base.add_content(
                    path=file_path,
                    skip_if_exists= True,
                    reader=PDFReader(
                        name="Agentic Chunking Reader",
                        chunking_strategy=AgenticChunking(),

                    )
                )


                #knowledge_base.insert(path="cookbook/08_knowledge/testing_resources/cv_1.pdf",reader=PDFReader())
                
                st.success("Sucesso! O PDF foi processado e salvo no PgVector.")
            except Exception as e:
                st.error(f"Erro ao processar o PDF: {e}")

# --- Interface de Chat ---
st.divider()
st.subheader("Converse com o Agente")

# Inicializar histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input 
if prompt := st.chat_input("Faça uma pergunta sobre o PDF..."):
    # Guardar pergunta no histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gerar resposta
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Instanciar o Agente com o Knowledge Base conectado ao PgVector
        # Nota: Instanciamos aqui para garantir que ele pegue o estado atual do DB
        try:
            vector_db = get_vector_db()
            knowledge_base = Knowledge(
                #path="temp_pdfs", # Aponta para a pasta (o agente pode buscar em tudo que está lá/no banco)
                vector_db=vector_db,
                readers=PDFReader(chunk=True)
            )
            
            agent = Agent(
                model=OpenAIChat(id="gpt-5-nano"),
                knowledge=knowledge_base,
                search_knowledge=True, # Força o agente a buscar no banco de vetores
                #show_tool_calls=True,

                # acesso as conversas anteriores. para informar a resposta
                db=db,
                # Give the Agent the ability to update memories
                enable_agentic_memory=True,
                # OR - Run the MemoryManager automatically after each response
                enable_user_memories=True,


                debug_mode=True,
                markdown=True,
                instructions=["Use sempre o knowledge base para responder. Se não encontrar, diga que não sabe."]
            )
            
            # Executar o agente (stream=True para efeito de digitação)
            full_response = ""
            # O metodo run do Agno retorna um gerador se stream=True
            response_generator = agent.run(prompt, stream=True)
            
            for chunk in response_generator:
                if chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # Guardar resposta no histórico
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Erro ao gerar resposta: {e}")

# Limpeza (Opcional): Remover arquivos temporários ao reiniciar/fechar pode ser implementado depois