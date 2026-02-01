import os
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from dotenv import load_dotenv

# Importações do Agno e componentes

from router.rag_core import process_pdf,get_agent

# --- Configurações Iniciais ---
# Verificação de segurança da API Key
load_dotenv()
app = FastAPI(title="API Agente RAG", version="1.0")

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Erro: OPENAI_API_KEY não encontrada no arquivo .env")
    exit()

api_key = os.environ["OPENAI_API_KEY"] 
DB_URL = os.environ["db_url"] 

TEMP_DIR = "temp_pdfs"
# Garantir que a pasta temporária existe
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Modelos de Dados (Pydantic) ---
class ChatRequest(BaseModel):
    message: str
    user_id: str       # Obrigatório para isolamento
    session_id: str    # Obrigatório para isolamento
    
class ChatResponse(BaseModel):
    response: str

# --- Funções Auxiliares ---
# no router
# --- ROTAS DA API ---

@app.get("/")
def health_check():
    return {"status": "ok", "message": "API do Agente RAG está rodando!"}

@app.post("/upload")
def upload_pdf(
    file: UploadFile = File(...), 
    user_id: str = Form(...),    # Recebe como form-data
    session_id: str = Form(...)  # Recebe como form-data
):
    """
    Faz upload e salva em uma tabela SQL específica para user_id + session_id
    """
    try:
        # 1. Salvar arquivo
        # Dica: Adicionar user_id ao nome do arquivo evita colisão de nomes no disco
        safe_filename = f"{user_id}_{session_id}_{file.filename}"
        file_path = os.path.join(TEMP_DIR, safe_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Chamar a lógica do core passando os IDs
        success, message = process_pdf(file_path, user_id, session_id)
        
        # Limpeza opcional: remover arquivo local após processar
        # os.remove(file_path) 
        
        if not success:
            raise HTTPException(status_code=500, detail=message)
            
        return {
            "status": "success", 
            "filename": file.filename, 
            "target_table": message # Retorna onde foi salvo para debug
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Recebe uma pergunta JSON {"message": "..."} e retorna a resposta do agente.
    """
    try:
        agent = get_agent(
            api_key=api_key,
            user_id=request.user_id,
            session_id=request.session_id
            )
        
        # Executa o agente (stream=False para pegar a resposta inteira de uma vez)
        response_obj = agent.run(request.message, stream=False)
        
        # O objeto de resposta do Agno geralmente tem .content 
        answer_text = response_obj.content 
        
        return {"response": answer_text}
    
        
    except Exception as e:
        print(f"Erro no chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)