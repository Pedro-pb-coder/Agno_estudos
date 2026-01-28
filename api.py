import os
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Importações do Agno e componentes

from router.rag_core import process_pdf,get_agent

# --- Configurações Iniciais ---
app = FastAPI(title="API Agente RAG", version="1.0")

# Substitua pela sua chave ou garanta que está no ambiente
# os.environ["OPENAI_API_KEY"] = "sk-..." 

DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"
TEMP_DIR = "temp_pdfs"

# Garantir que a pasta temporária existe
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Modelos de Dados (Pydantic) ---
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session" # O frontend pode enviar um ID único por usuário
    
class ChatResponse(BaseModel):
    response: str

# --- Funções Auxiliares ---
# no router
# --- ROTAS DA API ---

@app.get("/")
def health_check():
    return {"status": "ok", "message": "API do Agente RAG está rodando!"}

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    # 1. Salvar arquivo
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Chamar a lógica do core
    success, message = process_pdf(file_path)
    
    if not success:
        raise HTTPException(status_code=500, detail=message)
        
    return {"status": "success", "filename": file.filename}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Recebe uma pergunta JSON {"message": "..."} e retorna a resposta do agente.
    """
    try:
        agent = get_agent(session_id=request.session_id)
        
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