from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.openai import OpenAITools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
import os

# Verificação de segurança da API Key
load_dotenv()
# Verificação de segurança da API Key
print(os.getenv("cavalo"))
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Erro: OPENAI_API_KEY não encontrada no arquivo .env")
    exit()


agent = Agent(
    model= OpenAIChat(id="gpt-5-nano",
                      instructions=["responda tudo como se fosse jornalista"]
                      ),
    tools=[OpenAITools(all),DuckDuckGoTools()],
    debug_mode=True,
    markdown=True
                                
    )

#agent.print_response("qual a previsão dotempo para uberlandia mg dia 12/01/2026 ? ")

while True:
    pergunta = input("\nVocê: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        break
    
    # O agente processa a pergunta e busca no DuckDB
    agent.print_response(pergunta, stream=True)