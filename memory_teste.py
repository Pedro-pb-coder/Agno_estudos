from agno.agent import Agent
from agno.db.sqlite import SqliteDb

# Setup the SQLite database
db = SqliteDb(db_file="tmp/data.db")

# Setup a basic agent with the SQLite database
agent = Agent(
    db=db,
    enable_user_memories=True,
    markdown=True,
    debug_mode=True,
)

while True:
    pergunta = input("\nVocê: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        break
    
    # O agente processa a pergunta e busca no DuckDB
    agent.print_response(pergunta, stream=True)

#agent.print_response("My name is John Doe and I like to play basketball on the weekends.")
#agent.print_response("What's do I do in weekends?")