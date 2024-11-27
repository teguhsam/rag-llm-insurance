import os
from dotenv import load_dotenv

MODEL = "gpt-4o-mini"
DB_NAME = "vector_db"

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')