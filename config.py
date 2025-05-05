import os
import secrets
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(16))
    BOOTSTRAP_BOOTSWATCH_THEME = 'flatly'
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    LLM_MODEL = os.environ.get('LLM_MODEL', "meta-llama/llama-4-scout-17b-16e-instruct")
    DATA_DIR = os.environ.get('DATA_DIR', 'decision_data')