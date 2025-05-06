import requests
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

class LLMClient:
    """
    Client for LLM API interaction and embedding generation.
    """
    
    def __init__(self, api_key: str, model: str):
        """
        Initialize with API credentials and embedding model.
        """
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Send prompt to LLM API and return the response.
        """
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            if "choices" in response.json():
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"Unknown API response format: {response.json()}")
                return "Error: Unknown API response format"
        except Exception as e:
            print(f"LLM API call failed: {str(e)}")
            if 'response' in locals():
                print(f"Response: {response.text if hasattr(response, 'text') else 'No response'}")
            return f"Error calling LLM API: {str(e)}"
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Return embedding vector for input text.
        """
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return [0.0] * 384
