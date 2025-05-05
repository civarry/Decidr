import requests
import json
import hashlib
from typing import List, Dict, Any, Optional

class LLMClient:
    """
    Client for interacting with LLM APIs like Groq.
    Handles API calls and embedding generation.
    """
    
    def __init__(self, api_key: str, model: str):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key for the LLM service
            model: The LLM model to use
        """
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Call the LLM API to get a response.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The LLM's response as a string
        """
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse OpenAI-compatible response format
            if "choices" in response.json():
                return response.json()["choices"][0]["message"]["content"]
            # Fallback for other formats
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
        Get embedding for text using the LLM.
        
        In a production system, you would use a dedicated embedding model.
        This is a simplified approach for the prototype.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding
        """
        # For simplicity, we'll use the LLM to generate embeddings
        response = self.call_llm(
            prompt=f"Convert the following text to a concise vector representation by listing its key features and concepts: {text}",
            max_tokens=100
        )
        print(response)
        
        # For now, we'll use a simple hash-based approach for demo purposes
        hash_value = hashlib.md5(response.encode()).digest()
        # Convert to 32 floats between -1 and 1
        return [(b / 127.5) - 1 for b in hash_value]