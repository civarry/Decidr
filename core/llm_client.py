import requests
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

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
        
        # Initialize the SentenceTransformer model for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
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
        Get embedding for text using the SentenceTransformer model.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding
        """
        # Use the SentenceTransformer model to get embeddings
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            
            # Convert numpy array to list for serialization
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return a zero vector as fallback (adjust dimension if needed)
            return [0.0] * 384  # all-MiniLM-L6-v2 has 384 dimensions