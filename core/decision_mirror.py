import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.llm_client import LLMClient
from models.decision import Decision

from chromadb import Client, Settings
from chromadb.utils import embedding_functions


class DecisionMirror:
    """
    A system that learns from your decisions and can simulate how you would respond
    to similar situations in the future.
    """
    
    def __init__(self, api_key: str, model: str, data_dir: str = "decision_data"):
        # Initialize the LLM client
        self.llm_client = LLMClient(api_key=api_key, model=model)
        
        # Create data directory if it doesn't exist
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Path to the decisions database
        self.decisions_path = self.data_dir / "decisions.json"
        
        # Load existing decisions or create empty database
        self.decisions = self._load_decisions()
        
        # Initialize Chroma client
        self.chroma_dir = self.data_dir / "chroma_db"
        self.chroma_dir.mkdir(exist_ok=True)  # Ensure chroma directory exists
        
        # CHANGE: Use PersistentClient instead of Client with Settings
        # This forces ChromaDB to use persistent storage
        try:
            self.chroma_client = PersistentClient(path=str(self.chroma_dir))
            print(f"Successfully initialized PersistentClient at {self.chroma_dir}")
        except Exception as e:
            print(f"Error initializing PersistentClient: {e}")
            # Fall back to regular client if PersistentClient fails
            self.chroma_client = Client(Settings(
                persist_directory=str(self.chroma_dir),
                anonymized_telemetry=False,
                is_persistent=True  # Explicitly set persistence flag
            ))
            print(f"Falling back to regular Client with is_persistent=True")
        
        # Create or get the collection
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Check if collection exists
        collection_names = self.chroma_client.list_collections()
        collection_exists = any(c.name == "decisions" for c in collection_names)
        
        if collection_exists:
            self.collection = self.chroma_client.get_collection(
                name="decisions",
                embedding_function=self.embedding_function
            )
        else:
            self.collection = self.chroma_client.create_collection(
                name="decisions",
                embedding_function=self.embedding_function
            )
        
        # Ensure all decisions are in Chroma
        self._sync_decisions_to_chroma()
    
    def _load_decisions(self) -> List[Decision]:
        """Load existing decisions from file or create empty list."""
        if self.decisions_path.exists():
            with open(self.decisions_path, 'r') as f:
                decision_dicts = json.load(f)
                return [Decision.from_dict(d) for d in decision_dicts]
        return []
    
    def _save_decisions(self) -> None:
        """Save decisions to file."""
        decision_dicts = [d.to_dict() for d in self.decisions]
        with open(self.decisions_path, 'w') as f:
            json.dump(decision_dicts, f, indent=2)

    def _sync_decisions_to_chroma(self):
        """Ensure all decisions are in the Chroma database"""
        # Get existing decision IDs in Chroma
        if self.collection.count() == 0:
            existing_ids = []
        else:
            existing_ids = [str(id) for id in self.collection.get()["ids"]]
        
        # Add any missing decisions to Chroma
        for decision in self.decisions:
            decision_id = str(decision.id)
            if decision_id not in existing_ids:
                text = self._prepare_text_for_embedding(decision)
                metadata = {
                    "problem": decision.problem,
                    "chosen": decision.chosen,
                    "mood": decision.mood or ""
                }
                self.collection.add(
                    ids=[decision_id],
                    documents=[text],
                    metadatas=[metadata]
                )
        
        # Note: With ChromaDB with a persist_directory, 
        # persistence should happen automatically
    
    def add_decision(self, problem: str, options: List[str], chosen: str, 
                    reasoning: Optional[str] = None, mood: Optional[str] = None) -> Decision:
        # Create new decision with next ID
        new_id = len(self.decisions) + 1
        decision = Decision.create(
            id=new_id,
            problem=problem,
            options=options,
            chosen=chosen,
            reasoning=reasoning,
            mood=mood
        )
        
        # Add decision to database
        self.decisions.append(decision)
        self._save_decisions()
        
        # Add to Chroma
        decision_text = self._prepare_text_for_embedding(decision)
        metadata = {
            "problem": decision.problem,
            "chosen": decision.chosen,
            "mood": decision.mood or ""
        }
        self.collection.add(
            ids=[str(decision.id)],
            documents=[decision_text],
            metadatas=[metadata]
        )
        
        return decision
    
    def _prepare_text_for_embedding(self, decision: Decision) -> str:
        """
        Prepare decision text for embedding in a consistent format.
        
        Args:
            decision: The decision to prepare text for
            
        Returns:
            Formatted text ready for embedding
        """
        text_parts = [
            f"Problem: {decision.problem}",
            f"Options: {', '.join(decision.options)}",
            f"Chosen: {decision.chosen}"
        ]
        
        if decision.reasoning:
            text_parts.append(f"Reasoning: {decision.reasoning}")
        if decision.mood:
            text_parts.append(f"Mood: {decision.mood}")
            
        return " ".join(text_parts)
    
    def find_similar_decisions(self, problem: str, top_k: int = 3) -> List[Decision]:
        """Find decisions similar to the given problem."""
        if not self.decisions:
            return []
        
        # Query Chroma for similar decisions
        results = self.collection.query(
            query_texts=[problem],
            n_results=top_k
        )
        
        # Map results back to Decision objects
        similar_decisions = []
        for id in results["ids"][0]:  # First element because we have a single query
            decision_id = int(id)
            decision = self.get_decision(decision_id)
            if decision:
                similar_decisions.append(decision)
        
        return similar_decisions
    
    def predict_decision(self, 
                        problem: str, 
                        options: List[str]) -> Dict[str, Any]:
        """
        Predict how you would decide in a given situation.
        
        Args:
            problem: The problem or situation
            options: List of possible options
            
        Returns:
            Prediction including chosen option, reasoning, and confidence
        """
        # Find similar past decisions
        similar_decisions = self.find_similar_decisions(problem)
        
        if not similar_decisions:
            return {
                "chosen": None,
                "reasoning": "Not enough past decisions to make a prediction.",
                "confidence": 0.0,
                "similar_decisions": []
            }
        
        # Format similar decisions for the LLM
        similar_decisions_text = ""
        for i, decision in enumerate(similar_decisions):
            similar_decisions_text += f"\nPAST DECISION #{i+1}:\n"
            similar_decisions_text += f"Problem: {decision.problem}\n"
            similar_decisions_text += f"Options: {', '.join(decision.options)}\n"
            similar_decisions_text += f"Chosen: {decision.chosen}\n"
            if decision.reasoning:
                similar_decisions_text += f"Reasoning: {decision.reasoning}\n"
            if decision.mood:
                similar_decisions_text += f"Mood: {decision.mood}\n"
        
        # Create prompt for the LLM
        prompt = f"""
        You are simulating a specific person's decision-making process based on their past decisions. 
        Your task is to predict how this person would decide in a new situation.

        NEW SITUATION:
        Problem: {problem}
        Options: {', '.join(options)}

        SIMILAR PAST DECISIONS:
        {similar_decisions_text}

        Based solely on these past decisions, predict:
        1. Which option the person would choose
        2. Their likely reasoning
        3. Your confidence in this prediction (0-100%)

        DO NOT HALLUCINATE. Only use patterns from the provided past decisions.
        Format your response as a JSON object with keys: "chosen", "reasoning", and "confidence".
        """
        
        # Get prediction from LLM
        response = self.llm_client.call_llm(prompt)
        
        # Parse the response
        try:
            # Extract JSON from the response
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                prediction = json.loads(json_match.group(1))
            else:
                prediction = json.loads(response)
            
            # Ensure the prediction has the expected keys
            if not all(key in prediction for key in ["chosen", "reasoning", "confidence"]):
                raise ValueError("Missing required keys in prediction")
                
            # Add similar decisions to the prediction
            prediction["similar_decisions"] = similar_decisions
            return prediction
        except Exception as e:
            # Fallback if parsing fails
            return {
                "chosen": None,
                "reasoning": f"Failed to parse prediction from model: {str(e)}",
                "confidence": 0.0,
                "raw_response": response,
                "similar_decisions": similar_decisions
            }
    
    def get_decisions(self, limit: int = None, sort_desc: bool = True) -> List[Decision]:
        """Return a list of decisions, sorted by ID."""
        sorted_decisions = sorted(self.decisions, key=lambda x: x.id, reverse=sort_desc)
        if limit:
            return sorted_decisions[:limit]
        return sorted_decisions

    def get_decision(self, decision_id: int) -> Optional[Decision]:
        """Get a specific decision by ID."""
        for decision in self.decisions:
            if decision.id == decision_id:
                return decision
        return None

    def delete_decision(self, decision_id: int) -> bool:
        """Delete a decision by ID."""
        for i, decision in enumerate(self.decisions):
            if decision.id == decision_id:
                self.decisions.pop(i)
                self._save_decisions()
                # Also remove from Chroma
                self.collection.delete(ids=[str(decision_id)])
                return True
        return False
        
    def regenerate_all_embeddings(self) -> int:
        """
        Regenerate embeddings for all decisions using ChromaDB.
        This is useful after updating the embedding model.
        
        Returns:
            The number of decisions processed
        """
        # Delete the collection if it exists
        try:
            self.chroma_client.delete_collection("decisions")
        except Exception as e:
            print(f"Note: Could not delete collection: {e}")
            pass  # Collection might not exist
        
        # Recreate the collection
        self.collection = self.chroma_client.create_collection(
            name="decisions",
            embedding_function=self.embedding_function
        )
        
        # Add all decisions to the collection
        for decision in self.decisions:
            decision_text = self._prepare_text_for_embedding(decision)
            metadata = {
                "problem": decision.problem,
                "chosen": decision.chosen,
                "mood": decision.mood or ""
            }
            self.collection.add(
                ids=[str(decision.id)],
                documents=[decision_text],
                metadatas=[metadata]
            )
        
        return len(self.decisions)