import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.llm_client import LLMClient
from models.decision import Decision


class DecisionMirror:
    """
    A system that learns from your decisions and can simulate how you would respond
    to similar situations in the future.
    """
    
    def __init__(self, api_key: str, model: str, data_dir: str = "decision_data"):
        """
        Initialize the Decision Mirror system.
        
        Args:
            api_key: Your Groq API key
            model: The LLM model to use for prediction
            data_dir: Directory to store decision data
        """
        # Initialize the LLM client
        self.llm_client = LLMClient(api_key=api_key, model=model)
        
        # Create data directory if it doesn't exist
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Path to the decisions database
        self.decisions_path = self.data_dir / "decisions.json"
        
        # Load existing decisions or create empty database
        self.decisions = self._load_decisions()
        
        # Path to embeddings cache
        self.embeddings_path = self.data_dir / "embeddings.json"
        self.embeddings = self._load_embeddings()
    
    def _load_decisions(self) -> List[Decision]:
        """Load existing decisions from file or create empty list."""
        if self.decisions_path.exists():
            with open(self.decisions_path, 'r') as f:
                decision_dicts = json.load(f)
                return [Decision.from_dict(d) for d in decision_dicts]
        return []
    
    def _load_embeddings(self) -> Dict[str, List[float]]:
        """Load existing embeddings from file or create empty dict."""
        if self.embeddings_path.exists():
            with open(self.embeddings_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_decisions(self) -> None:
        """Save decisions to file."""
        decision_dicts = [d.to_dict() for d in self.decisions]
        with open(self.decisions_path, 'w') as f:
            json.dump(decision_dicts, f, indent=2)
    
    def _save_embeddings(self) -> None:
        """Save embeddings to file."""
        with open(self.embeddings_path, 'w') as f:
            json.dump(self.embeddings, f, indent=2)
    
    def add_decision(self, 
                    problem: str, 
                    options: List[str], 
                    chosen: str, 
                    reasoning: Optional[str] = None, 
                    mood: Optional[str] = None) -> Decision:
        """
        Add a new decision to the database.
        
        Args:
            problem: The problem or situation
            options: List of possible options
            chosen: The option you chose
            reasoning: Your reasoning for the decision
            mood: Your mood or emotional state at the time
            
        Returns:
            The newly added decision record
        """
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
        
        # Create and store embedding for this decision
        decision_text = f"Problem: {problem}\nOptions: {', '.join(options)}\nChosen: {chosen}"
        if reasoning:
            decision_text += f"\nReasoning: {reasoning}"
        if mood:
            decision_text += f"\nMood: {mood}"
            
        decision_id = str(decision.id)
        self.embeddings[decision_id] = self.llm_client.get_embedding(decision_text)
        self._save_embeddings()
        
        return decision
    
    def find_similar_decisions(self, problem: str, top_k: int = 3) -> List[Decision]:
        """
        Find decisions similar to the given problem.
        
        Args:
            problem: The problem to find similar decisions for
            top_k: Number of similar decisions to return
            
        Returns:
            List of similar decisions
        """
        if not self.decisions:
            return []
        
        # Get embedding for the query
        query_embedding = self.llm_client.get_embedding(problem)
        
        # Calculate similarities
        similarities = []
        for decision in self.decisions:
            decision_id = str(decision.id)
            if decision_id in self.embeddings:
                embedding = self.embeddings[decision_id]
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((decision, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k similar decisions
        return [decision for decision, _ in similarities[:top_k]]
    
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
        except:
            # Fallback if parsing fails
            return {
                "chosen": None,
                "reasoning": "Failed to parse prediction from model.",
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
                # Also remove from embeddings
                if str(decision_id) in self.embeddings:
                    del self.embeddings[str(decision_id)]
                    self._save_embeddings()
                return True
        return False