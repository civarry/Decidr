import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.llm_client import LLMClient
from models.decision import Decision

from chromadb import Client, Settings


class DecisionMirror:
    def __init__(self, api_key: str, model: str, data_dir: str = "decision_data"):
        self.llm_client = LLMClient(api_key=api_key, model=model)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.decisions_path = self.data_dir / "decisions.json"
        self.decisions = self._load_decisions()
        self.chroma_dir = self.data_dir / "chroma_db"
        self.chroma_dir.mkdir(exist_ok=True)
        self.chroma_client = Client(Settings(
            persist_directory=str(self.chroma_dir),
            anonymized_telemetry=False,
            is_persistent=True
        ))
        self.embedding_function = self._initialize_embedding_function()
        self.collection = self._get_or_create_collection()
        self._sync_decisions_to_chroma()

    def _initialize_embedding_function(self):
        from chromadb.utils import embedding_functions
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    def _get_or_create_collection(self):
        collection_names = self.chroma_client.list_collections()
        collection_exists = any(c.name == "decisions" for c in collection_names)
        if collection_exists:
            return self.chroma_client.get_collection(
                name="decisions", 
                embedding_function=self.embedding_function
            )
        else:
            return self.chroma_client.create_collection(
                name="decisions",
                embedding_function=self.embedding_function
            )

    def _load_decisions(self) -> List[Decision]:
        if self.decisions_path.exists():
            with open(self.decisions_path, 'r') as f:
                decision_dicts = json.load(f)
                return [Decision.from_dict(d) for d in decision_dicts]
        return []

    def _save_decisions(self) -> None:
        decision_dicts = [d.to_dict() for d in self.decisions]
        with open(self.decisions_path, 'w') as f:
            json.dump(decision_dicts, f, indent=2)

    def _sync_decisions_to_chroma(self):
        if self.collection.count() == 0:
            existing_ids = []
        else:
            existing_ids = [str(id) for id in self.collection.get()["ids"]]
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

    def add_decision(self, problem: str, options: List[str], chosen: str, 
                     reasoning: Optional[str] = None, mood: Optional[str] = None) -> Decision:
        new_id = len(self.decisions) + 1
        decision = Decision.create(
            id=new_id,
            problem=problem,
            options=options,
            chosen=chosen,
            reasoning=reasoning,
            mood=mood
        )
        self.decisions.append(decision)
        self._save_decisions()
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
        self.regenerate_all_embeddings()
        return decision

    def _prepare_text_for_embedding(self, decision: Decision) -> str:
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
        if not self.decisions:
            return []
        results = self.collection.query(
            query_texts=[problem],
            n_results=top_k
        )
        similar_decisions = []
        for id in results["ids"][0]:
            decision_id = int(id)
            decision = self.get_decision(decision_id)
            if decision:
                similar_decisions.append(decision)
        return similar_decisions

    def predict_decision(self, problem: str, options: List[str]) -> Dict[str, Any]:
        similar_decisions = self.find_similar_decisions(problem)
        if not similar_decisions:
            return {
                "chosen": None,
                "reasoning": "Not enough past decisions to make a prediction.",
                "confidence": 0.0,
                "similar_decisions": []
            }
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

        Format your response as a JSON object with keys: "chosen", "reasoning", and "confidence".
        """
        response = self.llm_client.call_llm(prompt)
        print(f"This is the prompt:{prompt}")
        print(f"This is the response:{response}")
        try:
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                prediction = json.loads(json_match.group(1))
            else:
                prediction = json.loads(response)
            if not all(key in prediction for key in ["chosen", "reasoning", "confidence"]):
                raise ValueError("Missing required keys in prediction")
            prediction["similar_decisions"] = similar_decisions
            return prediction
        except Exception as e:
            return {
                "chosen": None,
                "reasoning": f"Failed to parse prediction from model: {str(e)}",
                "confidence": 0.0,
                "raw_response": response,
                "similar_decisions": similar_decisions
            }

    def get_decisions(self, limit: int = None, sort_desc: bool = True) -> List[Decision]:
        sorted_decisions = sorted(self.decisions, key=lambda x: x.id, reverse=sort_desc)
        if limit:
            return sorted_decisions[:limit]
        return sorted_decisions

    def get_decision(self, decision_id: int) -> Optional[Decision]:
        for decision in self.decisions:
            if decision.id == decision_id:
                return decision
        return None

    def delete_decision(self, decision_id: int) -> bool:
        for i, decision in enumerate(self.decisions):
            if decision.id == decision_id:
                self.decisions.pop(i)
                self._save_decisions()
                self.collection.delete(ids=[str(decision_id)])
                self.regenerate_all_embeddings()
                return True
        return False

    def regenerate_all_embeddings(self) -> int:
        self.collection.delete(ids=[str(d.id) for d in self.decisions])
        for decision in self.decisions:
            text = self._prepare_text_for_embedding(decision)
            metadata = {
                "problem": decision.problem,
                "chosen": decision.chosen,
                "mood": decision.mood or ""
            }
            self.collection.add(
                ids=[str(decision.id)],
                documents=[text],
                metadatas=[metadata]
            )
        return len(self.decisions)
