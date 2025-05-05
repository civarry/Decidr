import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

@dataclass
class Decision:
    """
    Data model for a decision.
    """
    id: int
    timestamp: str
    problem: str
    options: List[str]
    chosen: str
    reasoning: Optional[str] = None
    mood: Optional[str] = None
    
    @classmethod
    def create(cls, 
              id: int,
              problem: str, 
              options: List[str], 
              chosen: str, 
              reasoning: Optional[str] = None, 
              mood: Optional[str] = None) -> 'Decision':
        """
        Create a new decision.
        
        Args:
            id: The decision ID
            problem: The problem or situation
            options: List of possible options
            chosen: The option that was chosen
            reasoning: The reasoning behind the choice
            mood: The mood or emotional state at the time
            
        Returns:
            A new Decision instance
        """
        return cls(
            id=id,
            timestamp=datetime.datetime.now().isoformat(),
            problem=problem,
            options=options,
            chosen=chosen,
            reasoning=reasoning,
            mood=mood
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the decision to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Decision':
        """Create a Decision from a dictionary."""
        return cls(**data)