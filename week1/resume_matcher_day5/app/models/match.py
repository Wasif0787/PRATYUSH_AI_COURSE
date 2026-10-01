from pydantic import BaseModel


class MatchResult(BaseModel):
    score: float
    details: dict
