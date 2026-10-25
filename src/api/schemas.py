"""Validated request schemas."""
from typing import Optional

from pydantic import BaseModel, Field


class AnomalyRequest(BaseModel):
    study_id: str = Field(..., min_length=8, max_length=32, pattern=r"^[0-9a-f]+$")


class AskRequest(BaseModel):
    study_id: str = Field(..., min_length=8, max_length=32, pattern=r"^[0-9a-f]+$")
    question: str = Field(..., min_length=3, max_length=500)
    earlier_study_id: Optional[str] = Field(None, min_length=8, max_length=32, pattern=r"^[0-9a-f]+$")


class CompareRequest(BaseModel):
    earlier_study_id: str = Field(..., min_length=8, max_length=32, pattern=r"^[0-9a-f]+$")
    later_study_id: str = Field(..., min_length=8, max_length=32, pattern=r"^[0-9a-f]+$")
    question: str = Field("Has the suspected lesion increased or decreased in size since the previous scan?",
                          min_length=3, max_length=500)
