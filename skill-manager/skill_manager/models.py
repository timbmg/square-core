from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator
from square_skill_api.models.prediction import Prediction as SkillPrediction

from skill_manager.mongo.mongo_model import MongoModel
from skill_manager.mongo.py_object_id import PyObjectId


class SkillType(str, Enum):
    """Enum for different skill types."""

    abstractive = "abstractive"
    span_extraction = "span-extraction"
    multiple_choice = "multiple-choice"
    categorical = "categorical"


class SkillSettings(BaseModel):
    """Input Settings for the Skill."""

    requires_context: bool = Field(
        False, description="If `True`, the skill requires an additional context input."
    )
    requires_multiple_choices: int = Field(
        0,
        ge=0,
        description="Defines the minmal number of answer choices for the skill.",
    )


class SkillInputExample(BaseModel):
    """Holds an examplary input for a skill."""

    query: str = Field(..., description="The input to a skill, for example a question.")
    context: Optional[str] = Field(
        description="Additional input to the skill, for example background knowledge."
    )
    answers: Optional[List[str]] = Field(
        None, description="List of answer candidates for multiple-choice skills."
    )


class Skill(MongoModel):
    """Holds all skill information."""

    id: Optional[PyObjectId] = Field(
        None, description="Identifier generated by mongoDB"
    )
    name: str = Field(..., description="The name of the skill.")
    url: str = Field(..., description="The url where the skill is running.")
    skill_type: SkillType
    skill_settings: SkillSettings
    user_id: str = Field(..., description="Username of the skill author.")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp of skill creation."
    )
    skill_input_examples: Optional[List[SkillInputExample]]
    description: Optional[str] = Field(
        None,
        description="A description of the skill, for example describing its pipeline.",
    )
    default_skill_args: Optional[Dict] = Field(
        None,
        description="A dictionary holding key-value pairs that should always be sent to the skill as input. This allows to use the same skill implementataion in different ways.",
    )
    published: bool = Field(
        False,
        description="If `True`, the skill will be publicly availble, ready to be used by anyone. If `False`, only the skill author has access to it.",
    )
    client_id: Optional[str] = Field(
        None, description="The clientId of the skill stored in Keycloak."
    )
    client_secret: Optional[str] = Field(
        None, description="The cleint secret of the skill stored in Keycloak."
    )

    @validator("url")
    def validate_url(cls, url: str) -> str:
        """Checks if the provided url is valid. Has to start with `http`. Trailing slashed are removed.

        Args:
            url (str): The url where the skill is running.

        Raises:
            ValueError: url was found not to be valid.

        Returns:
            str: url with trimmed trailing slash.
        """
        if not url.startswith("http"):
            raise ValueError(url)
        if url.endswith("/"):
            url = url[:-1]
        return url

    class Config:
        schema_extra = {
            "example": {
                "name": "HAL9000",
                "url": "http://h.al:9000",
                "description": "Heuristically programmed Algorithmic computer",
                "skill_type": "abstractive",
                "skill_settings": {
                    "requires_context": False,
                    "requires_multiple_choices": 0,
                },
                "default_skill_args": {},
                "user_id": "Dave",
                "published": False,
                "skill_input_examples": [
                    {
                        "query": "What arms did Moonwatchers band carry?",
                        "context": "At the water's edge, Moonwatcher and his band stop. They carry their bone clubs and bone knives. Led by One-ear, the Others half-heartly resume the battle-chant. But they are suddenly confrunted with a vision that cuts the sound from their throats, and strikes terror into their hearts.",
                    }
                ],
            }
        }


class Prediction(MongoModel):
    """Holds all prediction information, i.e. the output of a skill when queried."""

    id: Optional[PyObjectId] = Field(
        None, description="Identifier generated by mongoDB."
    )
    skill_id: PyObjectId = Field(
        ..., description="Identifier of the skill that generated the prediction."
    )
    skill_name: str = Field(
        ..., description="Name of the skill that generated the prediction."
    )
    query: str = Field(
        ..., description="The input query that resulted in the prediction."
    )
    query_time: datetime = Field(
        default_factory=datetime.now,
        description="Time when the prediction was generated.",
    )
    user_id: str = Field(..., description="User that issued the query.")
    predictions: List[SkillPrediction]
