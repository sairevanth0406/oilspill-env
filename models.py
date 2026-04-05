from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MyAction(Action):
    task_type: str = Field(..., description="easy | medium | hard")
    prediction: str = Field(..., description="Agent prediction or response")


class MyObservation(Observation):
    input_data: str = Field(..., description="Image description / data")
    ground_truth: str = Field(..., description="Correct answer")

    # ✅ ADD THESE
    reward: float = Field(default=0.0, description="Reward value")
    done: bool = Field(default=False, description="Episode completion flag")