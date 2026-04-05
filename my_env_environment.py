import random
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import MyAction, MyObservation


class MyEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.data = [
            {
                "input": "Satellite image with oil spill near coast",
                "label": "yes",
                "region": "coast",
                "severity": "high"
            },
            {
                "input": "Clean ocean satellite image",
                "label": "no",
                "region": "none",
                "severity": "none"
            }
        ]

        self.current = None

    def reset(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current = random.choice(self.data)

        return MyObservation(
            input_data=self.current["input"],
            ground_truth=self.current["label"],
            reward=0.0,
            done=False
        )

    def step(self, action: MyAction):
        self._state.step_count += 1

        if self.current is None:
            self.current = random.choice(self.data)

        task = action.task_type.lower()
        pred = (action.prediction or "").lower()

        reward = 0.0

        # 🟢 EASY
        if task == "easy":
            if pred == self.current["label"]:
                reward = 1.0
            elif pred in ["yes", "no"]:
                reward = 0.5

        # 🟡 MEDIUM
        elif task == "medium":
            if pred == self.current["region"]:
                reward = 1.0
            elif pred in ["coast", "none"]:
                reward = 0.5

        # 🔴 HARD
        elif task == "hard":
            score = 0.0

            if self.current["label"] in pred:
                score += 0.4
            if self.current["region"] in pred:
                score += 0.3
            if self.current["severity"] in pred:
                score += 0.3

            reward = score

        done = True

        return MyObservation(
            input_data=self.current["input"],
            ground_truth=self.current["label"],
            reward=reward,
            done=done
        )

    @property
    def state(self):
        return self._state