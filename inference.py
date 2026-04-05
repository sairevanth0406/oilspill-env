import asyncio
import os
from typing import List

from openai import OpenAI

from server.my_env_environment import MyEnvironment, MyAction


API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

MAX_STEPS = 5


def log_start():
    print(f"[START] task=oilspill env=my_env model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


async def main():
    env = MyEnvironment()

    rewards = []
    steps_taken = 0

    log_start()

    obs = env.reset()

    for step in range(1, MAX_STEPS + 1):
        # simple agent logic
        action_text = "yes"

        action = MyAction(task_type="easy", prediction=action_text)

        obs = env.step(action)

        reward = obs.reward
        done = obs.done

        rewards.append(reward)
        steps_taken = step

        log_step(step, action_text, reward, done)

        if done:
            break

    score = sum(rewards) / len(rewards)
    success = score > 0.5

    log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())