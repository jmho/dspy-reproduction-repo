import dspy
import mlflow
from mlflow.dspy import log_model
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("deploy_dspy_program")

api_key = os.getenv("OPENAI_API_KEY")

lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
dspy.settings.configure(lm=lm)

dspy_program = dspy.ChainOfThought("question -> answer")

with mlflow.start_run():
    log_model(
        dspy_program,
        "dspy_program",
        input_example={"messages": [{"role": "user", "content": "What is LLM agent?"}]},
        task="llm/v1/chat",
    )
