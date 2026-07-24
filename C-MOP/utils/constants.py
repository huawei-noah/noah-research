import os
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL = "./embedding_model/all-MiniLM-L6-v2"

TASK_MODEL_URL = os.getenv("TASK_MODEL_URL")
TASK_MODEL_NAME = os.getenv("TASK_MODEL_NAME")
TASK_MODEL_APIKEY = os.getenv("TASK_MODEL_APIKEY")

OPTIMIZER_MODEL_URL = os.getenv("OPTIMIZER_MODEL_URL")
OPTIMIZER_MODEL_NAME = os.getenv("OPTIMIZER_MODEL_NAME")
OPTIMIZER_MODEL_APIKEY = os.getenv("OPTIMIZER_MODEL_APIKEY")