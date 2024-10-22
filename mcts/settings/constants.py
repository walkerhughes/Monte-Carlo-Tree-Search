import os

RANDOM_SEED = 17
MAX_CHILDREN = 3

LLAMA_MODEL = "llama3-8b-8192"
OPENAI_MODEL = "gpt-4o-mini"
GROQ_ENDPOINT = "https://api.groq.com/openai"
OPENAI_ENDPOINT = "https://api.openai.com"
OPENAI_API_BASE = OPENAI_ENDPOINT + "/v1"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

SEED_ANSWERS = [
    "I don't know the answer",
    "I'm not sure",
    "I can't say on that one"
]