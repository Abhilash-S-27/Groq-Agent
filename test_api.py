import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_groq import ChatGroq
import time

models = ["openai/gpt-oss-20b", "allam-2-7b", "qwen/qwen3.8-27b"]
for m in models:
    try:
        llm = ChatGroq(model_name=m, temperature=0)
        start = time.time()
        resp = llm.invoke("What is Python? Reply in 1 sentence.")
        elapsed = time.time() - start
        print(f"{m:30s} -> {elapsed:.2f}s")
    except Exception as e:
        print(f"{m:30s} -> FAILED: {str(e)[:80]}")
