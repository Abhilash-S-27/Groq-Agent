"""
AI Research Agent - Flask Web Server with SSE Streaming
Run: python app.py
Open: http://localhost:5000
"""

import json
import queue
import threading
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

from tools import ALL_TOOLS

load_dotenv(override=True)

app = Flask(__name__)


# ── Pydantic response model ─────────────────────────────────────────────────
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# ── Streaming callback handler ──────────────────────────────────────────────
class ProgressHandler(BaseCallbackHandler):
    """Pushes live progress events to a queue for SSE streaming."""

    def __init__(self, q: queue.Queue):
        self.q = q

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.q.put({"type": "tool_start", "tool": tool_name})

    def on_tool_end(self, output, **kwargs):
        self.q.put({"type": "tool_done"})

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.q.put({"type": "thinking"})


# ── LLM, parser, prompt ─────────────────────────────────────────────────────
llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a research assistant. Help users by using the available tools.
Use search_wikipedia for factual info. Use search_web for current events.
Use save_research_to_file only if asked to save/export.
Be concise. Wrap output in this format only:\n{format_instructions}""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=ALL_TOOLS)
agent_executor = AgentExecutor(
    agent=agent,
    tools=ALL_TOOLS,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,         # Reduced from 10 -> faster
    early_stopping_method="generate",
)

# ── Session state ────────────────────────────────────────────────────────────
chat_history: list = []


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def handle_query():
    global chat_history
    data = request.get_json()
    query_text = data.get("query", "").strip()

    if not query_text:
        return jsonify({"error": "Empty query"}), 400

    try:
        raw_response = agent_executor.invoke(
            {"query": query_text, "chat_history": chat_history}
        )
        output = raw_response.get("output", "")

        chat_history.append(HumanMessage(content=query_text))
        chat_history.append(AIMessage(content=output))

        parsed = _try_parse(output)
        if parsed:
            return jsonify(
                {
                    "structured": True,
                    "topic": parsed.topic,
                    "summary": parsed.summary,
                    "sources": parsed.sources,
                    "tools_used": parsed.tools_used,
                }
            )
        return jsonify({"structured": False, "raw": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stream", methods=["POST"])
def handle_stream():
    """SSE endpoint — streams tool progress + final answer."""
    global chat_history
    data = request.get_json()
    query_text = data.get("query", "").strip()

    if not query_text:
        return jsonify({"error": "Empty query"}), 400

    q: queue.Queue = queue.Queue()
    handler = ProgressHandler(q)
    result_holder: dict = {}
    error_holder: dict = {}

    def run_agent():
        try:
            raw_response = agent_executor.invoke(
                {"query": query_text, "chat_history": chat_history},
                config={"callbacks": [handler]},
            )
            result_holder["response"] = raw_response
        except Exception as e:
            error_holder["error"] = str(e)
        finally:
            q.put({"type": "done"})

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    def generate():
        while True:
            try:
                event = q.get(timeout=120)
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Timeout'})}\n\n"
                return

            if event["type"] == "done":
                # Send final result
                if "error" in error_holder:
                    yield f"data: {json.dumps({'type': 'error', 'message': error_holder['error']})}\n\n"
                else:
                    raw = result_holder.get("response", {})
                    output = raw.get("output", "")

                    # Update history
                    chat_history.append(HumanMessage(content=query_text))
                    chat_history.append(AIMessage(content=output))

                    parsed = _try_parse(output)
                    if parsed:
                        payload = {
                            "type": "result",
                            "structured": True,
                            "topic": parsed.topic,
                            "summary": parsed.summary,
                            "sources": parsed.sources,
                            "tools_used": parsed.tools_used,
                        }
                    else:
                        payload = {
                            "type": "result",
                            "structured": False,
                            "raw": output,
                        }
                    yield f"data: {json.dumps(payload)}\n\n"
                return
            else:
                yield f"data: {json.dumps(event)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/clear", methods=["POST"])
def clear_history():
    global chat_history
    chat_history = []
    return jsonify({"ok": True})


# ── Helpers ──────────────────────────────────────────────────────────────────
def _try_parse(output: str) -> ResearchResponse | None:
    try:
        return parser.parse(output)
    except Exception:
        pass
    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(output[start:end])
            return ResearchResponse(**data)
    except Exception:
        pass
    return None


if __name__ == "__main__":
    print("\n[*] AI Research Agent is running!")
    print("    Open: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
