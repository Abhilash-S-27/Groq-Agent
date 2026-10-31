"""
AI Research Agent - Enhanced Version
Features:
  - Wikipedia search
  - DuckDuckGo web search
  - Save research to file
  - Chat history across queries
  - Structured JSON output
  - Colorful terminal UI
"""

import json
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

from tools import ALL_TOOLS  # Import all tools from tools.py

load_dotenv(override=True)

# ── ANSI Colors ──────────────────────────────────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"


# ── Pydantic response model ─────────────────────────────────────────────────
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# ── LLM & Parser ────────────────────────────────────────────────────────────
llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


# ── Prompt ───────────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a powerful research assistant. Your job is to help users by:
1. Searching Wikipedia for factual/encyclopedic information.
2. Searching the web for current events and recent data.
3. Saving research results to a file when asked.
4. Providing the current date/time when relevant.

Always use the available tools to gather information before answering.
If the user asks to save or export, use the save_research_to_file tool.

Wrap the output in this format and provide no other text:
{format_instructions}""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


# ── Agent ────────────────────────────────────────────────────────────────────
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=ALL_TOOLS)
agent_executor = AgentExecutor(
    agent=agent,
    tools=ALL_TOOLS,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)


# ── Helper functions ─────────────────────────────────────────────────────────
def print_banner():
    """Print a fancy startup banner."""
    banner = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗
║              🔬  AI Research Agent  🔬                  ║
╠══════════════════════════════════════════════════════════╣
║  Tools: Wikipedia │ Web Search │ Save │ DateTime        ║
║  Type 'quit' or 'exit' to stop                          ║
║  Type 'history' to see past queries                     ║
║  Type 'clear' to reset chat history                     ║
╚══════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def print_response(response: ResearchResponse, query_num: int):
    """Pretty-print a structured research response."""
    print(f"\n{GREEN}{BOLD}{'─' * 60}")
    print(f"  📋 Result #{query_num}")
    print(f"{'─' * 60}{RESET}")
    print(f"  {BOLD}Topic:{RESET}   {response.topic}")
    print(f"\n  {BOLD}Summary:{RESET}")
    # Word-wrap the summary for readability
    words = response.summary.split()
    line = "    "
    for word in words:
        if len(line) + len(word) + 1 > 70:
            print(line)
            line = "    " + word
        else:
            line += " " + word if line.strip() else "    " + word
    if line.strip():
        print(line)

    if response.sources:
        print(f"\n  {BOLD}Sources:{RESET}")
        for src in response.sources:
            print(f"    🔗 {CYAN}{src}{RESET}")

    if response.tools_used:
        print(f"\n  {BOLD}Tools Used:{RESET}")
        for tool_name in response.tools_used:
            print(f"    🛠️  {YELLOW}{tool_name}{RESET}")
    print(f"{GREEN}{'─' * 60}{RESET}\n")


def parse_response(raw_response: dict) -> ResearchResponse | None:
    """Try to parse the agent output into a ResearchResponse."""
    output = raw_response.get("output", "")
    # Try direct Pydantic parse
    try:
        return parser.parse(output)
    except Exception:
        pass
    # Try extracting JSON from the output string
    try:
        # Look for JSON block in output
        start = output.find("{")
        end = output.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(output[start:end])
            return ResearchResponse(**data)
    except Exception:
        pass
    return None


def show_history(history: list[dict]):
    """Display the chat history."""
    if not history:
        print(f"\n{DIM}  No queries yet.{RESET}\n")
        return
    print(f"\n{MAGENTA}{BOLD}  📜 Chat History ({len(history)} queries){RESET}")
    print(f"{MAGENTA}  {'─' * 40}{RESET}")
    for i, entry in enumerate(history, 1):
        print(f"  {BOLD}{i}.{RESET} {entry['query']}")
        if entry.get("topic"):
            print(f"     → {DIM}{entry['topic']}{RESET}")
    print()


# ── Main loop ────────────────────────────────────────────────────────────────
def main():
    print_banner()

    chat_history = []       # LangChain message history
    query_log = []          # Human-readable log
    query_num = 0

    while True:
        try:
            user_input = input(f"{CYAN}{BOLD}🔍 Enter your query:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}Goodbye! 👋{RESET}")
            break

        if not user_input:
            continue

        # ── Special commands ─────────────────────────────────────────
        cmd = user_input.lower()
        if cmd in ("quit", "exit", "q"):
            print(f"\n{YELLOW}Goodbye! 👋{RESET}")
            break
        if cmd == "history":
            show_history(query_log)
            continue
        if cmd == "clear":
            chat_history.clear()
            query_log.clear()
            query_num = 0
            print(f"{GREEN}  ✓ Chat history cleared.{RESET}\n")
            continue

        query_num += 1
        print(f"\n{DIM}  ⏳ Researching...{RESET}\n")

        try:
            raw_response = agent_executor.invoke(
                {
                    "query": user_input,
                    "chat_history": chat_history,
                }
            )

            # Try structured parse
            parsed = parse_response(raw_response)
            if parsed:
                print_response(parsed, query_num)
                query_log.append({"query": user_input, "topic": parsed.topic})
            else:
                # Fallback: just print the raw output nicely
                output = raw_response.get("output", str(raw_response))
                print(f"\n{GREEN}{BOLD}  Result #{query_num}:{RESET}")
                print(f"  {output}\n")
                query_log.append({"query": user_input, "topic": None})

            # Update chat history for context in future queries
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(
                AIMessage(content=raw_response.get("output", ""))
            )

        except KeyboardInterrupt:
            print(f"\n{YELLOW}  Query cancelled.{RESET}\n")
            continue
        except Exception as e:
            print(f"\n{RED}  ❌ Error: {e}{RESET}\n")
            continue


if __name__ == "__main__":
    main()
