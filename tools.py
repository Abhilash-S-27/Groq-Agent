"""
Custom tools for the AI Research Agent.
Includes: Wikipedia search, Web search, Save-to-file, and a DateTime tool.
Optimized for speed — tool outputs are trimmed to reduce LLM processing time.
"""

import os
import json
import datetime
from langchain_core.tools import tool


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a given query and return a summary.
    Use this when you need factual, encyclopedic information about a topic."""
    try:
        import wikipedia

        wikipedia.set_lang("en")
        results = wikipedia.search(query, results=2)
        if not results:
            return f"No Wikipedia results found for: {query}"

        # Only get the FIRST result, limit to 500 chars for speed
        try:
            page = wikipedia.page(results[0], auto_suggest=False)
            return (
                f"Title: {page.title}\n"
                f"URL: {page.url}\n"
                f"Summary: {page.summary[:500]}"
            )
        except (wikipedia.DisambiguationError, wikipedia.PageError):
            return f"Could not retrieve Wikipedia page for: {query}"

    except Exception as e:
        return f"Wikipedia search error: {str(e)}"


@tool
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo for a given query.
    Use this for current events, recent data, or topics not well-covered by Wikipedia."""
    try:
        from langchain_community.tools import DuckDuckGoSearchResults

        search = DuckDuckGoSearchResults(max_results=3)
        results = search.invoke(query)
        # Trim to 800 chars max
        if results and len(results) > 800:
            results = results[:800]
        return results if results else f"No web results found for: {query}"
    except ImportError:
        return "DuckDuckGo search is not available."
    except Exception as e:
        return f"Web search error: {str(e)}"


@tool
def save_research_to_file(filename: str, content: str) -> str:
    """Save research content to a text file in the 'research_output' folder.
    Use this when the user asks to save, export, or store their research."""
    try:
        output_dir = os.path.join(os.path.dirname(__file__), "research_output")
        os.makedirs(output_dir, exist_ok=True)

        safe_name = "".join(
            c if c.isalnum() or c in ("_", "-", " ") else "_" for c in filename
        )
        if not safe_name.endswith(".txt"):
            safe_name += ".txt"

        filepath = os.path.join(output_dir, safe_name)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Research Report\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(content)

        return f"Saved to: {filepath}"
    except Exception as e:
        return f"Error saving file: {str(e)}"


@tool
def get_current_datetime() -> str:
    """Get the current date and time. Useful for time-sensitive research queries."""
    now = datetime.datetime.now()
    return now.strftime("Date: %A, %B %d, %Y | Time: %I:%M %p")


ALL_TOOLS = [search_wikipedia, search_web, save_research_to_file, get_current_datetime]
