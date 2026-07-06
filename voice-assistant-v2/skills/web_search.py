# DuckDuckGo web search tool.

from skills.base import ToolRegistry


@ToolRegistry.register("web_search", "Search the web using DuckDuckGo", mode="celestial")
def web_search(query: str) -> dict:
    # Return the top 5 search results.
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return {"display": "No results found.", "results": []}
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "body": r.get("body", ""),
                "url": r.get("href", "")
            })
        display_lines = [f"Web results for '{query}':"]
        for i, r in enumerate(formatted[:3], 1):
            display_lines.append(f"\n{i}. {r['title']}\n   {r['body'][:200]}\n   {r['url']}")
        return {"results": formatted, "display": "\n".join(display_lines)}
    except ImportError:
        return {"error": "duckduckgo_search not installed", "display": "Install: pip install duckduckgo-search"}
    except Exception as e:
        return {"error": str(e), "display": f"Search failed: {e}"}
