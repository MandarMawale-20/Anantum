# Safe math evaluator using AST checks.

import re
import math

from skills.base import ToolRegistry


@ToolRegistry.register("calculate", "Evaluate a math expression safely", mode="both")
def calculate(expression: str = "", raw_text: str = "") -> dict:
    # Normalize natural language operators.
    expr = expression or raw_text
    expr = re.sub(r"\bplus\b", "+", expr)
    expr = re.sub(r"\bminus\b", "-", expr)
    expr = re.sub(r"\btimes\b|multiplied by", "*", expr)
    expr = re.sub(r"\bdivided by\b", "/", expr)
    expr = re.sub(r"\bsquared\b", "**2", expr)
    expr = re.sub(r"\bcubed\b", "**3", expr)
    expr = re.sub(r"\bsquare root of\b", "sqrt", expr)

    # Handle percent-of patterns directly.
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", expr)
    if pct_match:
        pct, total = float(pct_match.group(1)), float(pct_match.group(2))
        result = pct / 100 * total
        return {"result": result, "display": f"{pct}% of {total} = {result}"}

    sqrt_match = re.search(r"sqrt\s*\(?(\d+(?:\.\d+)?)\)?", expr)
    if sqrt_match:
        n = float(sqrt_match.group(1))
        result = math.sqrt(n)
        return {"result": result, "display": f"\u221a{n} = {result}"}

    # Keep only basic arithmetic characters.
    safe_expr = re.sub(r"[^\d\s\+\-\*\/\.\(\)\^%]", "", expr)
    safe_expr = safe_expr.replace("^", "**")

    try:
        import ast
        tree = ast.parse(safe_expr, mode='eval')

        # Allow only arithmetic AST nodes.
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
                   ast.FloorDiv, ast.USub, ast.UAdd)
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                return {"error": "Unsafe expression", "display": "I can only compute basic math expressions."}

        result = eval(compile(tree, '<string>', 'eval'))
        return {"result": result, "expression": safe_expr, "display": f"{safe_expr} = {result}"}
    except Exception as e:
        return {"error": str(e), "display": f"Couldn't compute that. Try: '2 + 2' or '15% of 200'"}
