import os
import json
import ast
import operator
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

# Load environment variables from .env
load_dotenv()

# Initialize API clients
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")


# ==========================================
# Tool 1: Web Search (Tavily)
# ==========================================
def web_search(query: str, **kwargs) -> str:
    """
    Searches the web using Tavily and returns relevant search results.
    Snippets are trimmed to stay well within model context and rate limits.
    """
    try:
        response = tavily.search(query=query, max_results=3)
        results = response.get("results", [])
        if not results:
            return "No relevant search results found."

        formatted_results = []
        for idx, item in enumerate(results, start=1):
            title = item.get("title", "No Title")
            content = item.get("content", "").strip()
            # Trim snippet length to save tokens
            if len(content) > 300:
                content = content[:300] + "..."
            url = item.get("url", "")
            formatted_results.append(
                f"[{idx}] {title}\nSnippet: {content}\nSource: {url}"
            )

        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error executing web search: {str(e)}"


# ==========================================
# Tool 2: Maths Calculation
# ==========================================
def calculate(expression: str, **kwargs) -> str:
    """
    Evaluates a basic mathematical expression (e.g., '2+2', '15 * (4 + 2)', '100 / 4').
    Safely parses and computes arithmetic expressions using AST.
    """
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](left, right)
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](operand)
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        else:
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    try:
        # Pre-clean expression (replace caret with power operator if used)
        cleaned = expression.strip().replace("^", "**")
        parsed = ast.parse(cleaned, mode="eval")
        result = _eval(parsed.body)
        return str(result)
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# Map tool names to python functions
TOOL_MAP = {
    "web_search": web_search,
    "calculate": calculate,
}

# ==========================================
# Standard Tool Definitions for LLM
# ==========================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date information, facts, news, and external knowledge using Tavily.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate basic mathematical expressions (e.g., '2+2', '12 * 8', '100 / 4', '2 ** 8').",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, like '2+2' or '(14 + 6) * 3'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# ==========================================
# Agent Function
# ==========================================
def run_agent(user_query: str, max_iterations: int = 3) -> str:
    """
    Takes an input string query, lets the LLM decide which tool to call in an agent loop,
    executes the tools, and returns the final answer.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an efficient and helpful AI assistant with access to 'calculate' and 'web_search' tools. "
                "Use 'calculate' for math expressions and 'web_search' for real-time questions. "
                "Be token-efficient: limit yourself to 1 or 2 targeted searches at most. "
                "Once you retrieve relevant search results, synthesize them immediately into a clear, direct answer."
            ),
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    for step in range(1, max_iterations + 1):
        print(f"\n--- [Iteration {step}] ---")

        # Always provide tools and tool_choice="auto" so the model can call tools if needed
        response = groq.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # If the model did not call any tools, it has reached its final answer
        if not tool_calls:
            print("-> [Agent Decision] Final response generated (no more tool calls needed).")
            return response_message.content

        # Append the assistant's response (with tool calls) to message history
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            try:
                function_args = json.loads(tool_call.function.arguments)
            except Exception:
                function_args = {}

            print(f"-> [Agent Decision] Tool Called: {function_name}")
            print(f"-> [Arguments]: {function_args}")

            if function_name in TOOL_MAP:
                tool_func = TOOL_MAP[function_name]
                try:
                    tool_result = tool_func(**function_args)
                except Exception as e:
                    tool_result = f"Error executing tool '{function_name}': {str(e)}"
            else:
                tool_result = f"Error: Function '{function_name}' is not recognized."

            preview = (
                str(tool_result)[:120] + "..."
                if len(str(tool_result)) > 120
                else str(tool_result)
            )
            print(f"-> [Tool Result]: {preview}")

            # Append the tool result back to the conversation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": str(tool_result),
                }
            )

    # If maximum iterations reached, produce final answer from accumulated context
    print(f"\n--- [Max iterations reached ({max_iterations})] Synthesizing final response ---")
    final_response = groq.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
    )
    return final_response.choices[0].message.content


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        try:
            query = input("Enter your query (or press Enter for default): ").strip()
            if not query:
                query = "What is 45 * 12 + 250 and who won the latest ICC Cricket World Cup?"
        except EOFError:
            query = (
                "What is 45 * 12 + 250 and who won the latest ICC Cricket World Cup?"
            )

    print(f"\nUser Query: {query}\n" + "-" * 50)
    final_answer = run_agent(query)
    print("\n" + "=" * 50 + "\nFinal Answer:\n" + "=" * 50)
    print(final_answer)
