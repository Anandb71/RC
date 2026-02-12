"""
RC-Oracle Backend — Hybrid Symbolic-Neural Solver
FastAPI server with Data Factory, Symbolic Heuristics, OpenAI Agentic Loop, and C++ Translation.
"""

import os
import re
import sys
import json
import math
import base64
import subprocess
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import random

# Groq uses the same OpenAI SDK — just different base_url
from openai import OpenAI

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(title="RC-Oracle", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Groq Setup (FREE, no billing required) ─────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
MODEL_NAME = "llama-3.3-70b-versatile"  # Free tier: 30 RPM, 1000 RPD, 128K context
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

if GROQ_API_KEY:
    oai_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    print(f"✅ Groq configured — model: {MODEL_NAME}")
else:
    oai_client = None
    print("⚠ GROQ_API_KEY not set — AI features disabled.")

# ─── Pydantic Models ────────────────────────────────────────────────────────

class ProbeRequest(BaseModel):
    executable_path: str
    custom_inputs: Optional[list[str]] = None

class IOPair(BaseModel):
    input: str
    output: str

class SolveRequest(BaseModel):
    pairs: list[IOPair]

class TranslateRequest(BaseModel):
    python_code: str

# ─── Synthetic Input Generator ──────────────────────────────────────────────
# Competitive programming problems typically expect:
# - Single number
# - First line = N, second line = N space-separated values
# - Multi-line structured input

def generate_synthetic_inputs() -> list[str]:
    """Generate 100+ diverse synthetic inputs for competitive programming problems."""
    inputs = []

    # === Single integers (small, medium, large, edge cases) ===
    single_ints = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 15, 16, 17, 20, 25, 30, 31, 32, 42,
        50, 63, 64, 100, 127, 128, 200, 255, 256, 500, 999, 1000,
        -1, -2, -5, -10, -50, -100,
    ]
    for n in single_ints:
        inputs.append(str(n))

    # === Multi-line: N followed by N space-separated random ints ===
    for n in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        nums = [str(random.randint(-100, 100)) for _ in range(n)]
        inputs.append(f"{n}\n{' '.join(nums)}")

    # === Multi-line: N followed by N positive-only ints ===
    for n in [1, 3, 5, 8, 10]:
        nums = [str(random.randint(1, 1000)) for _ in range(n)]
        inputs.append(f"{n}\n{' '.join(nums)}")

    # === Space-separated numbers on single line (no N prefix) ===
    for count in [2, 3, 4, 5]:
        nums = [str(random.randint(1, 100)) for _ in range(count)]
        inputs.append(' '.join(nums))

    # === Two integers on separate lines ===
    for a, b in [(1, 2), (3, 5), (10, 20), (7, 3), (100, 50)]:
        inputs.append(f"{a}\n{b}")

    # === Single-line strings ===
    inputs.extend(["hello", "world", "abc", "abcdef", "a", "ab", "xyz"])

    # Shuffle to avoid ordering bias, but keep first 10 as simple ints
    simple = inputs[:10]
    rest = inputs[10:]
    random.shuffle(rest)
    return simple + rest


def select_best_pairs(pairs: list[dict], max_pairs: int = 15) -> list[dict]:
    """Select the best subset of I/O pairs for AI analysis.
    Prioritizes diversity: different input lengths, output values, input types."""
    if len(pairs) <= max_pairs:
        return pairs

    selected = []
    seen_outputs = set()
    seen_input_types = set()  # single-line vs multi-line

    # First pass: grab diverse outputs
    for p in pairs:
        out_key = p["output"]
        inp_type = "multi" if "\n" in p["input"] else "single"
        type_key = f"{inp_type}_{len(p['input'])}"

        if out_key not in seen_outputs or type_key not in seen_input_types:
            selected.append(p)
            seen_outputs.add(out_key)
            seen_input_types.add(type_key)
            if len(selected) >= max_pairs:
                break

    # Fill remaining slots with pairs we haven't selected
    if len(selected) < max_pairs:
        remaining = [p for p in pairs if p not in selected]
        selected.extend(remaining[:max_pairs - len(selected)])

    return selected


# ─── Data Factory — Proactive Probing ────────────────────────────────────────

def run_executable(exe_path: str, input_value: str, timeout: float = 10.0) -> Optional[str]:
    """Execute a binary with a single input and return its stdout."""
    try:
        # Ensure we pass the input with a trailing newline
        full_input = input_value if input_value.endswith("\n") else input_value + "\n"
        result = subprocess.run(
            [exe_path],
            input=full_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout.strip()
        if stdout:
            return stdout
        return None
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def probe_executable(exe_path: str, custom_inputs: Optional[list[str]] = None) -> list[dict]:
    """Run the executable with 100+ synthetic inputs and collect I/O pairs."""
    inputs = generate_synthetic_inputs()
    if custom_inputs:
        inputs.extend(custom_inputs)

    pairs = []
    for inp in inputs:
        output = run_executable(exe_path, inp)
        if output is not None and output != "":
            pairs.append({"input": inp, "output": output})
    return pairs

# ─── Symbolic Heuristics Engine ──────────────────────────────────────────────

def try_numeric_pairs(pairs: list[dict]) -> Optional[dict]:
    """Check if all I/O pairs can be explained by a numeric transformation."""
    numeric_pairs = []
    for p in pairs:
        try:
            x = float(p["input"])
            y = float(p["output"])
            numeric_pairs.append((x, y))
        except (ValueError, TypeError):
            continue

    if len(numeric_pairs) < 3:
        return None

    # Identity: y == x
    if all(y == x for x, y in numeric_pairs):
        return {"type": "identity", "logic": "y = x", "code": "def solve(s):\n    return s"}

    # Constant: all outputs the same
    outputs = [y for _, y in numeric_pairs]
    if len(set(outputs)) == 1:
        c = outputs[0]
        c_repr = int(c) if c == int(c) else c
        return {
            "type": "constant",
            "logic": f"y = {c_repr}",
            "code": f"def solve(s):\n    return str({c_repr})",
        }

    # Linear: y = a*x + b
    if len(numeric_pairs) >= 2:
        x1, y1 = numeric_pairs[0]
        x2, y2 = numeric_pairs[1]
        if x2 != x1:
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            if all(abs(a * x + b - y) < 1e-9 for x, y in numeric_pairs):
                a_repr = int(a) if a == int(a) else a
                b_repr = int(b) if b == int(b) else b
                if b_repr == 0:
                    logic = f"y = {a_repr} * x"
                    code = f"def solve(s):\n    return str(int(float(s)) * {a_repr})"
                elif a_repr == 1:
                    logic = f"y = x + {b_repr}"
                    code = f"def solve(s):\n    return str(int(float(s)) + {b_repr})"
                else:
                    sign = "+" if b_repr >= 0 else "-"
                    logic = f"y = {a_repr} * x {sign} {abs(b_repr)}"
                    code = f"def solve(s):\n    return str(int(float(s) * {a_repr} + {b_repr}))"
                return {"type": "linear", "logic": logic, "code": code}

    # Quadratic: y = x^2
    if all(y == x * x for x, y in numeric_pairs):
        return {
            "type": "quadratic",
            "logic": "y = x²",
            "code": "def solve(s):\n    return str(int(float(s)) ** 2)",
        }

    # Power of 2: y = 2^x
    if all(x >= 0 and y == 2**x for x, y in numeric_pairs):
        return {
            "type": "power_of_2",
            "logic": "y = 2^x",
            "code": "def solve(s):\n    return str(2 ** int(float(s)))",
        }

    # Factorial
    def factorial_match(x, y):
        if x < 0 or x != int(x):
            return False
        return y == math.factorial(int(x))

    if all(x >= 0 for x, _ in numeric_pairs) and all(factorial_match(x, y) for x, y in numeric_pairs):
        return {
            "type": "factorial",
            "logic": "y = x!",
            "code": "import math\ndef solve(s):\n    return str(math.factorial(int(s)))",
        }

    # Absolute value
    if all(y == abs(x) for x, y in numeric_pairs):
        return {
            "type": "absolute",
            "logic": "y = |x|",
            "code": "def solve(s):\n    return str(abs(int(float(s))))",
        }

    return None


def try_string_transforms(pairs: list[dict]) -> Optional[dict]:
    """Check for common string transformations."""
    if not pairs:
        return None

    # Reverse
    if all(p["output"] == p["input"][::-1] for p in pairs):
        return {
            "type": "reverse",
            "logic": "reverse the string",
            "code": "def solve(s):\n    return s[::-1]",
        }

    # Uppercase
    if all(p["output"] == p["input"].upper() for p in pairs):
        return {
            "type": "upper",
            "logic": "convert to UPPERCASE",
            "code": "def solve(s):\n    return s.upper()",
        }

    # Lowercase
    if all(p["output"] == p["input"].lower() for p in pairs):
        return {
            "type": "lower",
            "logic": "convert to lowercase",
            "code": "def solve(s):\n    return s.lower()",
        }

    # Title case
    if all(p["output"] == p["input"].title() for p in pairs):
        return {
            "type": "title",
            "logic": "convert to Title Case",
            "code": "def solve(s):\n    return s.title()",
        }

    # Strip whitespace — only match if there's an actual whitespace difference
    if (all(p["output"] == p["input"].strip() for p in pairs) and
            any(p["output"] != p["input"] for p in pairs)):
        return {
            "type": "strip",
            "logic": "strip leading/trailing whitespace",
            "code": "def solve(s):\n    return s.strip()",
        }

    # String length
    if all(p["output"] == str(len(p["input"])) for p in pairs):
        return {
            "type": "length",
            "logic": "y = length of input string",
            "code": "def solve(s):\n    return str(len(s))",
        }

    # Sort characters
    if all(p["output"] == "".join(sorted(p["input"])) for p in pairs):
        return {
            "type": "sort_chars",
            "logic": "sort characters alphabetically",
            "code": 'def solve(s):\n    return "".join(sorted(s))',
        }

    # Word count
    if all(p["output"] == str(len(p["input"].split())) for p in pairs if p["input"].strip()):
        return {
            "type": "word_count",
            "logic": "y = number of words",
            "code": "def solve(s):\n    return str(len(s.split()))",
        }

    return None


def try_encoding_transforms(pairs: list[dict]) -> Optional[dict]:
    """Check for Base64, Hex, Caesar cipher."""
    if not pairs or not all(p["input"] for p in pairs):
        return None

    # Base64 encode
    try:
        if all(p["output"] == base64.b64encode(p["input"].encode()).decode() for p in pairs):
            return {
                "type": "base64_encode",
                "logic": "Base64 encode",
                "code": "import base64\ndef solve(s):\n    return base64.b64encode(s.encode()).decode()",
            }
    except Exception:
        pass

    # Base64 decode
    try:
        if all(p["output"] == base64.b64decode(p["input"].encode()).decode() for p in pairs):
            return {
                "type": "base64_decode",
                "logic": "Base64 decode",
                "code": "import base64\ndef solve(s):\n    return base64.b64decode(s.encode()).decode()",
            }
    except Exception:
        pass

    # Hex encode
    try:
        if all(p["output"] == p["input"].encode().hex() for p in pairs):
            return {
                "type": "hex_encode",
                "logic": "Hex encode",
                "code": "def solve(s):\n    return s.encode().hex()",
            }
    except Exception:
        pass

    # Caesar cipher (try shifts 1-25)
    def caesar(text, shift):
        result = []
        for c in text:
            if c.isalpha():
                base = ord("A") if c.isupper() else ord("a")
                result.append(chr((ord(c) - base + shift) % 26 + base))
            else:
                result.append(c)
        return "".join(result)

    for shift in range(1, 26):
        if all(p["output"] == caesar(p["input"], shift) for p in pairs):
            return {
                "type": "caesar",
                "logic": f"Caesar cipher (shift {shift})",
                "code": (
                    f"def solve(s):\n"
                    f"    result = []\n"
                    f"    for c in s:\n"
                    f"        if c.isalpha():\n"
                    f"            base = ord('A') if c.isupper() else ord('a')\n"
                    f"            result.append(chr((ord(c) - base + {shift}) % 26 + base))\n"
                    f"        else:\n"
                    f"            result.append(c)\n"
                    f"    return ''.join(result)"
                ),
            }

    return None


def symbolic_solve(pairs: list[dict]) -> Optional[dict]:
    """Run all symbolic heuristic checks. Returns match or None."""
    result = try_numeric_pairs(pairs)
    if result:
        return result

    result = try_string_transforms(pairs)
    if result:
        return result

    result = try_encoding_transforms(pairs)
    if result:
        return result

    return None


# ─── Groq Agentic Synthesis — Code-Run-Verify Loop ────────────────────────

SYSTEM_PROMPT = """You are an expert competitive programmer. You are given input-output pairs from a black-box function.
Your task is to analyze the pattern and write a Python function `def solve(s):` that takes the FULL raw input as a single string `s` and returns the expected output as a string.

IMPORTANT — Competitive Programming Format:
- The input `s` is often MULTI-LINE. For example:
  - First line might be N (count), second line might be N space-separated numbers.
  - Or it might be just a single number.
- You should parse the input string accordingly (split lines, split by spaces, convert to int, etc.).
- Always return the result as a string.

Common patterns to consider:
- Arithmetic: sums, products, differences, averages, min/max
- Number theory: primes, GCD, LCM, modular arithmetic
- Bit manipulation: XOR, AND, OR, bit reversal, popcount, binary representation
- String operations: reversal, sorting, counting, encoding
- Array operations: sorting, prefix sums, cumulative operations
- Mathematical formulas: polynomials, sequences, combinatorics
- Think about what happens to the BINARY representation of numbers

Rules:
1. Your function MUST be named `solve` and take exactly one string argument `s`.
2. Your function MUST return a string.
3. Output ONLY the Python function code. No explanations, no markdown fences, no extra text.
4. Parse the input by splitting lines and values as needed.
5. Handle edge cases (empty string, negative numbers, single values, etc.).
6. Think step-by-step about what mathematical or algorithmic pattern maps each input to its output.
7. If one approach fails, try a COMPLETELY DIFFERENT approach — don't just tweak numbers.
"""

def extract_solve_function(response_text: str) -> str:
    """Extract the solve() function from the AI response."""
    # Try to find code between markdown fences first
    fence_match = re.search(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
    if fence_match:
        code = fence_match.group(1).strip()
    else:
        code = response_text.strip()

    # Ensure it contains def solve
    if "def solve" not in code:
        raise ValueError("Response does not contain a solve() function")

    # Extract from first import or def to end
    lines = code.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("import ") or line.strip().startswith("from ") or line.strip().startswith("def solve"):
            start_idx = i
            break
    code = "\n".join(lines[start_idx:])
    return code


def execute_solve(code: str, test_input: str) -> str:
    """Execute the solve() function in a sandbox and return its output."""
    namespace = {}
    exec(code, namespace)
    solve_fn = namespace.get("solve")
    if solve_fn is None:
        raise ValueError("No solve() function defined in the generated code")
    result = solve_fn(test_input)
    return str(result)


def verify_solution(code: str, pairs: list[dict]) -> tuple[bool, Optional[dict]]:
    """Test the solve function against all I/O pairs. Whitespace-tolerant comparison."""
    for p in pairs:
        try:
            got = execute_solve(code, p["input"]).strip()
            expected = p["output"].strip()
            if got != expected:
                return False, {
                    "input": p["input"],
                    "expected": expected,
                    "got": got,
                    "error": None,
                }
        except Exception as e:
            return False, {
                "input": p["input"],
                "expected": p["output"],
                "got": None,
                "error": str(e),
            }
    return True, None


def score_solution(code: str, pairs: list[dict]) -> int:
    """Count how many I/O pairs the code matches. Used to rank best-effort solutions."""
    matched = 0
    for p in pairs:
        try:
            got = execute_solve(code, p["input"]).strip()
            if got == p["output"].strip():
                matched += 1
        except Exception:
            pass
    return matched


async def ai_solve(pairs: list[dict], max_retries: int = 20) -> dict:
    """Agentic loop: ask AI, run code, verify, retry on failure.
    Tracks the best-scoring attempt so even on total failure we return closest solution."""
    if not oai_client:
        raise HTTPException(status_code=503, detail="Groq API key not configured")

    # Select best pairs for AI (limit to 15 for efficiency)
    ai_pairs = select_best_pairs(pairs, max_pairs=15)

    pair_text = "\n".join(f"  Input: {p['input']!r}  →  Output: {p['output']!r}" for p in ai_pairs)
    user_prompt = (
        f"Here are {len(ai_pairs)} input-output pairs from a black-box program:\n{pair_text}\n\n"
        f"Note: inputs may be multi-line (e.g., first line = N, second line = N numbers).\n"
        f"Analyze the pattern carefully and write the solve(s) function."
    )

    # Build conversation history for multi-turn retry
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Track the best attempt across all retries
    best_code = None
    best_score = -1
    best_attempt = 0

    for attempt in range(1, max_retries + 1):
        try:
            # Increase temperature slightly on later attempts to explore different approaches
            temp = 0.2 if attempt <= 10 else 0.5
            response = oai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temp,
                max_tokens=2048,
            )
            response_text = response.choices[0].message.content

            # Add assistant response to history for multi-turn
            messages.append({"role": "assistant", "content": response_text})

            code = extract_solve_function(response_text)

            # Score this attempt
            score = score_solution(code, ai_pairs)
            if score > best_score:
                best_score = score
                best_code = code
                best_attempt = attempt

            success, failure = verify_solution(code, ai_pairs)

            if success:
                return {
                    "status": "solved",
                    "code": code,
                    "attempts": attempt,
                }

            # Build retry prompt with failure info
            if failure["error"]:
                retry_msg = (
                    f"Your solution FAILED on attempt {attempt}.\n"
                    f"Input: {failure['input']!r}\n"
                    f"Expected: {failure['expected']!r}\n"
                    f"Error: {failure['error']}\n\n"
                    f"Fix the solve(s) function. Output ONLY the corrected Python code."
                )
            else:
                retry_msg = (
                    f"Your solution FAILED on attempt {attempt}.\n"
                    f"Input: {failure['input']!r}\n"
                    f"Expected: {failure['expected']!r}\n"
                    f"Got: {failure['got']!r}\n\n"
                    f"Fix the solve(s) function. Output ONLY the corrected Python code."
                )
            messages.append({"role": "user", "content": retry_msg})

            # Reset conversation every 8 attempts to avoid context bloat
            if attempt % 8 == 0 and attempt < max_retries:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt + f"\n\nPrevious {attempt} attempts all failed. Try a COMPLETELY DIFFERENT algorithm or approach."},
                ]

        except Exception as e:
            error_msg = f"Error executing your code: {str(e)}\nPlease fix and output ONLY the corrected Python solve(s) function."
            messages.append({"role": "user", "content": error_msg})

    # Return best-effort solution even if not perfect
    if best_code and best_score > 0:
        return {
            "status": "best_effort",
            "code": best_code,
            "attempts": max_retries,
            "score": f"{best_score}/{len(ai_pairs)}",
        }

    return {
        "status": "failed",
        "code": None,
        "attempts": max_retries,
        "error": "Max retries exceeded",
    }


# ─── C++ Translation ────────────────────────────────────────────────────────

CPP_TRANSLATE_PROMPT = """You are an expert C++ competitive programmer. Translate the following Python `solve()` function into an equivalent C++ program.

CRITICAL Requirements:
1. The C++ code MUST use a `void solve()` function (NOT string, NOT int — VOID).
2. Inside `void solve()`, read input from `cin` and write output to `cout`.
3. Use `#include <bits/stdc++.h>` and `using namespace std;`.
4. The `main()` function should simply call `solve()` and return 0.
5. Do NOT use `string solve(string s)` — that is WRONG.
6. Output ONLY the complete C++ code. No explanations.

Example structure:
```cpp
#include <bits/stdc++.h>
using namespace std;

void solve() {{
    // read from cin, compute, write to cout
}}

int main() {{
    solve();
    return 0;
}}
```

Python code to translate:
```python
{python_code}
```
"""

async def translate_to_cpp(python_code: str) -> str:
    """Use GPT-4o-mini to translate Python solve() to C++."""
    if not oai_client:
        raise HTTPException(status_code=503, detail="Groq API key not configured")

    prompt = CPP_TRANSLATE_PROMPT.format(python_code=python_code)
    response = oai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert C++ programmer. Output ONLY code."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    text = response.choices[0].message.content
    fence_match = re.search(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text.strip()


# ─── API Routes ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@app.get("/")
async def root():
    return {
        "service": "RC-Oracle",
        "version": "1.0.0",
        "status": "online",
        "ai": "connected" if oai_client else "disconnected",
        "model": MODEL_NAME,
    }

@app.get("/factory")
async def factory_page():
    """Serve the Data Factory HTML page."""
    html_path = PROJECT_ROOT / "data-factory.html"
    if not html_path.is_file():
        raise HTTPException(status_code=404, detail="data-factory.html not found")
    return FileResponse(str(html_path), media_type="text/html")


@app.post("/probe")
async def probe(req: ProbeRequest):
    """Data Factory: Run executable with synthetic inputs, return I/O pairs."""
    # Clean up the path — strip quotes and whitespace
    exe_path = req.executable_path.strip().strip('"').strip("'")

    if not os.path.isfile(exe_path):
        raise HTTPException(status_code=404, detail=f"Executable not found: {exe_path}")

    # Generate inputs and count them
    all_inputs = generate_synthetic_inputs()
    if req.custom_inputs:
        all_inputs.extend(req.custom_inputs)
    total_count = len(all_inputs)

    pairs = probe_executable(exe_path, req.custom_inputs)

    if not pairs:
        raise HTTPException(status_code=500, detail="No successful I/O pairs generated — the executable may need multi-line input. Try adding custom inputs.")

    return {
        "executable": exe_path,
        "total_inputs": total_count,
        "successful_pairs": len(pairs),
        "pairs": pairs,
    }


@app.post("/solve")
async def solve(req: SolveRequest):
    """
    Solver Pipeline:
      1. Symbolic Heuristics — instant pattern match
      2. Groq Agentic Synthesis — Code-Run-Verify loop (max 5 retries)
    """
    pairs = [{"input": p.input, "output": p.output} for p in req.pairs]

    if len(pairs) < 1:
        raise HTTPException(status_code=400, detail="Need at least 1 I/O pair")

    # Step 1: Symbolic Heuristics
    heuristic_result = symbolic_solve(pairs)
    if heuristic_result:
        # Verify heuristic solution against all pairs
        success, _ = verify_solution(heuristic_result["code"], pairs)
        if success:
            return {
                "method": "symbolic",
                "logic": heuristic_result["logic"],
                "type": heuristic_result["type"],
                "python_code": heuristic_result["code"],
                "attempts": 0,
            }

    # Step 2: AI Agentic Synthesis (20 retries, best-effort fallback)
    result = await ai_solve(pairs)

    if result["status"] in ("solved", "best_effort"):
        # Derive plain-English logic from AI
        logic_desc = "AI-inferred logic"
        try:
            if oai_client:
                logic_resp = oai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Describe code logic in one short sentence. Use math notation if helpful. Example: '(n * 2) + length'"},
                        {"role": "user", "content": f"```python\n{result['code']}\n```"},
                    ],
                    temperature=0.1,
                    max_tokens=100,
                )
                logic_desc = logic_resp.choices[0].message.content.strip()
        except Exception:
            pass

        method = "groq" if result["status"] == "solved" else "groq_best_effort"
        score_info = f" (matched {result.get('score', '?')} pairs)" if result["status"] == "best_effort" else ""

        return {
            "method": method,
            "logic": logic_desc + score_info,
            "python_code": result["code"],
            "attempts": result["attempts"],
        }

    raise HTTPException(
        status_code=422,
        detail=f"Failed to find any working solution after {result['attempts']} attempts",
    )


@app.post("/translate")
async def translate(req: TranslateRequest):
    """Translate verified Python solve() to C++."""
    if not req.python_code.strip():
        raise HTTPException(status_code=400, detail="No Python code provided")

    cpp_code = await translate_to_cpp(req.python_code)

    return {
        "python_code": req.python_code,
        "cpp_code": cpp_code,
    }


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
