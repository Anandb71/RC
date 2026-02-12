"""
RC-Oracle Backend — Industry-Grade C++ Solver
FastAPI server with Data Factory, Symbolic Heuristics, and Groq Agentic Loop.
Generates C++ code natively — no translation step.
"""

import os
import re
import sys
import json
import math
import base64
import subprocess
import tempfile
import shutil
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
    boilerplate: Optional[str] = None   # Editor template from portal
    description: Optional[str] = None   # Problem description text

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

CPP_TEMPLATE = '''#include <bits/stdc++.h>
using namespace std;

void solve() {{
{body}
}}

int main() {{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    solve();
    return 0;
}}
'''

def make_cpp(body: str) -> str:
    """Wrap a solve() body in the standard C++ template."""
    return CPP_TEMPLATE.format(body=body)


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
        return {"type": "identity", "logic": "y = x",
            "code": make_cpp('    long long x; cin >> x; cout << x << endl;')}

    # Constant: all outputs the same
    outputs = [y for _, y in numeric_pairs]
    if len(set(outputs)) == 1:
        c = int(outputs[0]) if outputs[0] == int(outputs[0]) else outputs[0]
        return {"type": "constant", "logic": f"y = {c}",
            "code": make_cpp(f'    cout << {c} << endl;')}

    # Linear: y = a*x + b
    if len(numeric_pairs) >= 2:
        x1, y1 = numeric_pairs[0]
        x2, y2 = numeric_pairs[1]
        if x2 != x1:
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            if all(abs(a * x + b - y) < 1e-9 for x, y in numeric_pairs):
                a_int = int(a) if a == int(a) else a
                b_int = int(b) if b == int(b) else b
                if b_int == 0:
                    logic = f"y = {a_int} * x"
                    body = f'    long long x; cin >> x; cout << x * {a_int} << endl;'
                elif a_int == 1:
                    logic = f"y = x + {b_int}"
                    body = f'    long long x; cin >> x; cout << x + {b_int} << endl;'
                else:
                    sign = "+" if b_int >= 0 else "-"
                    logic = f"y = {a_int} * x {sign} {abs(b_int)}"
                    body = f'    long long x; cin >> x; cout << x * {a_int} + {b_int} << endl;'
                return {"type": "linear", "logic": logic, "code": make_cpp(body)}

    # Quadratic: y = x^2
    if all(y == x * x for x, y in numeric_pairs):
        return {"type": "quadratic", "logic": "y = x²",
            "code": make_cpp('    long long x; cin >> x; cout << x * x << endl;')}

    # Power of 2: y = 2^x
    if all(x >= 0 and y == 2**x for x, y in numeric_pairs):
        return {"type": "power_of_2", "logic": "y = 2^x",
            "code": make_cpp('    long long x; cin >> x; cout << (1LL << x) << endl;')}

    # Factorial
    def factorial_match(x, y):
        if x < 0 or x != int(x): return False
        return y == math.factorial(int(x))

    if all(x >= 0 for x, _ in numeric_pairs) and all(factorial_match(x, y) for x, y in numeric_pairs):
        return {"type": "factorial", "logic": "y = x!",
            "code": make_cpp('    long long x; cin >> x;\n    long long f = 1;\n    for (int i = 2; i <= x; i++) f *= i;\n    cout << f << endl;')}

    # Absolute value
    if all(y == abs(x) for x, y in numeric_pairs):
        return {"type": "absolute", "logic": "y = |x|",
            "code": make_cpp('    long long x; cin >> x; cout << abs(x) << endl;')}

    return None


def try_string_transforms(pairs: list[dict]) -> Optional[dict]:
    """Check for common string transformations. All output C++ code."""
    if not pairs:
        return None

    # Reverse
    if all(p["output"] == p["input"][::-1] for p in pairs):
        return {"type": "reverse", "logic": "reverse the string",
            "code": make_cpp('    string s; cin >> s;\n    reverse(s.begin(), s.end());\n    cout << s << endl;')}

    # Uppercase
    if all(p["output"] == p["input"].upper() for p in pairs):
        return {"type": "upper", "logic": "convert to UPPERCASE",
            "code": make_cpp('    string s; cin >> s;\n    transform(s.begin(), s.end(), s.begin(), ::toupper);\n    cout << s << endl;')}

    # Lowercase
    if all(p["output"] == p["input"].lower() for p in pairs):
        return {"type": "lower", "logic": "convert to lowercase",
            "code": make_cpp('    string s; cin >> s;\n    transform(s.begin(), s.end(), s.begin(), ::tolower);\n    cout << s << endl;')}

    # String length
    if all(p["output"] == str(len(p["input"])) for p in pairs):
        return {"type": "length", "logic": "y = length of input string",
            "code": make_cpp('    string s; cin >> s;\n    cout << s.size() << endl;')}

    # Sort characters
    if all(p["output"] == "".join(sorted(p["input"])) for p in pairs):
        return {"type": "sort_chars", "logic": "sort characters alphabetically",
            "code": make_cpp('    string s; cin >> s;\n    sort(s.begin(), s.end());\n    cout << s << endl;')}

    return None


def try_encoding_transforms(pairs: list[dict]) -> Optional[dict]:
    """Check for Caesar cipher. All output C++ code."""
    if not pairs or not all(p["input"] for p in pairs):
        return None

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
            body = f'''    string s; cin >> s;
    for (auto &c : s) {{
        if (isalpha(c)) {{
            char base = isupper(c) ? 'A' : 'a';
            c = (c - base + {shift}) % 26 + base;
        }}
    }}
    cout << s << endl;'''
            return {"type": "caesar", "logic": f"Caesar cipher (shift {shift})",
                "code": make_cpp(body)}

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


# ─── Groq Agentic Synthesis — C++ Native Code-Compile-Verify Loop ─────────

# Check for g++ compiler
GPP_PATH = shutil.which("g++")
if GPP_PATH:
    print(f"✅ g++ found: {GPP_PATH}")
else:
    print("⚠ g++ not found — C++ verification will be disabled")

SYSTEM_PROMPT = """You are an elite competitive programmer who writes ONLY C++ code.
You are given input-output pairs from a black-box function.
Your task is to analyze the pattern and write a COMPLETE C++ program that reads from stdin and writes to stdout.

MANDATORY STRUCTURE:
```cpp
#include <bits/stdc++.h>
using namespace std;

void solve() {
    // Read from cin, compute, write to cout
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    solve();
    return 0;
}
```

CRITICAL RULES:
1. Write ONLY C++ code. NEVER write Python. NEVER use def. NEVER use print().
2. Use #include <bits/stdc++.h> and using namespace std;
3. The solve() function must be void — read from cin, write to cout.
4. Use `long long` for integers to avoid overflow.
5. Output ONLY the complete C++ code. No explanations, no markdown fences outside the code.
6. Handle edge cases: negative numbers, zero, single values, large values.
7. Use `endl` or `"\n"` to end output lines.

INPUT FORMAT HANDLING:
- Single integer: `long long n; cin >> n;`
- N then N numbers: `int n; cin >> n; vector<long long> a(n); for(auto &x : a) cin >> x;`
- Space-separated: read until EOF or known count.
- Multi-line: use getline() or multiple cin >>.
- Strings: `string s; cin >> s;` or `getline(cin, s);`

COMMON COMPETITIVE PROGRAMMING PATTERNS:
- Arithmetic: sums, products, averages, min/max
- Number theory: primes, GCD, LCM, modular arithmetic, divisors
- Bit manipulation: XOR, AND, OR, bit reversal, popcount, __builtin_popcount()
- Binary representation: converting to/from binary
- String operations: reversal, sorting, substring, counting
- Array operations: sorting, prefix sums, cumulative operations
- Math formulas: polynomials, sequences, combinatorics, Fibonacci
- Digit operations: digit sum, digit count, digit reversal

REMEMBER: You are a C++ ONLY programmer. Never output any other language.
"""

def extract_cpp_code(response_text: str) -> str:
    """Extract C++ code from the AI response."""
    # Try markdown fences first
    fence_match = re.search(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", response_text, re.DOTALL)
    if fence_match:
        code = fence_match.group(1).strip()
    else:
        code = response_text.strip()

    # Validate it's actually C++
    if "def solve" in code or "def " in code:
        raise ValueError("AI returned Python instead of C++")
    cpp_markers = ["#include", "int main", "void solve", "cin", "cout", "iostream"]
    if not any(m in code for m in cpp_markers):
        raise ValueError("Response does not contain valid C++ code")

    # Clean up: remove any text before #include
    lines = code.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("#include") or line.strip().startswith("using "):
            start_idx = i
            break
    code = "\n".join(lines[start_idx:])
    return code


def compile_cpp(code: str) -> tuple[Optional[str], Optional[str]]:
    """Compile C++ code with g++. Returns (exe_path, error_msg)."""
    if not GPP_PATH:
        return None, "g++ not found"

    tmpdir = tempfile.mkdtemp(prefix="rc_oracle_")
    src_path = os.path.join(tmpdir, "solution.cpp")
    exe_path = os.path.join(tmpdir, "solution.exe" if sys.platform == "win32" else "solution")

    with open(src_path, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        result = subprocess.run(
            [GPP_PATH, "-O2", "-std=c++17", "-o", exe_path, src_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None, result.stderr.strip()
        return exe_path, None
    except subprocess.TimeoutExpired:
        return None, "Compilation timed out"
    except Exception as e:
        return None, str(e)


def run_cpp(exe_path: str, test_input: str, timeout: float = 10.0) -> str:
    """Run compiled C++ binary with input, return stdout."""
    full_input = test_input if test_input.endswith("\n") else test_input + "\n"
    result = subprocess.run(
        [exe_path], input=full_input, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Runtime error: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_cpp_solution(code: str, pairs: list[dict]) -> tuple[bool, Optional[dict]]:
    """Compile C++ code and test against all I/O pairs."""
    exe_path, compile_err = compile_cpp(code)
    if compile_err:
        return False, {"input": "", "expected": "", "got": None, "error": f"Compile error: {compile_err}"}

    try:
        for p in pairs:
            try:
                got = run_cpp(exe_path, p["input"]).strip()
                expected = p["output"].strip()
                if got != expected:
                    return False, {"input": p["input"], "expected": expected, "got": got, "error": None}
            except Exception as e:
                return False, {"input": p["input"], "expected": p["output"], "got": None, "error": str(e)}
        return True, None
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(os.path.dirname(exe_path), ignore_errors=True)
        except Exception:
            pass


def score_cpp_solution(code: str, pairs: list[dict]) -> int:
    """Count how many I/O pairs the compiled C++ code matches."""
    exe_path, compile_err = compile_cpp(code)
    if compile_err:
        return 0

    matched = 0
    try:
        for p in pairs:
            try:
                got = run_cpp(exe_path, p["input"]).strip()
                if got == p["output"].strip():
                    matched += 1
            except Exception:
                pass
    finally:
        try:
            shutil.rmtree(os.path.dirname(exe_path), ignore_errors=True)
        except Exception:
            pass
    return matched


async def ai_solve(pairs: list[dict], boilerplate: Optional[str] = None, description: Optional[str] = None, max_retries: int = 20) -> dict:
    """Agentic loop: ask AI for C++ code, compile, run, verify, retry on failure.
    Tracks the best-scoring attempt so even on total failure we return closest solution."""
    if not oai_client:
        raise HTTPException(status_code=503, detail="Groq API key not configured")

    ai_pairs = select_best_pairs(pairs, max_pairs=15)

    pair_text = "\n".join(f"  Input: {p['input']!r}  →  Output: {p['output']!r}" for p in ai_pairs)

    # Build user prompt with optional boilerplate and description
    user_prompt_parts = [f"Here are {len(ai_pairs)} input-output pairs from a black-box program:\n{pair_text}\n"]

    if description:
        user_prompt_parts.append(f"\nPROBLEM DESCRIPTION from the portal:\n{description}\n")

    if boilerplate and boilerplate.strip():
        user_prompt_parts.append(
            f"\nIMPORTANT — The portal provides this BOILERPLATE TEMPLATE that you MUST use:\n"
            f"```cpp\n{boilerplate}\n```\n"
            f"You MUST keep the template structure intact. Fill in the logic inside the template.\n"
            f"Do NOT add extra #include lines if they are already in the template.\n"
            f"Do NOT change main() if it's already defined in the template.\n"
            f"ONLY add your solution logic where the template expects it (comments like '// write your code here', empty function bodies, etc.)."
        )
    else:
        user_prompt_parts.append(
            "\nNo template provided. Write a COMPLETE standalone C++ program.\n"
            "Use #include <bits/stdc++.h>, void solve(), read cin, write cout."
        )

    user_prompt_parts.append("\nAnalyze the pattern carefully. REMEMBER: C++ ONLY. NEVER use Python.")
    user_prompt = "\n".join(user_prompt_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    best_code = None
    best_score = -1
    best_attempt = 0

    for attempt in range(1, max_retries + 1):
        try:
            temp = 0.2 if attempt <= 10 else 0.5
            response = oai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temp,
                max_tokens=2048,
            )
            response_text = response.choices[0].message.content
            messages.append({"role": "assistant", "content": response_text})

            code = extract_cpp_code(response_text)

            # Score this attempt
            score = score_cpp_solution(code, ai_pairs)
            if score > best_score:
                best_score = score
                best_code = code
                best_attempt = attempt

            success, failure = verify_cpp_solution(code, ai_pairs)

            if success:
                return {"status": "solved", "code": code, "attempts": attempt}

            # Build retry prompt with failure info
            if failure["error"]:
                retry_msg = (
                    f"Your C++ solution FAILED on attempt {attempt}.\n"
                    f"Input: {failure['input']!r}\n"
                    f"Expected: {failure['expected']!r}\n"
                    f"Error: {failure['error']}\n\n"
                    f"Fix the C++ code. Output ONLY the corrected COMPLETE C++ program.\n"
                    f"DO NOT use Python. Use #include <bits/stdc++.h>, void solve(), cin/cout."
                )
            else:
                retry_msg = (
                    f"Your C++ solution FAILED on attempt {attempt}.\n"
                    f"Input: {failure['input']!r}\n"
                    f"Expected: {failure['expected']!r}\n"
                    f"Got: {failure['got']!r}\n\n"
                    f"Fix the C++ code. Output ONLY the corrected COMPLETE C++ program.\n"
                    f"DO NOT use Python. Use #include <bits/stdc++.h>, void solve(), cin/cout."
                )
            messages.append({"role": "user", "content": retry_msg})

            # Reset conversation every 8 attempts
            if attempt % 8 == 0 and attempt < max_retries:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt + f"\n\nPrevious {attempt} attempts all failed. Try a COMPLETELY DIFFERENT algorithm."},
                ]

        except Exception as e:
            error_msg = f"Error with your C++ code: {str(e)}\nFix and output ONLY the corrected COMPLETE C++ program. NO PYTHON."
            messages.append({"role": "user", "content": error_msg})

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


# Translation step removed — AI generates C++ natively now.


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
    Industry-Grade C++ Solver Pipeline:
      1. Symbolic Heuristics — instant pattern match → C++ code
      2. Groq AI — C++ native Code-Compile-Verify loop (20 retries)
    """
    pairs = [{"input": p.input, "output": p.output} for p in req.pairs]

    if len(pairs) < 1:
        raise HTTPException(status_code=400, detail="Need at least 1 I/O pair")

    # Step 1: Symbolic Heuristics (outputs C++ directly)
    heuristic_result = symbolic_solve(pairs)
    if heuristic_result:
        success, _ = verify_cpp_solution(heuristic_result["code"], pairs)
        if success:
            return {
                "method": "symbolic",
                "logic": heuristic_result["logic"],
                "type": heuristic_result["type"],
                "cpp_code": heuristic_result["code"],
                "attempts": 0,
            }

    # Step 2: AI Agentic Synthesis — C++ native (20 retries, best-effort fallback)
    result = await ai_solve(pairs, boilerplate=req.boilerplate, description=req.description)

    if result["status"] in ("solved", "best_effort"):
        # Derive logic description
        logic_desc = "AI-inferred logic"
        try:
            if oai_client:
                logic_resp = oai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Describe code logic in one short sentence. Use math notation if helpful."},
                        {"role": "user", "content": f"```cpp\n{result['code']}\n```"},
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
            "cpp_code": result["code"],
            "attempts": result["attempts"],
        }

    raise HTTPException(
        status_code=422,
        detail=f"Failed to find any working solution after {result['attempts']} attempts",
    )


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
