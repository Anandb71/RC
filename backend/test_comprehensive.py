"""Comprehensive test suite for RC-Oracle backend."""
import requests
import json
import time
import sys

BASE = "http://localhost:8000"

def test(name, fn):
    try:
        result = fn()
        print(f"  ✅ {name}: {result}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return False

def run_all():
    passed = 0
    failed = 0

    print("\n═══ 1. Health Check ═══")
    def t1():
        r = requests.get(f"{BASE}/", timeout=5)
        assert r.status_code == 200
        d = r.json()
        assert d["ai"] == "connected"
        return f"status={d['status']}, ai={d['ai']}, model={d['model']}"
    if test("Backend online + Groq connected", t1): passed += 1
    else: failed += 1

    print("\n═══ 2. Symbolic Solver Tests ═══")

    # Test: identity
    def t2():
        r = requests.post(f"{BASE}/solve", json={"pairs": [
            {"input": "5", "output": "5"},
            {"input": "10", "output": "10"},
            {"input": "42", "output": "42"},
        ]}, timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert d["method"] == "symbolic"
        return f"method={d['method']}, logic={d['logic']}"
    if test("Identity (echo)", t2): passed += 1
    else: failed += 1

    # Test: constant
    def t3():
        r = requests.post(f"{BASE}/solve", json={"pairs": [
            {"input": "1", "output": "42"},
            {"input": "99", "output": "42"},
            {"input": "hello", "output": "42"},
        ]}, timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert d["method"] == "symbolic"
        return f"method={d['method']}, logic={d['logic']}"
    if test("Constant output", t3): passed += 1
    else: failed += 1

    # Test: reverse
    def t4():
        r = requests.post(f"{BASE}/solve", json={"pairs": [
            {"input": "hello", "output": "olleh"},
            {"input": "abc", "output": "cba"},
            {"input": "world", "output": "dlrow"},
        ]}, timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert d["method"] == "symbolic"
        return f"method={d['method']}, logic={d['logic']}"
    if test("String reverse", t4): passed += 1
    else: failed += 1

    # Test: length
    def t5():
        r = requests.post(f"{BASE}/solve", json={"pairs": [
            {"input": "hello", "output": "5"},
            {"input": "abc", "output": "3"},
            {"input": "ab", "output": "2"},
        ]}, timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert d["method"] == "symbolic"
        return f"method={d['method']}, logic={d['logic']}"
    if test("String length", t5): passed += 1
    else: failed += 1

    print("\n═══ 3. AI Solver Tests ═══")

    # Test: n*2+1
    def t6():
        r = requests.post(f"{BASE}/solve", json={"pairs": [
            {"input": "1", "output": "3"},
            {"input": "2", "output": "5"},
            {"input": "3", "output": "7"},
            {"input": "5", "output": "11"},
            {"input": "10", "output": "21"},
        ]}, timeout=120)
        assert r.status_code == 200
        d = r.json()
        return f"method={d['method']}, logic={d['logic']}, attempts={d['attempts']}"
    if test("n*2+1 pattern", t6): passed += 1
    else: failed += 1

    # Test: n squared
    def t7():
        r = requests.post(f"{BASE}/solve", json={"pairs": [
            {"input": "1", "output": "1"},
            {"input": "2", "output": "4"},
            {"input": "3", "output": "9"},
            {"input": "5", "output": "25"},
            {"input": "10", "output": "100"},
        ]}, timeout=120)
        assert r.status_code == 200
        d = r.json()
        return f"method={d['method']}, logic={d['logic']}, attempts={d['attempts']}"
    if test("n^2 pattern", t7): passed += 1
    else: failed += 1

    # Test: factorial
    def t8():
        r = requests.post(f"{BASE}/solve", json={"pairs": [
            {"input": "1", "output": "1"},
            {"input": "2", "output": "2"},
            {"input": "3", "output": "6"},
            {"input": "4", "output": "24"},
            {"input": "5", "output": "120"},
            {"input": "6", "output": "720"},
        ]}, timeout=120)
        assert r.status_code == 200
        d = r.json()
        return f"method={d['method']}, logic={d['logic']}, attempts={d['attempts']}"
    if test("Factorial", t8): passed += 1
    else: failed += 1

    # Test: sum of array (multi-line input)
    def t9():
        r = requests.post(f"{BASE}/solve", json={"pairs": [
            {"input": "3\n1 2 3", "output": "6"},
            {"input": "5\n10 20 30 40 50", "output": "150"},
            {"input": "1\n100", "output": "100"},
            {"input": "4\n5 5 5 5", "output": "20"},
            {"input": "2\n-1 1", "output": "0"},
        ]}, timeout=120)
        assert r.status_code == 200
        d = r.json()
        return f"method={d['method']}, logic={d['logic']}, attempts={d['attempts']}"
    if test("Sum of array (multi-line)", t9): passed += 1
    else: failed += 1

    # Test: max of array (multi-line input)
    def t10():
        r = requests.post(f"{BASE}/solve", json={"pairs": [
            {"input": "3\n1 5 3", "output": "5"},
            {"input": "5\n10 20 5 40 50", "output": "50"},
            {"input": "1\n100", "output": "100"},
            {"input": "4\n5 8 2 9", "output": "9"},
            {"input": "3\n-1 -5 -2", "output": "-1"},
        ]}, timeout=120)
        assert r.status_code == 200
        d = r.json()
        return f"method={d['method']}, logic={d['logic']}, attempts={d['attempts']}"
    if test("Max of array (multi-line)", t10): passed += 1
    else: failed += 1

    # Test: fibonacci
    def t11():
        r = requests.post(f"{BASE}/solve", json={"pairs": [
            {"input": "1", "output": "1"},
            {"input": "2", "output": "1"},
            {"input": "3", "output": "2"},
            {"input": "4", "output": "3"},
            {"input": "5", "output": "5"},
            {"input": "6", "output": "8"},
            {"input": "7", "output": "13"},
            {"input": "10", "output": "55"},
        ]}, timeout=120)
        assert r.status_code == 200
        d = r.json()
        return f"method={d['method']}, logic={d['logic']}, attempts={d['attempts']}"
    if test("Fibonacci", t11): passed += 1
    else: failed += 1

    # Test: count digits
    def t12():
        r = requests.post(f"{BASE}/solve", json={"pairs": [
            {"input": "0", "output": "1"},
            {"input": "9", "output": "1"},
            {"input": "10", "output": "2"},
            {"input": "99", "output": "2"},
            {"input": "100", "output": "3"},
            {"input": "999", "output": "3"},
            {"input": "1000", "output": "4"},
        ]}, timeout=120)
        assert r.status_code == 200
        d = r.json()
        return f"method={d['method']}, logic={d['logic']}, attempts={d['attempts']}"
    if test("Count digits", t12): passed += 1
    else: failed += 1

    print("\n═══ 4. C++ Translation Test ═══")
    def t13():
        r = requests.post(f"{BASE}/translate", json={
            "python_code": "def solve(s):\n    n = int(s)\n    return str(n * 2 + 1)"
        }, timeout=30)
        assert r.status_code == 200
        d = r.json()
        assert "void solve" in d["cpp_code"] or "int main" in d["cpp_code"]
        return f"C++ generated, length={len(d['cpp_code'])}"
    if test("Python->C++ translation", t13): passed += 1
    else: failed += 1

    print("\n═══ 5. Edge Cases ═══")
    # Test: single pair
    def t14():
        r = requests.post(f"{BASE}/solve", json={"pairs": [
            {"input": "5", "output": "10"},
        ]}, timeout=120)
        # Should not crash, may or may not solve
        return f"status={r.status_code}"
    if test("Single pair (no crash)", t14): passed += 1
    else: failed += 1

    # Test: empty pairs
    def t15():
        r = requests.post(f"{BASE}/solve", json={"pairs": []}, timeout=10)
        assert r.status_code == 400
        return f"Correctly rejected with 400"
    if test("Empty pairs rejected", t15): passed += 1
    else: failed += 1

    print(f"\n{'='*50}")
    print(f"  Results: {passed}/{passed+failed} passed, {failed} failed")
    print(f"{'='*50}\n")

    return failed == 0

if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
