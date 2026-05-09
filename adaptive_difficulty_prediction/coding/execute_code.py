"""
Safe code execution sandbox for evaluating generated coding solutions.

Handles both stdin/stdout problems and function-call problems.
Uses subprocess with timeout and resource limits.

Usage (standalone test):
    python execute_code.py --code "print(int(input())+1)" --stdin "5" --expected "6"
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import traceback
from typing import Optional


DEFAULT_TIMEOUT = 10
DEFAULT_MAX_OUTPUT = 50_000


def extract_code(raw_output: str) -> str:
    """Extract Python code from model output that may contain markdown fences."""
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```Python\s*\n(.*?)```",
        r"```py\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pat in patterns:
        matches = re.findall(pat, raw_output, re.DOTALL)
        if matches:
            return matches[-1].strip()

    lines = raw_output.strip().split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or _looks_like_code(stripped):
            code_lines.append(line)
            in_code = True

    if code_lines:
        return "\n".join(code_lines).strip()

    return raw_output.strip()


def _looks_like_code(line: str) -> bool:
    """Heuristic: does this line look like Python code?"""
    code_indicators = [
        "import ", "from ", "def ", "class ", "if ", "for ", "while ",
        "return ", "print(", "input(", "sys.", "=", "#",
    ]
    return any(line.startswith(ind) or line.lstrip().startswith(ind)
               for ind in code_indicators)


def execute_stdio(
    code: str,
    stdin_input: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_output: int = DEFAULT_MAX_OUTPUT,
) -> tuple[bool, str, str]:
    """
    Execute code with stdin input, return (success, stdout, stderr).
    success means the process exited with code 0 and produced output.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, "-u", tmp_path],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        stdout = result.stdout[:max_output]
        stderr = result.stderr[:max_output]
        return result.returncode == 0, stdout, stderr
    except subprocess.TimeoutExpired:
        return False, "", "TimeoutExpired"
    except Exception as e:
        return False, "", str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def execute_function_call(
    code: str,
    fn_name: str,
    fn_inputs: list,
    timeout: int = DEFAULT_TIMEOUT,
    max_output: int = DEFAULT_MAX_OUTPUT,
) -> tuple[bool, str, str]:
    """Execute code and call a specific function with given inputs."""
    wrapper = f"""
import sys, json
{code}

_inputs = json.loads(sys.argv[1])
_result = {fn_name}(*_inputs)
print(json.dumps(_result))
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(wrapper)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, "-u", tmp_path, json.dumps(fn_inputs)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        stdout = result.stdout[:max_output]
        stderr = result.stderr[:max_output]
        return result.returncode == 0, stdout, stderr
    except subprocess.TimeoutExpired:
        return False, "", "TimeoutExpired"
    except Exception as e:
        return False, "", str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def compare_output(actual: str, expected: str, tolerance: float = 1e-6) -> bool:
    """Compare actual vs expected output with whitespace normalization."""
    actual_clean = actual.strip()
    expected_clean = expected.strip()

    if actual_clean == expected_clean:
        return True

    actual_lines = [l.strip() for l in actual_clean.splitlines() if l.strip()]
    expected_lines = [l.strip() for l in expected_clean.splitlines() if l.strip()]

    if len(actual_lines) != len(expected_lines):
        return False

    for a, e in zip(actual_lines, expected_lines):
        if a == e:
            continue
        a_tokens = a.split()
        e_tokens = e.split()
        if len(a_tokens) != len(e_tokens):
            return False
        for at, et in zip(a_tokens, e_tokens):
            if at == et:
                continue
            try:
                if abs(float(at) - float(et)) < tolerance:
                    continue
            except ValueError:
                pass
            return False
    return True


def compare_function_output(actual_stdout: str, expected_output, tolerance: float = 1e-6) -> bool:
    """Compare function call output (JSON-encoded) with expected."""
    try:
        actual = json.loads(actual_stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return compare_output(actual_stdout.strip(), str(expected_output).strip(), tolerance)

    if actual == expected_output:
        return True
    if isinstance(actual, float) and isinstance(expected_output, (int, float)):
        return abs(actual - expected_output) < tolerance
    if isinstance(actual, list) and isinstance(expected_output, list):
        if len(actual) != len(expected_output):
            return False
        return all(
            (a == e) or (isinstance(a, float) and isinstance(e, (int, float)) and abs(a - e) < tolerance)
            for a, e in zip(actual, expected_output)
        )
    return str(actual).strip() == str(expected_output).strip()


def evaluate_solution(
    code: str,
    input_output: dict,
    fn_name: Optional[str] = None,
    timeout_per_case: int = DEFAULT_TIMEOUT,
    max_test_cases: int = 50,
) -> dict:
    """
    Evaluate a single code solution against test cases.

    Returns:
        {
            "passed": int,
            "total": int,
            "reward": float (0 or 1, pass-all-or-nothing),
            "pass_rate": float,
            "errors": [str, ...],
            "details": [{"passed": bool, "error": str | None}, ...]
        }
    """
    is_fn = bool(fn_name)
    inputs = input_output.get("inputs", [])
    outputs = input_output.get("outputs", [])
    n_cases = min(len(inputs), len(outputs), max_test_cases)

    if n_cases == 0:
        return {"passed": 0, "total": 0, "reward": 0, "pass_rate": 0.0,
                "errors": ["no test cases"], "details": []}

    passed = 0
    details = []
    errors = []

    for i in range(n_cases):
        try:
            if is_fn:
                ok, stdout, stderr = execute_function_call(
                    code, fn_name, inputs[i], timeout=timeout_per_case
                )
                if ok:
                    correct = compare_function_output(stdout, outputs[i])
                else:
                    correct = False
            else:
                stdin_text = inputs[i] if isinstance(inputs[i], str) else str(inputs[i])
                expected = outputs[i] if isinstance(outputs[i], str) else str(outputs[i])
                ok, stdout, stderr = execute_stdio(
                    code, stdin_text, timeout=timeout_per_case
                )
                correct = ok and compare_output(stdout, expected)

            if not correct and stderr:
                errors.append(f"case_{i}: {stderr[:200]}")

            details.append({"passed": correct, "error": stderr[:200] if stderr else None})
            if correct:
                passed += 1

        except Exception as e:
            details.append({"passed": False, "error": str(e)[:200]})
            errors.append(f"case_{i}: {str(e)[:200]}")

    pass_rate = passed / n_cases if n_cases > 0 else 0.0
    reward = 1 if passed == n_cases else 0

    return {
        "passed": passed,
        "total": n_cases,
        "reward": reward,
        "pass_rate": pass_rate,
        "errors": errors[:5],
        "details": details,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=str, help="Code string to execute")
    parser.add_argument("--stdin", type=str, default="", help="Stdin input")
    parser.add_argument("--expected", type=str, default="", help="Expected output")
    args = parser.parse_args()

    if args.code:
        ok, stdout, stderr = execute_stdio(args.code, args.stdin)
        print(f"Exit OK: {ok}")
        print(f"Stdout: {repr(stdout)}")
        print(f"Stderr: {repr(stderr)}")
        if args.expected:
            match = compare_output(stdout, args.expected)
            print(f"Match: {match}")
    else:
        test_code = "n = int(input())\nprint(n * 2)"
        ok, stdout, stderr = execute_stdio(test_code, "21\n")
        assert ok and stdout.strip() == "42", f"Self-test failed: {stdout!r} {stderr!r}"
        print("Self-test passed: 21*2 = 42")

        test_io = {"inputs": ["5\n", "10\n"], "outputs": ["10\n", "20\n"]}
        result = evaluate_solution(test_code, test_io)
        print(f"Evaluate result: {result}")
        assert result["reward"] == 1
        print("All self-tests passed.")
