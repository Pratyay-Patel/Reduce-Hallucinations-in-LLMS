"""
mac_pipeline/common/execute.py
------------------------------
Run a generated Python solution against its unit tests, in a separate interpreter with a
timeout. Used by the `codegen` dataset (HumanEval + MBPP): a question is correct when the
program exits 0, i.e. every assert passed.

The program runs with `python -I -B` (no site-packages from the user, no .pyc, no env vars
like PYTHONPATH) inside a fresh temporary directory that is deleted afterwards, and is killed
after `timeout` seconds. That is the standard protection these benchmarks use; it is not a
full sandbox, so only run model output on a machine you control.
"""

import os
import re
import subprocess
import sys
import tempfile

STATUS_PASS, STATUS_FAIL, STATUS_TIMEOUT, STATUS_ERROR = "pass", "fail", "timeout", "error"

_FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """The first fenced code block of a model reply, or the whole reply when there is none."""
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip("\n")
    # an unterminated fence: take everything after it
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*)$", text, re.S)
    if m:
        return m.group(1).strip("\n")
    return text.strip("\n")


def build_program(code: str, test: dict) -> str:
    """Assemble solution + tests into one script.
    test = {"kind": "humaneval", "prompt": ..., "entry_point": ..., "test": ...}
         | {"kind": "mbpp", "imports": [...], "asserts": [...]}"""
    if test["kind"] == "humaneval":
        ep = test["entry_point"]
        # a reply that only gives the function body (no `def`) continues the given prompt
        if not re.search(rf"\bdef\s+{re.escape(ep)}\s*\(", code):
            code = test["prompt"].rstrip("\n") + "\n" + code
        return f"{code}\n\n{test['test']}\n\ncheck({ep})\n"
    if test["kind"] == "mbpp":
        imports = "\n".join(test.get("imports") or [])
        asserts = "\n".join(test["asserts"])
        return f"{imports}\n{code}\n\n{asserts}\n"
    raise ValueError(f"unknown test kind {test['kind']}")


def run_program(src: str, timeout: float = 10.0) -> dict:
    """Execute `src` in a fresh interpreter. Returns status, exit code, and the tail of stderr."""
    with tempfile.TemporaryDirectory(prefix="codegen_") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
               "PYTHONIOENCODING": "utf-8"}
        try:
            p = subprocess.run([sys.executable, "-I", "-B", path], cwd=d, env=env,
                               capture_output=True, text=True, timeout=timeout,
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return dict(status=STATUS_TIMEOUT, returncode=None, stderr="")
        except OSError as e:
            return dict(status=STATUS_ERROR, returncode=None, stderr=str(e)[:300])
    if p.returncode == 0:
        return dict(status=STATUS_PASS, returncode=0, stderr="")
    return dict(status=STATUS_FAIL, returncode=p.returncode, stderr=(p.stderr or "")[-300:])


def check_solution(reply: str, test: dict, timeout: float = 10.0) -> dict:
    code = extract_code(reply)
    try:
        src = build_program(code, test)
    except Exception as e:                           # malformed test record
        return dict(status=STATUS_ERROR, returncode=None, stderr=str(e)[:300], code=code)
    r = run_program(src, timeout)
    r["code"] = code
    return r
