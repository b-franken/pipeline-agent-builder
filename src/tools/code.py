"""Code analysis, formatting, and sandboxed execution tools."""

import atexit
from typing import Any

from langchain_core.tools import StructuredTool

from src.config import settings


def _analyze_code(code: str, language: str = "python") -> str:
    issues: list[str] = []
    suggestions: list[str] = []

    if language == "python":
        if "import *" in code:
            issues.append("Avoid 'import *'")
        if "except:" in code and "except Exception" not in code:
            issues.append("Bare except - specify exception type")
        if "== None" in code or "!= None" in code:
            suggestions.append("Use 'is None' instead")
        if "eval(" in code or "exec(" in code:
            issues.append("Security risk: eval/exec usage")
        if "os.system(" in code or "subprocess.call(" in code:
            suggestions.append("Consider subprocess.run with shell=False")

    result = f"Analysis ({language}):\n"
    if issues:
        result += "Issues:\n" + "\n".join(f"  - {i}" for i in issues) + "\n"
    if suggestions:
        result += "Suggestions:\n" + "\n".join(f"  - {s}" for s in suggestions) + "\n"
    if not issues and not suggestions:
        result += "No issues found."

    return result


def _format_code(code: str, language: str = "python") -> str:
    if language == "python":
        try:
            import black
            from black.parsing import InvalidInput

            formatted = black.format_str(code, mode=black.Mode())
            return f"```{language}\n{formatted}```"
        except ImportError:
            return f"[Black not installed]\n```{language}\n{code}\n```"
        except InvalidInput as e:
            return f"[Format error: {e}]\n```{language}\n{code}\n```"
    return f"```{language}\n{code}\n```"


class E2BExecutor:
    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self._sandbox: Any = None

    def _get_sandbox(self) -> Any:
        if self._sandbox is None:
            try:
                from e2b_code_interpreter import Sandbox

                self._sandbox = Sandbox()
            except ImportError as e:
                raise ImportError("e2b-code-interpreter not installed. Run: pip install e2b-code-interpreter") from e
            except Exception as e:
                if "E2B_API_KEY" in str(e) or "api key" in str(e).lower():
                    raise ValueError("E2B_API_KEY environment variable not set. Get your key at https://e2b.dev") from e
                raise
        return self._sandbox

    def execute(self, code: str) -> str:
        try:
            sandbox = self._get_sandbox()
            execution = sandbox.run_code(code)

            output_parts: list[str] = []

            if execution.logs.stdout:
                output_parts.append("stdout:\n" + "\n".join(execution.logs.stdout))

            if execution.logs.stderr:
                output_parts.append("stderr:\n" + "\n".join(execution.logs.stderr))

            if execution.results:
                for result in execution.results:
                    if hasattr(result, "text") and result.text:
                        output_parts.append(f"Result: {result.text}")
                    elif hasattr(result, "png") and result.png:
                        output_parts.append("[Chart/Image generated]")

            if execution.error:
                output_parts.append(f"Error: {execution.error.name}: {execution.error.value}")

            return "\n\n".join(output_parts) if output_parts else "Code executed successfully (no output)"

        except ImportError as e:
            return f"E2B not available: {e}"
        except ValueError as e:
            return f"E2B configuration error: {e}"
        except Exception as e:
            return f"Execution error: {type(e).__name__}: {e}"

    def close(self) -> None:
        if self._sandbox is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._sandbox.kill()
            self._sandbox = None


_executor: E2BExecutor | None = None


def _get_executor() -> E2BExecutor:
    global _executor
    if _executor is None:
        _executor = E2BExecutor()
        atexit.register(_executor.close)
    return _executor


def _execute_python_safe(code: str) -> str:
    if not settings.e2b_api_key:
        return (
            "[E2B_API_KEY not set - execution disabled]\n"
            "Set E2B_API_KEY to enable sandboxed execution.\n"
            f"```python\n{code}\n```"
        )

    executor = _get_executor()
    return executor.execute(code)


analyze_code = StructuredTool.from_function(
    func=_analyze_code,
    name="analyze_code",
    description="Analyze code for common issues and anti-patterns.",
)

format_code = StructuredTool.from_function(
    func=_format_code,
    name="format_code",
    description="Format Python code using Black. Returns formatted code block.",
)

execute_python_safe = StructuredTool.from_function(
    func=_execute_python_safe,
    name="execute_python_safe",
    description="Execute Python code in E2B sandbox. Requires E2B_API_KEY.",
)


def create_code_tools() -> list[StructuredTool]:
    return [analyze_code, format_code, execute_python_safe]
