from unittest.mock import MagicMock, patch

from src.tools.code import (
    _analyze_code,
    _format_code,
    analyze_code,
    create_code_tools,
    format_code,
)


class TestAnalyzeCode:
    def test_detects_import_star(self) -> None:
        code = "from os import *"
        result = _analyze_code(code, "python")
        assert "Avoid 'import *'" in result

    def test_detects_bare_except(self) -> None:
        code = "try:\n    pass\nexcept:\n    pass"
        result = _analyze_code(code, "python")
        assert "Bare except" in result

    def test_does_not_flag_except_exception(self) -> None:
        code = "try:\n    pass\nexcept Exception:\n    pass"
        result = _analyze_code(code, "python")
        assert "Bare except" not in result

    def test_suggests_is_none(self) -> None:
        code = "if x == None:\n    pass"
        result = _analyze_code(code, "python")
        assert "is None" in result

    def test_detects_eval_security_risk(self) -> None:
        code = "eval(user_input)"
        result = _analyze_code(code, "python")
        assert "Security risk" in result
        assert "eval/exec" in result

    def test_detects_exec_security_risk(self) -> None:
        code = "exec(user_input)"
        result = _analyze_code(code, "python")
        assert "Security risk" in result

    def test_suggests_subprocess_run(self) -> None:
        code = "os.system('ls')"
        result = _analyze_code(code, "python")
        assert "subprocess.run" in result

    def test_clean_code_no_issues(self) -> None:
        code = "def foo():\n    return 42"
        result = _analyze_code(code, "python")
        assert "No issues found" in result

    def test_non_python_language(self) -> None:
        code = "console.log('hello')"
        result = _analyze_code(code, "javascript")
        assert "Analysis (javascript)" in result
        assert "No issues found" in result

    def test_multiple_issues(self) -> None:
        code = "from os import *\neval(x)\nif y == None: pass"
        result = _analyze_code(code, "python")
        assert "Avoid 'import *'" in result
        assert "Security risk" in result
        assert "is None" in result


class TestFormatCode:
    def test_format_python_with_black(self) -> None:
        code = "def foo():    return 42"
        result = _format_code(code, "python")
        assert "```python" in result
        assert "def foo():" in result

    def test_format_python_invalid_syntax(self) -> None:
        code = "def foo( invalid syntax"
        result = _format_code(code, "python")
        assert "[Format error" in result or "```python" in result

    def test_format_non_python_language(self) -> None:
        code = "function foo() { return 42; }"
        result = _format_code(code, "javascript")
        assert "```javascript" in result
        assert code in result

    def test_format_code_tool_exists(self) -> None:
        assert format_code is not None
        assert format_code.name == "format_code"


class TestAnalyzeCodeTool:
    def test_analyze_code_tool_exists(self) -> None:
        assert analyze_code is not None
        assert analyze_code.name == "analyze_code"
        assert "Analyze code" in analyze_code.description


class TestCreateCodeTools:
    def test_returns_three_tools(self) -> None:
        tools = create_code_tools()
        assert len(tools) == 3

    def test_includes_analyze_code(self) -> None:
        tools = create_code_tools()
        tool_names = [t.name for t in tools]
        assert "analyze_code" in tool_names

    def test_includes_format_code(self) -> None:
        tools = create_code_tools()
        tool_names = [t.name for t in tools]
        assert "format_code" in tool_names

    def test_includes_execute_python_safe(self) -> None:
        tools = create_code_tools()
        tool_names = [t.name for t in tools]
        assert "execute_python_safe" in tool_names


class TestE2BExecutor:
    def test_execute_without_api_key(self) -> None:
        from src.tools.code import _execute_python_safe

        with patch("src.config.settings") as mock_settings:
            mock_settings.e2b_api_key = None

            result = _execute_python_safe("print('hello')")

            assert "E2B_API_KEY not set" in result
            assert "print('hello')" in result

    def test_execute_with_api_key_but_no_e2b_installed(self) -> None:
        from src.tools.code import E2BExecutor

        executor = E2BExecutor()

        with patch.object(executor, "_get_sandbox", side_effect=ImportError("e2b not installed")):
            result = executor.execute("print('hello')")
            assert "E2B not available" in result

    def test_executor_close(self) -> None:
        from src.tools.code import E2BExecutor

        executor = E2BExecutor()
        executor._sandbox = MagicMock()

        executor.close()

        assert executor._sandbox is None

    def test_executor_close_without_sandbox(self) -> None:
        from src.tools.code import E2BExecutor

        executor = E2BExecutor()
        executor._sandbox = None

        executor.close()

        assert executor._sandbox is None
