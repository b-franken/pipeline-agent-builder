import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "documents"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_output_dir(temp_output_dir: Path):
    with patch.dict(os.environ, {"DOCUMENT_OUTPUT_DIR": str(temp_output_dir)}):
        from src.tools import documents

        documents.OUTPUT_DIR = temp_output_dir
        documents._ensure_output_dir()
        yield temp_output_dir


class TestTemplateSystem:
    def test_builtin_templates_exist(self):
        from src.tools.documents import BUILTIN_TEMPLATES

        assert "invoice" in BUILTIN_TEMPLATES
        assert "report" in BUILTIN_TEMPLATES
        assert "letter" in BUILTIN_TEMPLATES
        assert "table" in BUILTIN_TEMPLATES

    def test_template_loader_builtin(self):
        from src.tools.documents import _get_jinja_env

        env = _get_jinja_env()
        template = env.get_template("invoice")
        assert template is not None

    def test_template_loader_raw_html(self):
        from src.tools.documents import _render_html_template

        raw_html = "<html><body>{{ name }}</body></html>"
        result = _render_html_template(raw_html, {"name": "Test"})
        assert "Test" in result

    def test_template_not_found(self):
        from jinja2 import TemplateNotFound

        from src.tools.documents import _get_jinja_env

        env = _get_jinja_env()
        with pytest.raises(TemplateNotFound):
            env.get_template("nonexistent_template_xyz")

    def test_list_templates(self):
        from src.tools.documents import _list_templates

        result = _list_templates()
        assert "invoice" in result
        assert "report" in result
        assert "letter" in result
        assert "table" in result


class TestCSVGeneration:
    def test_generate_csv_basic(self, mock_output_dir: Path):
        from src.tools.documents import _generate_csv

        data = json.dumps([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])

        result = _generate_csv(data)
        assert "CSV file generated successfully" in result
        assert "Rows: 2" in result

        csv_files = list(mock_output_dir.glob("*.csv"))
        assert len(csv_files) == 1

        content = csv_files[0].read_text()
        assert "name,age" in content
        assert "Alice,30" in content
        assert "Bob,25" in content

    def test_generate_csv_custom_filename(self, mock_output_dir: Path):
        from src.tools.documents import _generate_csv

        data = json.dumps([{"col1": "val1"}])
        result = _generate_csv(data, filename="custom_export.csv")

        assert "custom_export.csv" in result
        assert (mock_output_dir / "custom_export.csv").exists()

    def test_generate_csv_custom_delimiter(self, mock_output_dir: Path):
        from src.tools.documents import _generate_csv

        data = json.dumps([{"a": "1", "b": "2"}])
        _generate_csv(data, delimiter=";")

        csv_files = list(mock_output_dir.glob("*.csv"))
        content = csv_files[0].read_text()
        assert "a;b" in content

    def test_generate_csv_no_header(self, mock_output_dir: Path):
        from src.tools.documents import _generate_csv

        data = json.dumps([{"x": "1", "y": "2"}])
        _generate_csv(data, include_header=False)

        csv_files = list(mock_output_dir.glob("*.csv"))
        content = csv_files[0].read_text()
        assert "x,y" not in content
        assert "1,2" in content

    def test_generate_csv_empty_data(self, mock_output_dir: Path):
        from src.tools.documents import _generate_csv

        result = _generate_csv("[]")
        assert "Error: No data provided" in result

    def test_generate_csv_invalid_json(self, mock_output_dir: Path):
        from src.tools.documents import _generate_csv

        result = _generate_csv("not valid json")
        assert "Error: Invalid JSON" in result

    def test_generate_csv_single_object(self, mock_output_dir: Path):
        from src.tools.documents import _generate_csv

        data = json.dumps({"single": "object"})
        result = _generate_csv(data)
        assert "CSV file generated successfully" in result
        assert "Rows: 1" in result


class TestExcelGeneration:
    def test_generate_excel_basic(self, mock_output_dir: Path):
        pytest.importorskip("xlsxwriter")
        from src.tools.documents import _generate_excel

        data = json.dumps([{"name": "Alice", "salary": 50000}, {"name": "Bob", "salary": 60000}])

        result = _generate_excel(data)
        assert "Excel file generated successfully" in result

        xlsx_files = list(mock_output_dir.glob("*.xlsx"))
        assert len(xlsx_files) == 1

    def test_generate_excel_custom_filename(self, mock_output_dir: Path):
        pytest.importorskip("xlsxwriter")
        from src.tools.documents import _generate_excel

        data = json.dumps([{"col": "val"}])
        result = _generate_excel(data, filename="my_report.xlsx")

        assert "my_report.xlsx" in result
        assert (mock_output_dir / "my_report.xlsx").exists()

    def test_generate_excel_multi_sheet(self, mock_output_dir: Path):
        pytest.importorskip("xlsxwriter")
        from src.tools.documents import _generate_excel

        data = json.dumps(
            {
                "sheets": {
                    "Employees": [{"name": "John", "dept": "IT"}],
                    "Departments": [{"name": "IT", "budget": 100000}],
                }
            }
        )

        result = _generate_excel(data)
        assert "Excel file generated successfully" in result

    def test_generate_excel_invalid_json(self, mock_output_dir: Path):
        pytest.importorskip("xlsxwriter")
        from src.tools.documents import _generate_excel

        result = _generate_excel("invalid json {")
        assert "Error: Invalid JSON" in result

    def test_generate_excel_adds_extension(self, mock_output_dir: Path):
        pytest.importorskip("xlsxwriter")
        from src.tools.documents import _generate_excel

        data = json.dumps([{"a": 1}])
        result = _generate_excel(data, filename="no_extension")
        assert ".xlsx" in result


class TestPDFGeneration:
    def test_generate_pdf_report_template(self, mock_output_dir: Path):
        pytest.importorskip("weasyprint")
        from src.tools.documents import _generate_pdf

        data = json.dumps(
            {
                "title": "Test Report",
                "author": "Test Author",
                "summary": "This is a test summary.",
                "sections": [{"title": "Section 1", "content": "Content for section 1."}],
            }
        )

        result = _generate_pdf("report", data)
        assert "PDF generated successfully" in result

        pdf_files = list(mock_output_dir.glob("*.pdf"))
        assert len(pdf_files) == 1

    def test_generate_pdf_invoice_template(self, mock_output_dir: Path):
        pytest.importorskip("weasyprint")
        from src.tools.documents import _generate_pdf

        data = json.dumps(
            {
                "company_name": "Acme Inc",
                "client_name": "John Doe",
                "items": [{"description": "Consulting", "quantity": 10, "unit_price": 150}],
                "subtotal": 1500,
                "total": 1500,
            }
        )

        result = _generate_pdf("invoice", data)
        assert "PDF generated successfully" in result

    def test_generate_pdf_letter_template(self, mock_output_dir: Path):
        pytest.importorskip("weasyprint")
        from src.tools.documents import _generate_pdf

        data = json.dumps(
            {
                "sender_name": "Jane Smith",
                "recipient_name": "Bob Wilson",
                "subject": "Meeting Request",
                "body": ["First paragraph.", "Second paragraph."],
            }
        )

        result = _generate_pdf("letter", data)
        assert "PDF generated successfully" in result

    def test_generate_pdf_table_template(self, mock_output_dir: Path):
        pytest.importorskip("weasyprint")
        from src.tools.documents import _generate_pdf

        data = json.dumps(
            {
                "title": "Sales Data",
                "data": [
                    {"product": "Widget A", "quantity": 100, "revenue": 5000},
                    {"product": "Widget B", "quantity": 50, "revenue": 3000},
                ],
            }
        )

        result = _generate_pdf("table", data)
        assert "PDF generated successfully" in result

    def test_generate_pdf_custom_filename(self, mock_output_dir: Path):
        pytest.importorskip("weasyprint")
        from src.tools.documents import _generate_pdf

        result = _generate_pdf("report", "{}", "my_report.pdf")
        assert "my_report.pdf" in result
        assert (mock_output_dir / "my_report.pdf").exists()

    def test_generate_pdf_invalid_template(self, mock_output_dir: Path):
        pytest.importorskip("weasyprint")
        from src.tools.documents import _generate_pdf

        result = _generate_pdf("nonexistent_template", "{}")
        assert "Error: Template" in result
        assert "not found" in result

    def test_generate_pdf_invalid_json(self, mock_output_dir: Path):
        pytest.importorskip("weasyprint")
        from src.tools.documents import _generate_pdf

        result = _generate_pdf("report", "invalid json")
        assert "Error: Invalid JSON" in result

    def test_generate_pdf_raw_html(self, mock_output_dir: Path):
        pytest.importorskip("weasyprint")
        from src.tools.documents import _generate_pdf

        raw_html = "<html><head></head><body><h1>{{ title }}</h1></body></html>"
        data = json.dumps({"title": "Custom Document"})

        result = _generate_pdf(raw_html, data)
        assert "PDF generated successfully" in result


class TestToolClasses:
    def test_generate_pdf_tool_sync(self, mock_output_dir: Path):
        pytest.importorskip("weasyprint")
        from src.tools.documents import GeneratePDFTool

        tool = GeneratePDFTool()
        result = tool._run("report", "{}")
        assert "PDF generated successfully" in result or "Error" in result

    def test_generate_excel_tool_sync(self, mock_output_dir: Path):
        pytest.importorskip("xlsxwriter")
        from src.tools.documents import GenerateExcelTool

        tool = GenerateExcelTool()
        result = tool._run('[{"a": 1}]')
        assert "Excel file generated successfully" in result

    def test_generate_csv_tool_sync(self, mock_output_dir: Path):
        from src.tools.documents import GenerateCSVTool

        tool = GenerateCSVTool()
        result = tool._run('[{"x": 1}]')
        assert "CSV file generated successfully" in result


class TestCreateDocumentTools:
    def test_create_document_tools_returns_list(self):
        from src.tools.documents import create_document_tools

        tools = create_document_tools()
        assert isinstance(tools, list)
        assert len(tools) == 4

    def test_create_document_tools_names(self):
        from src.tools.documents import create_document_tools

        tools = create_document_tools()
        names = [t.name for t in tools]

        assert "generate_pdf" in names
        assert "generate_excel" in names
        assert "generate_csv" in names
        assert "list_document_templates" in names


class TestToolRegistration:
    def test_tools_registered_in_registry(self) -> None:
        from unittest.mock import MagicMock, patch

        from src.tools.factory import create_tools_for_agent

        with (
            patch("src.tools.factory._create_generate_pdf") as mock_pdf,
            patch("src.tools.factory._create_generate_excel") as mock_excel,
            patch("src.tools.factory._create_generate_csv") as mock_csv,
            patch("src.tools.factory._create_list_document_templates") as mock_list,
        ):
            mock_pdf.return_value = MagicMock(name="generate_pdf")
            mock_excel.return_value = MagicMock(name="generate_excel")
            mock_csv.return_value = MagicMock(name="generate_csv")
            mock_list.return_value = MagicMock(name="list_document_templates")

            tools = create_tools_for_agent(
                ["generate_pdf", "generate_excel", "generate_csv", "list_document_templates"]
            )

            assert len(tools) == 4
            mock_pdf.assert_called_once()
            mock_excel.assert_called_once()
            mock_csv.assert_called_once()
            mock_list.assert_called_once()

    def test_document_plugin_registered(self) -> None:
        from src.tools.plugins import get_plugin_registry

        registry = get_plugin_registry()
        plugin = registry.get_plugin("documents")
        assert plugin is not None
        assert plugin.name == "documents"
