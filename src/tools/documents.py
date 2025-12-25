import asyncio
import base64
import csv
import json
import logging
import os
import re
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from jinja2 import BaseLoader, Environment, TemplateNotFound
from langchain_core.tools import BaseTool, StructuredTool

logger: Final = logging.getLogger(__name__)

OUTPUT_DIR: Final = Path(os.getenv("DOCUMENT_OUTPUT_DIR", "./data/documents"))

_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="doc_gen")
    return _executor


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


BUILTIN_TEMPLATES: dict[str, str] = {
    "invoice": """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @page { size: A4; margin: 2cm; }
        body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 12px; color: #333; }
        .header { display: flex; justify-content: space-between; margin-bottom: 30px; }
        .company-info { text-align: right; }
        .invoice-title { font-size: 28px; font-weight: bold; color: #2563eb; margin-bottom: 5px; }
        .invoice-number { color: #666; }
        .addresses { display: flex; justify-content: space-between; margin-bottom: 30px; }
        .address-block { width: 45%; }
        .address-block h3 { font-size: 11px; text-transform: uppercase; color: #666; margin-bottom: 8px; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 30px; }
        th { background: #f8fafc; padding: 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #e2e8f0; }
        td { padding: 12px; border-bottom: 1px solid #e2e8f0; }
        .amount { text-align: right; }
        .totals { margin-left: auto; width: 300px; }
        .totals table { margin-bottom: 0; }
        .totals td { border: none; padding: 8px 12px; }
        .totals .total-row { font-weight: bold; font-size: 16px; background: #f8fafc; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; font-size: 10px; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <div class="invoice-title">INVOICE</div>
            <div class="invoice-number">#{{ invoice_number | default('INV-' ~ (range(10000, 99999) | random)) }}</div>
        </div>
        <div class="company-info">
            <strong>{{ company_name | default('Your Company') }}</strong><br>
            {{ company_address | default('123 Business Street') }}<br>
            {{ company_city | default('City, Country') }}<br>
            {{ company_email | default('billing@company.com') }}
        </div>
    </div>

    <div class="addresses">
        <div class="address-block">
            <h3>Bill To</h3>
            <strong>{{ client_name | default('Client Name') }}</strong><br>
            {{ client_address | default('Client Address') }}<br>
            {{ client_city | default('Client City') }}<br>
            {{ client_email | default('client@example.com') }}
        </div>
        <div class="address-block">
            <h3>Invoice Details</h3>
            <strong>Date:</strong> {{ invoice_date | default(now().strftime('%Y-%m-%d')) }}<br>
            <strong>Due Date:</strong> {{ due_date | default('Upon Receipt') }}<br>
            <strong>Payment Terms:</strong> {{ payment_terms | default('Net 30') }}
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Description</th>
                <th>Quantity</th>
                <th>Unit Price</th>
                <th class="amount">Amount</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items | default([{'description': 'Service', 'quantity': 1, 'unit_price': 100}]) %}
            <tr>
                <td>{{ item.description }}</td>
                <td>{{ item.quantity }}</td>
                <td>{{ currency | default('€') }}{{ "%.2f" | format(item.unit_price) }}</td>
                <td class="amount">{{ currency | d('€') }}{{ "%.2f" | format(item.quantity * item.unit_price) }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <div class="totals">
        <table>
            <tr>
                <td>Subtotal</td>
                <td class="amount">{{ currency | default('€') }}{{ "%.2f" | format(subtotal | default(100)) }}</td>
            </tr>
            {% if tax_rate | default(0) > 0 %}
            <tr>
                <td>Tax ({{ tax_rate }}%)</td>
                <td class="amount">{{ currency | default('€') }}{{ "%.2f" | format(tax_amount | default(0)) }}</td>
            </tr>
            {% endif %}
            <tr class="total-row">
                <td>Total</td>
                <td class="amount">{{ currency | d('€') }}{{ "%.2f" | format(total | d(subtotal | d(100))) }}</td>
            </tr>
        </table>
    </div>

    <div class="footer">
        {{ footer_note | default('Thank you for your business!') }}<br>
        {{ payment_instructions | default('Please make payment within the specified terms.') }}
    </div>
</body>
</html>
""",
    "report": """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @page { size: A4; margin: 2.5cm; @bottom-center { content: "Page " counter(page) " of " counter(pages); } }
        body { font-family: 'Georgia', serif; font-size: 11pt; line-height: 1.6; color: #1a1a1a; }
        h1 { font-size: 24pt; color: #1e40af; margin-bottom: .5em; border-bottom: 2px solid #1e40af; }
        h2 { font-size: 16pt; color: #1e40af; margin-top: 1.5em; }
        h3 { font-size: 13pt; color: #374151; }
        .meta { color: #666; font-size: 10pt; margin-bottom: 2em; }
        .summary { background: #f0f9ff; padding: 1em; border-left: 4px solid #1e40af; margin: 1.5em 0; }
        table { width: 100%; border-collapse: collapse; margin: 1em 0; font-size: 10pt; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }
        th { background: #f9fafb; font-weight: 600; }
        .highlight { background: #fef3c7; padding: 2px 4px; }
        .footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid #e5e7eb; font-size: 9pt; color: #666; }
    </style>
</head>
<body>
    <h1>{{ title | default('Report') }}</h1>
    <div class="meta">
        <strong>Date:</strong> {{ date | default(now().strftime('%Y-%m-%d')) }} |
        <strong>Author:</strong> {{ author | default('AI Agent') }} |
        <strong>Department:</strong> {{ department | default('General') }}
    </div>

    {% if summary %}
    <div class="summary">
        <strong>Executive Summary</strong><br>
        {{ summary }}
    </div>
    {% endif %}

    {% for section in sections | default([{'title': 'Overview', 'content': 'Report content goes here.'}]) %}
    <h2>{{ section.title }}</h2>
    <p>{{ section.content }}</p>

    {% if section.data %}
    <table>
        <thead>
            <tr>
                {% for key in section.data[0].keys() %}
                <th>{{ key | title | replace('_', ' ') }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in section.data %}
            <tr>
                {% for value in row.values() %}
                <td>{{ value }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% endif %}
    {% endfor %}

    {% if conclusions %}
    <h2>Conclusions</h2>
    <p>{{ conclusions }}</p>
    {% endif %}

    <div class="footer">
        Generated on {{ now().strftime('%Y-%m-%d %H:%M') }} | {{ footer | default('Confidential') }}
    </div>
</body>
</html>
""",
    "letter": """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @page { size: A4; margin: 2.5cm; }
        body { font-family: 'Times New Roman', serif; font-size: 12pt; line-height: 1.5; }
        .sender { text-align: right; margin-bottom: 2em; }
        .date { margin-bottom: 2em; }
        .recipient { margin-bottom: 2em; }
        .subject { font-weight: bold; margin-bottom: 1.5em; }
        .salutation { margin-bottom: 1em; }
        .body { text-align: justify; }
        .body p { margin-bottom: 1em; }
        .closing { margin-top: 2em; }
        .signature { margin-top: 3em; }
    </style>
</head>
<body>
    <div class="sender">
        {{ sender_name | default('Your Name') }}<br>
        {{ sender_address | default('Your Address') }}<br>
        {{ sender_city | default('City, Country') }}<br>
        {{ sender_email | default('email@example.com') }}
    </div>

    <div class="date">{{ date | default(now().strftime('%B %d, %Y')) }}</div>

    <div class="recipient">
        {{ recipient_name | default('Recipient Name') }}<br>
        {{ recipient_title | default('') }}{% if recipient_title %}<br>{% endif %}
        {{ recipient_company | default('') }}{% if recipient_company %}<br>{% endif %}
        {{ recipient_address | default('Recipient Address') }}<br>
        {{ recipient_city | default('City, Country') }}
    </div>

    {% if subject %}
    <div class="subject">Re: {{ subject }}</div>
    {% endif %}

    <div class="salutation">{{ salutation | default('Dear ' ~ (recipient_name | default('Sir/Madam'))) }},</div>

    <div class="body">
        {% for paragraph in body | default(['Letter content goes here.']) %}
        <p>{{ paragraph }}</p>
        {% endfor %}
    </div>

    <div class="closing">{{ closing | default('Sincerely') }},</div>

    <div class="signature">
        {{ sender_name | default('Your Name') }}<br>
        {{ sender_title | default('') }}
    </div>
</body>
</html>
""",
    "table": """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @page { size: A4 landscape; margin: 1.5cm; }
        body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 10pt; }
        h1 { font-size: 18pt; color: #1f2937; margin-bottom: 0.5em; }
        .meta { color: #666; font-size: 9pt; margin-bottom: 1em; }
        table { width: 100%; border-collapse: collapse; }
        th { background: #1f2937; color: white; padding: 10px 8px; text-align: left; font-weight: 500; }
        td { padding: 8px; border-bottom: 1px solid #e5e7eb; }
        tr:nth-child(even) { background: #f9fafb; }
        tr:hover { background: #f3f4f6; }
        .number { text-align: right; font-family: 'Monaco', monospace; }
        .footer { margin-top: 1em; font-size: 8pt; color: #666; }
    </style>
</head>
<body>
    <h1>{{ title | default('Data Export') }}</h1>
    <div class="meta">Generated: {{ now().strftime('%Y-%m-%d %H:%M') }} | Records: {{ data | d([]) | length }}</div>

    {% if data %}
    <table>
        <thead>
            <tr>
                {% for key in columns | default(data[0].keys()) %}
                <th>{{ key | title | replace('_', ' ') }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in data %}
            <tr>
                {% for col in columns | default(data[0].keys()) %}
                <td{% if row[col] is number %} class="number"{% endif %}>{{ row[col] }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p>No data available.</p>
    {% endif %}

    <div class="footer">{{ footer | default('') }}</div>
</body>
</html>
""",
}


class TemplateLoader(BaseLoader):
    def get_source(self, environment: Environment, template: str) -> tuple[str, str | None, Callable[[], bool] | None]:
        if template in BUILTIN_TEMPLATES:
            source = BUILTIN_TEMPLATES[template]
            return source, template, lambda: True

        template_path = Path(template)
        if template_path.exists():
            source = template_path.read_text(encoding="utf-8")
            mtime = template_path.stat().st_mtime
            return source, str(template_path), lambda: template_path.stat().st_mtime == mtime

        templates_dir = Path(os.getenv("DOCUMENT_TEMPLATES_DIR", "./data/templates"))
        if templates_dir.exists():
            for ext in [".html", ".jinja2", ".j2", ""]:
                candidate = templates_dir / f"{template}{ext}"
                if candidate.exists():
                    source = candidate.read_text(encoding="utf-8")
                    mtime = candidate.stat().st_mtime
                    path = candidate
                    return source, str(path), lambda m=mtime, p=path: p.stat().st_mtime == m

        raise TemplateNotFound(template)


_jinja_env: Environment | None = None


def _get_jinja_env() -> Environment:
    global _jinja_env
    if _jinja_env is None:
        _jinja_env = Environment(
            loader=TemplateLoader(),
            autoescape=True,
        )
        _jinja_env.globals["now"] = lambda: datetime.now(UTC)
    return _jinja_env


def _render_html_template(template: str, data: dict[str, Any]) -> str:
    env = _get_jinja_env()

    if template.strip().startswith("<") or "<html" in template.lower():
        jinja_template = env.from_string(template)
    else:
        jinja_template = env.get_template(template)

    result: str = jinja_template.render(**data)
    return result


def _generate_pdf_sync(
    template: str,
    data: dict[str, Any],
    filename: str | None = None,
) -> dict[str, Any]:
    try:
        from weasyprint import HTML
    except ImportError as e:
        raise ImportError(
            "WeasyPrint not installed. Run: pip install weasyprint\n"
            "Note: WeasyPrint requires system dependencies. See https://doc.courtbouillon.org/weasyprint/stable/first_steps.html"
        ) from e

    html_content = _render_html_template(template, data)
    html_doc = HTML(string=html_content)
    pdf_bytes = html_doc.write_pdf()

    if not filename:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"document_{timestamp}_{unique_id}.pdf"

    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    output_dir = _ensure_output_dir()
    file_path = output_dir / filename
    file_path.write_bytes(pdf_bytes)

    return {
        "file_path": str(file_path.absolute()),
        "filename": filename,
        "size_bytes": len(pdf_bytes),
        "base64": base64.b64encode(pdf_bytes).decode("utf-8"),
    }


async def _generate_pdf_async(
    template: str,
    data: dict[str, Any],
    filename: str | None = None,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _get_executor(),
        _generate_pdf_sync,
        template,
        data,
        filename,
    )


def _generate_pdf(
    template: str = "report",
    data: str = "{}",
    filename: str | None = None,
) -> str:
    try:
        data_dict = (json.loads(data) if data.strip() else {}) if isinstance(data, str) else data

        result = _generate_pdf_sync(template, data_dict, filename)
        download_url = f"/api/documents/download/{result['filename']}"
        return (
            f"PDF generated successfully:\n"
            f"  File: {result['filename']}\n"
            f"  Size: {result['size_bytes']:,} bytes\n"
            f"  Download: {download_url}"
        )
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON data - {e}"
    except ImportError as e:
        return f"Error: {e}"
    except TemplateNotFound as e:
        available = ", ".join(BUILTIN_TEMPLATES.keys())
        return f"Error: Template '{e.name}' not found. Available templates: {available}"
    except Exception as e:
        logger.exception("PDF generation failed")
        return f"Error generating PDF: {type(e).__name__}: {e}"


def _generate_excel(
    data: str,
    filename: str | None = None,
    sheet_name: str = "Sheet1",
    title: str | None = None,
    include_header: bool = True,
    auto_filter: bool = True,
    freeze_header: bool = True,
) -> str:
    try:
        import xlsxwriter
    except ImportError:
        return "Error: XlsxWriter not installed. Run: pip install xlsxwriter"

    try:
        data_dict = json.loads(data) if isinstance(data, str) else data

        if not filename:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            filename = f"export_{timestamp}_{unique_id}.xlsx"

        if not filename.lower().endswith(".xlsx"):
            filename += ".xlsx"

        output_dir = _ensure_output_dir()
        file_path = output_dir / filename

        workbook = xlsxwriter.Workbook(str(file_path))

        header_format = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#1f2937",
                "font_color": "white",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        )
        title_format = workbook.add_format(
            {
                "bold": True,
                "font_size": 14,
                "align": "left",
            }
        )
        cell_format = workbook.add_format(
            {
                "border": 1,
                "valign": "vcenter",
            }
        )
        number_format = workbook.add_format(
            {
                "border": 1,
                "num_format": "#,##0.00",
                "valign": "vcenter",
            }
        )
        date_format = workbook.add_format(
            {
                "border": 1,
                "num_format": "yyyy-mm-dd",
                "valign": "vcenter",
            }
        )

        def write_sheet(ws: Any, sheet_data: list[dict[str, Any]], ws_title: str | None = None) -> None:
            if not sheet_data:
                ws.write(0, 0, "No data")
                return

            row = 0

            if ws_title:
                ws.write(row, 0, ws_title, title_format)
                row += 2

            columns = list(sheet_data[0].keys())

            if include_header:
                for col, header in enumerate(columns):
                    display_header = header.replace("_", " ").title()
                    ws.write(row, col, display_header, header_format)

                if auto_filter:
                    ws.autofilter(row, 0, row + len(sheet_data), len(columns) - 1)

                if freeze_header:
                    ws.freeze_panes(row + 1, 0)

                row += 1

            for record in sheet_data:
                for col, key in enumerate(columns):
                    value = record.get(key, "")

                    if isinstance(value, (int, float)):
                        ws.write_number(row, col, value, number_format)
                    elif isinstance(value, bool):
                        ws.write_boolean(row, col, value, cell_format)
                    else:
                        if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}", value):
                            try:
                                from datetime import datetime as dt

                                date_val = dt.fromisoformat(value.replace("Z", "+00:00"))
                                ws.write_datetime(row, col, date_val, date_format)
                            except (ValueError, TypeError):
                                ws.write_string(row, col, str(value), cell_format)
                        else:
                            ws.write_string(row, col, str(value) if value is not None else "", cell_format)
                row += 1

            for col, key in enumerate(columns):
                max_len = len(key.replace("_", " ").title())
                for record in sheet_data[:100]:
                    val = str(record.get(key, ""))
                    max_len = max(max_len, min(len(val), 50))
                ws.set_column(col, col, max_len + 2)

        if isinstance(data_dict, dict) and "sheets" in data_dict:
            for ws_name, ws_data in data_dict["sheets"].items():
                worksheet = workbook.add_worksheet(ws_name[:31])
                write_sheet(worksheet, ws_data)
        elif isinstance(data_dict, list):
            worksheet = workbook.add_worksheet(sheet_name[:31])
            write_sheet(worksheet, data_dict, title)
        else:
            worksheet = workbook.add_worksheet(sheet_name[:31])
            write_sheet(worksheet, [data_dict], title)

        workbook.close()

        file_size = file_path.stat().st_size
        download_url = f"/api/documents/download/{filename}"
        return (
            f"Excel file generated successfully:\n"
            f"  File: {filename}\n"
            f"  Size: {file_size:,} bytes\n"
            f"  Download: {download_url}"
        )

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON data - {e}"
    except Exception as e:
        logger.exception("Excel generation failed")
        return f"Error generating Excel: {type(e).__name__}: {e}"


def _generate_csv(
    data: str,
    filename: str | None = None,
    delimiter: str = ",",
    include_header: bool = True,
) -> str:
    try:
        data_list = json.loads(data) if isinstance(data, str) else data

        if not isinstance(data_list, list):
            data_list = [data_list]

        if not data_list:
            return "Error: No data provided"

        if not filename:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            filename = f"export_{timestamp}_{unique_id}.csv"

        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        output_dir = _ensure_output_dir()
        file_path = output_dir / filename

        columns: list[str] = []
        for record in data_list:
            if isinstance(record, dict):
                for key in record:
                    if key not in columns:
                        columns.append(key)

        with file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter=delimiter, extrasaction="ignore")

            if include_header:
                writer.writeheader()

            for record in data_list:
                if isinstance(record, dict):
                    row = {k: str(v) if v is not None else "" for k, v in record.items()}
                    writer.writerow(row)

        file_size = file_path.stat().st_size
        row_count = len(data_list)
        download_url = f"/api/documents/download/{filename}"
        return (
            f"CSV file generated successfully:\n"
            f"  File: {filename}\n"
            f"  Size: {file_size:,} bytes\n"
            f"  Rows: {row_count:,}\n"
            f"  Download: {download_url}"
        )

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON data - {e}"
    except Exception as e:
        logger.exception("CSV generation failed")
        return f"Error generating CSV: {type(e).__name__}: {e}"


def _list_templates() -> str:
    result = ["Available Document Templates:", "=" * 40, ""]

    result.append("Built-in Templates:")
    for name in BUILTIN_TEMPLATES:
        descriptions = {
            "invoice": "Professional invoice with line items, tax calculation, and company branding",
            "report": "Business report with sections, summary, and data tables",
            "letter": "Formal letter format with sender/recipient addresses",
            "table": "Simple data table export in landscape orientation",
        }
        result.append(f"  - {name}: {descriptions.get(name, 'Document template')}")

    templates_dir = Path(os.getenv("DOCUMENT_TEMPLATES_DIR", "./data/templates"))
    if templates_dir.exists():
        custom_templates = list(templates_dir.glob("*.html")) + list(templates_dir.glob("*.jinja2"))
        if custom_templates:
            result.append("")
            result.append("Custom Templates:")
            for template_path in custom_templates:
                result.append(f"  - {template_path.stem}")

    result.append("")
    result.append("Output Directory: " + str(OUTPUT_DIR.absolute()))

    return "\n".join(result)


generate_pdf = StructuredTool.from_function(
    func=_generate_pdf,
    name="generate_pdf",
    description=(
        "Generate a PDF document from a template. "
        "Templates: 'invoice' (line items, totals), 'report' (sections, data tables), "
        "'letter' (formal correspondence), 'table' (data export). "
        "Provide data as JSON string. Returns file path of generated PDF."
    ),
)

generate_excel = StructuredTool.from_function(
    func=_generate_excel,
    name="generate_excel",
    description=(
        "Generate an Excel (.xlsx) file from JSON data. "
        "Data should be an array of objects (rows) or {sheets: {SheetName: [rows]}} for multi-sheet. "
        "Includes formatting, auto-filter, and frozen headers. Returns file path."
    ),
)

generate_csv = StructuredTool.from_function(
    func=_generate_csv,
    name="generate_csv",
    description=(
        "Generate a CSV file from JSON data. "
        "Data should be an array of objects. Simple and fast for data export. "
        "Returns file path of generated CSV."
    ),
)

list_templates = StructuredTool.from_function(
    func=_list_templates,
    name="list_document_templates",
    description="List available document templates for PDF generation.",
)


class GeneratePDFTool(BaseTool):
    name: str = "generate_pdf"
    description: str = (
        "Generate a PDF document from a template and data. "
        "Use templates: 'invoice', 'report', 'letter', 'table', or provide custom HTML. "
        "Data should be a JSON string with template variables."
    )

    def _run(self, template: str = "report", data: str = "{}", filename: str | None = None) -> str:
        return _generate_pdf(template, data, filename)

    async def _arun(self, template: str = "report", data: str = "{}", filename: str | None = None) -> str:
        try:
            data_dict = json.loads(data) if isinstance(data, str) and data.strip() else {}
            result = await _generate_pdf_async(template, data_dict, filename)
            download_url = f"/api/documents/download/{result['filename']}"
            return (
                f"PDF generated successfully:\n"
                f"  File: {result['filename']}\n"
                f"  Size: {result['size_bytes']:,} bytes\n"
                f"  Download: {download_url}"
            )
        except Exception as e:
            logger.exception("Async PDF generation failed")
            return f"Error: {type(e).__name__}: {e}"


class GenerateExcelTool(BaseTool):
    name: str = "generate_excel"
    description: str = (
        "Generate an Excel (.xlsx) file from JSON data. "
        "Supports single sheet (array of objects) or multi-sheet ({sheets: {...}}). "
        "Includes professional formatting, auto-filter, and frozen headers."
    )

    def _run(self, data: str, filename: str | None = None, sheet_name: str = "Sheet1") -> str:
        return _generate_excel(data, filename, sheet_name)

    async def _arun(self, data: str, filename: str | None = None, sheet_name: str = "Sheet1") -> str:
        return self._run(data, filename, sheet_name)


class GenerateCSVTool(BaseTool):
    name: str = "generate_csv"
    description: str = (
        "Generate a CSV file from JSON data (array of objects). "
        "Fast and simple for data export. Compatible with all spreadsheet applications."
    )

    def _run(self, data: str, filename: str | None = None, delimiter: str = ",") -> str:
        return _generate_csv(data, filename, delimiter)

    async def _arun(self, data: str, filename: str | None = None, delimiter: str = ",") -> str:
        return self._run(data, filename, delimiter)


def create_document_tools() -> list[BaseTool]:
    return [
        GeneratePDFTool(),
        GenerateExcelTool(),
        GenerateCSVTool(),
        list_templates,
    ]
