//! File authoring tools: `generate_pdf`, `generate_xlsx`, `generate_csv`,
//! `generate_json`, and `generate_markdown`.
//!
//! This module is the Rust counterpart of the Python `file_authoring.py`
//! module (which used `fpdf2`/`reportlab` and `openpyxl`). It provides:
//!
//! - [`GeneratePdfTool`] — builds a PDF report from structured sections.
//! - [`GenerateXlsxTool`] — builds an XLSX workbook from sheets of rows.
//! - [`GenerateCsvTool`] — builds an RFC 4180 CSV document with an optional
//!   header row.
//! - [`GenerateJsonTool`] — writes a validated, pretty-printed JSON document.
//! - [`GenerateMarkdownTool`] — writes a Markdown document.
//!
//! All tools write into the configured workspace directory with path
//! traversal protection, run heavy generation and I/O off the async runtime
//! via [`tokio::task::spawn_blocking`], and return a [`ToolOutput`] carrying
//! the absolute file path plus metadata.

use crate::registry::{ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// MIME type for generated PDF files.
pub const MIME_PDF: &str = "application/pdf";
/// MIME type for generated XLSX workbooks.
pub const MIME_XLSX: &str =
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
/// MIME type for generated CSV documents.
pub const MIME_CSV: &str = "text/csv";
/// MIME type for generated JSON documents.
pub const MIME_JSON: &str = "application/json";
/// MIME type for generated Markdown documents.
pub const MIME_MARKDOWN: &str = "text/markdown";

/// US Letter page width in millimetres.
const DEFAULT_PAGE_WIDTH_MM: f64 = 215.9;
/// US Letter page height in millimetres.
const DEFAULT_PAGE_HEIGHT_MM: f64 = 279.4;
/// Default maximum generated-file size (20 MiB).
const DEFAULT_MAX_FILE_SIZE: u64 = 20 * 1024 * 1024;

/// Resolve and validate an output path against the workspace base directory.
///
/// Returns a [`ToolError::PATH_TRAVERSAL`] error when the requested path would
/// escape the base directory.
fn resolve_output_path(allowed_base: &Path, path_str: &str) -> ToolResult<PathBuf> {
    let path = PathBuf::from(path_str);
    let resolved = if path.is_relative() {
        allowed_base.join(&path)
    } else {
        path
    };

    if let Some(parent) = resolved.parent() {
        let parent_canonical = parent.canonicalize().map_err(|_| {
            ToolError::new(
                "PATH_INVALID",
                format!("Cannot access parent directory of '{}'", path_str),
            )
        })?;
        let full_path = parent_canonical.join(resolved.file_name().unwrap_or_default());
        if !full_path.starts_with(allowed_base) {
            return Err(ToolError::new(
                "PATH_TRAVERSAL",
                format!("Path '{}' is outside the workspace", path_str),
            ));
        }
        return Ok(full_path);
    }

    Err(ToolError::new(
        "PATH_INVALID",
        format!("Invalid path '{}'", path_str),
    ))
}

/// Ensure a filename carries the requested extension (e.g. `.pdf`).
fn ensure_extension(name: &str, ext: &str) -> String {
    if name.to_lowercase().ends_with(ext) {
        name.to_string()
    } else {
        format!("{name}{ext}")
    }
}

/// Convert an arbitrary JSON cell value into a CSV-friendly scalar string.
fn stringify_cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Escape a CSV field per RFC 4180.
fn escape_csv_field(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// Build a CSV document (RFC 4180) from an array of row arrays.
fn build_csv(rows: &Value, headers: Option<&[String]>) -> ToolResult<String> {
    let array = rows.as_array().ok_or_else(|| {
        ToolError::invalid_args("'rows' must be a non-empty array of row arrays")
    })?;
    if array.is_empty() {
        return Err(ToolError::invalid_args("'rows' must not be empty"));
    }

    let mut out = String::new();
    if let Some(headers) = headers {
        let line: Vec<String> = headers.iter().map(|h| escape_csv_field(h)).collect();
        out.push_str(&line.join(","));
        out.push('\n');
    }

    for (idx, row) in array.iter().enumerate() {
        let cells = row.as_array().ok_or_else(|| {
            ToolError::invalid_args(format!("rows[{idx}] must be an array of cells"))
        })?;
        let line: Vec<String> = cells
            .iter()
            .map(stringify_cell)
            .map(|c| escape_csv_field(&c))
            .collect();
        out.push_str(&line.join(","));
        out.push('\n');
    }
    Ok(out)
}

/// Sanitize an XLSX worksheet title: strip forbidden characters and cap at 31
/// characters (the Excel worksheet-title limit).
fn sanitize_sheet_title(name: &str, fallback_index: usize) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if "[]:*?/\\".contains(c) { '_' } else { c })
        .collect();
    let trimmed = cleaned.trim().trim_matches('\'');
    if trimmed.is_empty() {
        format!("Sheet{fallback_index}")
    } else {
        trimmed.chars().take(31).collect()
    }
}

/// Write an XLSX cell for an arbitrary JSON value.
fn write_xlsx_cell(
    worksheet: &mut rust_xlsxwriter::Worksheet,
    row: u32,
    col: u16,
    cell: &Value,
) -> Result<(), rust_xlsxwriter::XlsxError> {
    match cell {
        Value::String(s) => worksheet.write_string(row, col, s),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                worksheet.write_number(row, col, i as f64)
            } else if let Some(u) = n.as_u64() {
                worksheet.write_number(row, col, u as f64)
            } else {
                worksheet.write_number(row, col, n.as_f64().unwrap_or(0.0))
            }
        }
        Value::Bool(b) => worksheet.write_boolean(row, col, *b),
        Value::Null => worksheet.write_blank(row, col),
        other => worksheet.write_string(row, col, other.to_string()),
    }
}

/// Small shared helper that resolves paths, enforces the size budget, and
/// writes files off the async runtime.
struct FileAuthoringBase {
    allowed_base: PathBuf,
    max_file_size: u64,
}

impl FileAuthoringBase {
    fn new(allowed_base: PathBuf) -> Self {
        Self {
            allowed_base,
            max_file_size: DEFAULT_MAX_FILE_SIZE,
        }
    }

    fn resolve(&self, name: &str) -> ToolResult<PathBuf> {
        resolve_output_path(&self.allowed_base, name)
    }

    /// Write raw bytes to `path`, enforcing the file-size budget. The write
    /// runs on a blocking thread pool via [`tokio::task::spawn_blocking`].
    async fn write_bytes(
        &self,
        path: PathBuf,
        bytes: Vec<u8>,
        kind: &str,
    ) -> ToolResult<serde_json::Value> {
        if bytes.len() as u64 > self.max_file_size {
            return Err(ToolError::new(
                "FILE_TOO_LARGE",
                format!(
                    "Generated {kind} too large: {} bytes (max {})",
                    bytes.len(),
                    self.max_file_size
                ),
            ));
        }
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to create directory: {e}")))?;
        }

        let path_for_task = path.clone();
        let bytes_for_task = bytes.clone();
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            std::fs::write(&path_for_task, &bytes_for_task)
        })
        .await
        .map_err(|e| ToolError::new("IO_ERROR", format!("Write task failed: {e}")))?
        .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to write file: {e}")))?;

        Ok(json!({
            "path": path.to_string_lossy(),
            "size_bytes": bytes.len(),
            "filename": path
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default(),
        }))
    }

    async fn write_text(&self, path: PathBuf, text: String, kind: &str) -> ToolResult<serde_json::Value> {
        self.write_bytes(path, text.into_bytes(), kind).await
    }
}

/// Build a PDF report using `printpdf`.
///
/// Produces a title, optional structured sections (heading + body) or a plain
/// body string, and returns the raw PDF bytes.
fn build_pdf(
    title: &str,
    sections: &Value,
    body: Option<&str>,
    font_path: Option<&Path>,
) -> ToolResult<Vec<u8>> {
    use printpdf::{BuiltinFont, Mm, PdfDocument};

    let (mut doc, page, layer) = PdfDocument::new(
        title,
        Mm(DEFAULT_PAGE_WIDTH_MM),
        Mm(DEFAULT_PAGE_HEIGHT_MM),
        "OpenSquilla",
    );

    let base_font = if let Some(fp) = font_path {
        let font_data = std::fs::read(fp)
            .map_err(|e| ToolError::new("PDF_ERROR", format!("Failed to read font file: {e}")))?;
        doc.add_external_font(&font_data)
            .map_err(|e| ToolError::new("PDF_ERROR", format!("Failed to add external font: {e}")))?
    } else {
        doc.add_builtin_font(BuiltinFont::Helvetica)
            .map_err(|e| ToolError::new("PDF_ERROR", format!("Failed to add builtin font: {e}")))?
    };
    let bold_font = doc
        .add_builtin_font(BuiltinFont::HelveticaBold)
        .map_err(|e| ToolError::new("PDF_ERROR", format!("Failed to add bold font: {e}")))?;

    {
        let current_layer = doc.get_page(page).get_layer(layer);
        let mut y_mm = DEFAULT_PAGE_HEIGHT_MM - 25.0;

        current_layer
            .use_text(title, 20.0, Mm(20.0), Mm(y_mm), &bold_font)
            .map_err(|e| ToolError::new("PDF_ERROR", format!("Failed to render title: {e}")))?;
        y_mm -= 12.0;

        if let Some(section_array) = sections.as_array() {
            for (idx, section) in section_array.iter().enumerate() {
                let obj = section.as_object().ok_or_else(|| {
                    ToolError::invalid_args(format!("sections[{idx}] must be an object"))
                })?;
                let heading = obj
                    .get("heading")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&format!("Section {}", idx + 1));
                let section_body = obj.get("body").and_then(|v| v.as_str()).unwrap_or("");

                y_mm -= 10.0;
                if y_mm < 20.0 {
                    y_mm = DEFAULT_PAGE_HEIGHT_MM - 25.0;
                }
                current_layer
                    .use_text(heading, 14.0, Mm(20.0), Mm(y_mm), &bold_font)
                    .map_err(|e| ToolError::new("PDF_ERROR", format!("Failed to render heading: {e}")))?;
                y_mm -= 8.0;

                for line in section_body.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    if y_mm < 20.0 {
                        y_mm = DEFAULT_PAGE_HEIGHT_MM - 25.0;
                    }
                    current_layer
                        .use_text(line, 10.0, Mm(22.0), Mm(y_mm), &base_font)
                        .map_err(|e| ToolError::new("PDF_ERROR", format!("Failed to render body: {e}")))?;
                    y_mm -= 5.0;
                }
            }
        } else if let Some(body_text) = body {
            for line in body_text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                if y_mm < 20.0 {
                    y_mm = DEFAULT_PAGE_HEIGHT_MM - 25.0;
                }
                current_layer
                    .use_text(line, 10.0, Mm(20.0), Mm(y_mm), &base_font)
                    .map_err(|e| ToolError::new("PDF_ERROR", format!("Failed to render body: {e}")))?;
                y_mm -= 5.0;
            }
        }
    }

    let mut buf = Vec::new();
    doc.save(&mut std::io::Cursor::new(&mut buf))
        .map_err(|e| ToolError::new("PDF_ERROR", format!("Failed to save PDF: {e}")))?;
    Ok(buf)
}

/// Build an XLSX workbook from an array of `{ "name", "rows" }` sheet objects.
fn build_xlsx(sheets: &Value) -> ToolResult<Vec<u8>> {
    use rust_xlsxwriter::{Workbook, XlsxError};

    let sheet_array = sheets.as_array().ok_or_else(|| {
        ToolError::invalid_args("'sheets' must be a non-empty array of sheet objects")
    })?;
    if sheet_array.is_empty() {
        return Err(ToolError::invalid_args("'sheets' must not be empty"));
    }

    let mut workbook = Workbook::new();
    for (idx, sheet_value) in sheet_array.iter().enumerate() {
        let obj = sheet_value.as_object().ok_or_else(|| {
            ToolError::invalid_args(format!("sheets[{idx}] must be an object"))
        })?;
        let title = obj
            .get("name")
            .and_then(|v| v.as_str())
            .map(|n| sanitize_sheet_title(n, idx + 1))
            .unwrap_or_else(|| format!("Sheet{}", idx + 1));
        let rows = obj
            .get("rows")
            .ok_or_else(|| ToolError::invalid_args(format!("sheets[{idx}].rows is required")))?;
        let rows = rows
            .as_array()
            .ok_or_else(|| ToolError::invalid_args(format!("sheets[{idx}].rows must be an array")))?;

        let worksheet = workbook.add_worksheet();
        let _ = worksheet.set_name(&title);
        for (r, row) in rows.iter().enumerate() {
            let cells = row.as_array().ok_or_else(|| {
                ToolError::invalid_args(format!("sheets[{idx}].rows[{r}] must be an array of cells"))
            })?;
            for (c, cell) in cells.iter().enumerate() {
                write_xlsx_cell(worksheet, r as u32, c as u16, cell).map_err(|e| {
                    ToolError::new(
                        "XLSX_ERROR",
                        format!("Failed to write cell ({r},{c}): {e}"),
                    )
                })?;
            }
        }
    }

    let mut buf = Vec::new();
    workbook
        .save_to_buffer(&mut buf)
        .map_err(|e: XlsxError| ToolError::new("XLSX_ERROR", format!("Failed to save workbook: {e}")))?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// generate_pdf
// ---------------------------------------------------------------------------

/// Tool that generates a PDF report from structured sections.
pub struct GeneratePdfTool {
    base: FileAuthoringBase,
}

impl GeneratePdfTool {
    /// Create a new PDF generation tool rooted at the workspace directory.
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self {
            base: FileAuthoringBase::new(workspace_dir),
        }
    }

    /// The JSON Schema for this tool's parameters.
    pub fn input_schema(&self) -> serde_json::Value {
        self.definition().to_json_schema()
    }
}

#[async_trait]
impl Tool for GeneratePdfTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "generate_pdf",
                "Create a simple PDF report from structured text sections and save it to the "
                    + "workspace. Use this for channel PDF requests instead of returning PDF "
                    + "source text.",
                HashMap::from([
                    (
                        "filename".to_string(),
                        ParameterDefinition::required_string(
                            "Output filename. .pdf is appended if missing.",
                        ),
                    ),
                    (
                        "title".to_string(),
                        ParameterDefinition::required_string("Report title."),
                    ),
                    (
                        "sections".to_string(),
                        ParameterDefinition::array(
                            "Optional array of objects with 'heading' and 'body' fields.",
                            ParameterDefinition::string("Section field"),
                        ),
                    ),
                    (
                        "body".to_string(),
                        ParameterDefinition::string("Optional fallback body text."),
                    ),
                    (
                        "font_path".to_string(),
                        ParameterDefinition::string(
                            "Optional path to a TTF font file used for body text.",
                        ),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let filename = params["filename"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'filename' parameter"))?;
        let title = params["title"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'title' parameter"))?;
        let sections = params.get("sections").cloned().unwrap_or(Value::Null);
        let body = params.get("body").and_then(|v| v.as_str()).map(String::from);
        let font_path = params
            .get("font_path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from);

        let file_name = ensure_extension(filename, ".pdf");
        let path = self.base.resolve(&file_name)?;

        let title_for_task = title.to_string();
        let font_path_for_task = font_path;
        let bytes = tokio::task::spawn_blocking(move || {
            Self::build_pdf_impl(&title_for_task, &sections, body.as_deref(), font_path_for_task.as_deref())
        })
        .await
        .map_err(|e| ToolError::new("PDF_ERROR", format!("PDF generation task failed: {e}")))??;

        let data = self.base.write_bytes(path, bytes, "PDF report").await?;
        let size = data["size_bytes"].as_u64().unwrap_or(0);
        Ok(ToolOutput::success(format!(
            "Generated PDF report '{}' ({} bytes)",
            file_name, size
        ))
        .with_mime_type(MIME_PDF)
        .with_data(data))
    }
}

impl GeneratePdfTool {
    /// Helper kept inside an `impl` so the static definition stays minimal.
    fn build_pdf_impl(
        title: &str,
        sections: &Value,
        body: Option<&str>,
        font_path: Option<&Path>,
    ) -> ToolResult<Vec<u8>> {
        build_pdf(title, sections, body, font_path)
    }
}

// ---------------------------------------------------------------------------
// generate_xlsx
// ---------------------------------------------------------------------------

/// Tool that generates an XLSX workbook from structured sheets.
pub struct GenerateXlsxTool {
    base: FileAuthoringBase,
}

impl GenerateXlsxTool {
    /// Create a new XLSX generation tool rooted at the workspace directory.
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self {
            base: FileAuthoringBase::new(workspace_dir),
        }
    }

    /// The JSON Schema for this tool's parameters.
    pub fn input_schema(&self) -> serde_json::Value {
        self.definition().to_json_schema()
    }
}

#[async_trait]
impl Tool for GenerateXlsxTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "generate_xlsx",
                "Create an XLSX workbook from structured sheets and save it to the workspace. "
                    + "Use this for spreadsheet requests from channels.",
                HashMap::from([
                    (
                        "filename".to_string(),
                        ParameterDefinition::required_string(
                            "Output filename. .xlsx is appended if missing.",
                        ),
                    ),
                    (
                        "sheets".to_string(),
                        ParameterDefinition::array(
                            "Non-empty array of objects with optional 'name' and required 'rows'.",
                            ParameterDefinition::string("Sheet object"),
                        )
                        .required(),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let filename = params["filename"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'filename' parameter"))?;
        let sheets = params
            .get("sheets")
            .cloned()
            .ok_or_else(|| ToolError::invalid_args("Missing 'sheets' parameter"))?;

        let file_name = ensure_extension(filename, ".xlsx");
        let path = self.base.resolve(&file_name)?;

        let bytes = tokio::task::spawn_blocking(move || Self::build_xlsx_impl(&sheets))
            .await
            .map_err(|e| ToolError::new("XLSX_ERROR", format!("XLSX generation task failed: {e}")))??;

        let data = self.base.write_bytes(path, bytes, "XLSX workbook").await?;
        let size = data["size_bytes"].as_u64().unwrap_or(0);
        Ok(ToolOutput::success(format!(
            "Generated XLSX workbook '{}' ({} bytes)",
            file_name, size
        ))
        .with_mime_type(MIME_XLSX)
        .with_data(data))
    }
}

impl GenerateXlsxTool {
    fn build_xlsx_impl(sheets: &Value) -> ToolResult<Vec<u8>> {
        build_xlsx(sheets)
    }
}

// ---------------------------------------------------------------------------
// generate_csv
// ---------------------------------------------------------------------------

/// Tool that generates a CSV file from structured rows.
pub struct GenerateCsvTool {
    base: FileAuthoringBase,
}

impl GenerateCsvTool {
    /// Create a new CSV generation tool rooted at the workspace directory.
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self {
            base: FileAuthoringBase::new(workspace_dir),
        }
    }

    /// The JSON Schema for this tool's parameters.
    pub fn input_schema(&self) -> serde_json::Value {
        self.definition().to_json_schema()
    }
}

#[async_trait]
impl Tool for GenerateCsvTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "generate_csv",
                "Create a CSV file from structured rows and save it to the workspace. Use this "
                    + "for channel file requests instead of writing raw files or pasting CSV text.",
                HashMap::from([
                    (
                        "filename".to_string(),
                        ParameterDefinition::required_string(
                            "Output filename. .csv is appended if missing.",
                        ),
                    ),
                    (
                        "rows".to_string(),
                        ParameterDefinition::array(
                            "Non-empty array of row arrays. Values may be strings, numbers, "
                                + "booleans, null, arrays, or objects.",
                            ParameterDefinition::array(
                                "Cells",
                                ParameterDefinition::string("Cell value"),
                            ),
                        )
                        .required(),
                    ),
                    (
                        "headers".to_string(),
                        ParameterDefinition::array(
                            "Optional header row of column names.",
                            ParameterDefinition::string("Column name"),
                        ),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let filename = params["filename"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'filename' parameter"))?;
        let rows = params
            .get("rows")
            .cloned()
            .ok_or_else(|| ToolError::invalid_args("Missing 'rows' parameter"))?;
        let headers: Option<Vec<String>> = params
            .get("headers")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|h| h.as_str().unwrap_or("").to_string())
                    .collect()
            });

        let file_name = ensure_extension(filename, ".csv");
        let path = self.base.resolve(&file_name)?;

        let headers_for_task = headers.clone();
        let csv = tokio::task::spawn_blocking(move || -> ToolResult<String> {
            Ok(build_csv(&rows, headers_for_task.as_deref())?)
        })
        .await
        .map_err(|e| ToolError::new("CSV_ERROR", format!("CSV generation task failed: {e}")))??;

        let data = self.base.write_text(path, csv, "CSV document").await?;
        let size = data["size_bytes"].as_u64().unwrap_or(0);
        Ok(ToolOutput::success(format!(
            "Generated CSV file '{}' ({} bytes)",
            file_name, size
        ))
        .with_mime_type(MIME_CSV)
        .with_data(data))
    }
}

// ---------------------------------------------------------------------------
// generate_json
// ---------------------------------------------------------------------------

/// Tool that generates a validated, pretty-printed JSON file.
pub struct GenerateJsonTool {
    base: FileAuthoringBase,
}

impl GenerateJsonTool {
    /// Create a new JSON generation tool rooted at the workspace directory.
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self {
            base: FileAuthoringBase::new(workspace_dir),
        }
    }

    /// The JSON Schema for this tool's parameters.
    pub fn input_schema(&self) -> serde_json::Value {
        self.definition().to_json_schema()
    }
}

#[async_trait]
impl Tool for GenerateJsonTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "generate_json",
                "Validate and write a JSON document to the workspace. The content is parsed "
                    + "and re-serialized (pretty-printed) so invalid JSON fails fast.",
                HashMap::from([
                    (
                        "filename".to_string(),
                        ParameterDefinition::required_string(
                            "Output filename. .json is appended if missing.",
                        ),
                    ),
                    (
                        "content".to_string(),
                        ParameterDefinition::required_string("The JSON content to write."),
                    ),
                    (
                        "pretty".to_string(),
                        ParameterDefinition::boolean("Pretty-print the output (default true)."),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let filename = params["filename"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'filename' parameter"))?;
        let content = params["content"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'content' parameter"))?;
        let pretty = params.get("pretty").and_then(|v| v.as_bool()).unwrap_or(true);

        let parsed: Value = serde_json::from_str(content)
            .map_err(|e| ToolError::invalid_args(format!("Invalid JSON content: {e}")))?;
        let serialized = if pretty {
            serde_json::to_string_pretty(&parsed)
                .map_err(|e| ToolError::new("JSON_ERROR", format!("Failed to serialize JSON: {e}")))?
        } else {
            serde_json::to_string(&parsed)
                .map_err(|e| ToolError::new("JSON_ERROR", format!("Failed to serialize JSON: {e}")))?
        };
        let serialized = format!("{serialized}\n");

        let file_name = ensure_extension(filename, ".json");
        let path = self.base.resolve(&file_name)?;
        let data = self.base.write_text(path, serialized, "JSON document").await?;
        let size = data["size_bytes"].as_u64().unwrap_or(0);
        Ok(ToolOutput::success(format!(
            "Generated JSON file '{}' ({} bytes)",
            file_name, size
        ))
        .with_mime_type(MIME_JSON)
        .with_data(data))
    }
}

// ---------------------------------------------------------------------------
// generate_markdown
// ---------------------------------------------------------------------------

/// Tool that generates a Markdown document.
pub struct GenerateMarkdownTool {
    base: FileAuthoringBase,
}

impl GenerateMarkdownTool {
    /// Create a new Markdown generation tool rooted at the workspace directory.
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self {
            base: FileAuthoringBase::new(workspace_dir),
        }
    }

    /// The JSON Schema for this tool's parameters.
    pub fn input_schema(&self) -> serde_json::Value {
        self.definition().to_json_schema()
    }
}

#[async_trait]
impl Tool for GenerateMarkdownTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "generate_markdown",
                "Generate a Markdown document and save it to the workspace.",
                HashMap::from([
                    (
                        "filename".to_string(),
                        ParameterDefinition::required_string(
                            "Output filename. .md is appended if missing.",
                        ),
                    ),
                    (
                        "content".to_string(),
                        ParameterDefinition::required_string("The Markdown content to write."),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let filename = params["filename"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'filename' parameter"))?;
        let content = params["content"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'content' parameter"))?;

        let file_name = ensure_extension(filename, ".md");
        let path = self.base.resolve(&file_name)?;
        let data = self.base.write_text(path, content.to_string(), "Markdown document").await?;
        let size = data["size_bytes"].as_u64().unwrap_or(0);
        Ok(ToolOutput::success(format!(
            "Generated Markdown file '{}' ({} bytes)",
            file_name, size
        ))
        .with_mime_type(MIME_MARKDOWN)
        .with_data(data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_csv_with_headers() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GenerateCsvTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(json!({
                "filename": "people.csv",
                "rows": [["Alice", 30], ["Bob", 25]],
                "headers": ["name", "age"],
            }))
            .await;
        assert!(result.is_ok());

        let content = std::fs::read_to_string(dir.path().join("people.csv")).unwrap();
        assert!(content.contains("name,age"));
        assert!(content.contains("Alice,30"));
        assert!(result.unwrap().mime_type.as_deref() == Some(MIME_CSV));
    }

    #[tokio::test]
    async fn test_generate_csv_quotes_fields() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GenerateCsvTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(json!({
                "filename": "quoted.csv",
                "rows": [["Hello, world", "say \"hi\""]],
            }))
            .await;
        assert!(result.is_ok());

        let content = std::fs::read_to_string(dir.path().join("quoted.csv")).unwrap();
        assert!(content.contains("\"Hello, world\""));
        assert!(content.contains("\"say \"\"hi\"\"\""));
    }

    #[tokio::test]
    async fn test_generate_json_pretty_and_validates() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GenerateJsonTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(json!({
                "filename": "data.json",
                "content": r#"{"a": 1, "b": [true, null]}"#,
            }))
            .await;
        assert!(result.is_ok());

        let content = std::fs::read_to_string(dir.path().join("data.json")).unwrap();
        let parsed: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["a"], 1);
        assert_eq!(parsed["b"][0], true);
    }

    #[tokio::test]
    async fn test_generate_json_rejects_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GenerateJsonTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(json!({
                "filename": "bad.json",
                "content": "{ not json }",
            }))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_markdown() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GenerateMarkdownTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(json!({
                "filename": "doc.md",
                "content": "# Title\n\nSome **bold** text.",
            }))
            .await;
        assert!(result.is_ok());

        let content = std::fs::read_to_string(dir.path().join("doc.md")).unwrap();
        assert!(content.contains("# Title"));
        assert!(result.unwrap().mime_type.as_deref() == Some(MIME_MARKDOWN));
    }

    #[tokio::test]
    async fn test_generate_xlsx_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GenerateXlsxTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(json!({
                "filename": "book.xlsx",
                "sheets": [
                    {
                        "name": "People",
                        "rows": [["Name", "Age"], ["Alice", 30], ["Bob", 25]],
                    }
                ],
            }))
            .await;
        assert!(result.is_ok());

        let path = dir.path().join("book.xlsx");
        let mut workbook =
            calamine::open_workbook::<_, calamine::Xlsx<_>>(&path).unwrap();
        let range = workbook
            .worksheet_range("People")
            .unwrap_or_else(|_| panic!("missing sheet"));
        let rows: Vec<_> = range.rows().collect();
        assert_eq!(rows[0][0], calamine::Data::String("Name".into()));
        assert_eq!(rows[0][1], calamine::Data::String("Age".into()));
        assert_eq!(rows[1][0], calamine::Data::String("Alice".into()));
        assert_eq!(rows[1][1], calamine::Data::Float(30.0));
    }

    #[tokio::test]
    async fn test_generate_pdf_writes_valid_header() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GeneratePdfTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(json!({
                "filename": "report.pdf",
                "title": "Quarterly Report",
                "sections": [
                    { "heading": "Overview", "body": "Revenue grew 12% this quarter." },
                    { "heading": "Outlook", "body": "Hiring two engineers." },
                ],
            }))
            .await;
        assert!(result.is_ok());

        let bytes = std::fs::read(dir.path().join("report.pdf")).unwrap();
        assert!(bytes.starts_with(b"%PDF"));
        assert!(result.unwrap().mime_type.as_deref() == Some(MIME_PDF));
    }

    #[tokio::test]
    async fn test_path_traversal_denied() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GenerateMarkdownTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(json!({
                "filename": "../outside.md",
                "content": "secret",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "PATH_TRAVERSAL");
    }

    #[tokio::test]
    async fn test_xlsx_extension_appended() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GenerateXlsxTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(json!({
                "filename": "noext",
                "sheets": [{ "name": "S1", "rows": [[1, 2]] }],
            }))
            .await;
        assert!(result.is_ok());
        assert!(dir.path().join("noext.xlsx").exists());
    }
}
