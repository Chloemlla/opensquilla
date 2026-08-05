//! SSE stream assembly.
//!
//! Provides a generic `SseStream` adapter that reads a `reqwest::Response`
//! body as an SSE (Server-Sent Events) stream, parses individual `data:`
//! lines, and emits typed `StreamEvent` values via a user-provided parser
//! function.

use crate::types::{ProviderError, ProviderResult, StreamEvent};
use futures::{Stream, StreamExt};
use pin_project::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};
use tracing::trace;

/// A generic SSE stream adapter.
///
/// Wraps a `reqwest::Response` and parses SSE-formatted data lines, calling
/// the provided `parser` function to convert each `data:` line into a
/// `StreamEvent`.
#[pin_project]
pub struct SseStream<F> {
    /// The underlying byte stream from the response body.
    #[pin]
    body: futures::stream::BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    /// Buffer for accumulating partial lines across chunks.
    buffer: Vec<u8>,
    /// User-provided parser: maps a data line string to an optional event.
    parser: F,
    /// Whether the stream has finished.
    done: bool,
}

impl<F> SseStream<F>
where
    F: Fn(&str) -> Option<ProviderResult<StreamEvent>>,
{
    /// Create a new SSE stream from a `reqwest::Response`.
    pub fn new(response: reqwest::Response, parser: F) -> Self {
        Self {
            body: response.bytes_stream().boxed(),
            buffer: Vec::new(),
            parser,
            done: false,
        }
    }
}

/// Process a single SSE data line.
///
/// Strips the `data:` prefix and delegates to the parser.
fn process_line<F>(parser: &F, line: &str) -> Option<ProviderResult<StreamEvent>>
where
    F: Fn(&str) -> Option<ProviderResult<StreamEvent>>,
{
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Look for "data:" prefix (SSE format)
    let data = if let Some(content) = trimmed.strip_prefix("data: ") {
        content
    } else if let Some(content) = trimmed.strip_prefix("data:") {
        content
    } else {
        // Lines without "data:" prefix are ignored (e.g. event type, id)
        return None;
    };

    let data = data.trim();
    if data.is_empty() {
        return None;
    }

    trace!("SSE data: {data}");
    parser(data)
}

impl<F> Stream for SseStream<F>
where
    F: Fn(&str) -> Option<ProviderResult<StreamEvent>>,
{
    type Item = ProviderResult<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if *this.done {
            return Poll::Ready(None);
        }

        // Read available bytes from the response body
        let mut chunk = Vec::new();
        match this.body.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                chunk.extend_from_slice(&bytes);
            }
            Poll::Ready(Some(Err(e))) => {
                *this.done = true;
                return Poll::Ready(Some(Err(ProviderError::Network(e))));
            }
            Poll::Ready(None) => {
                *this.done = true;
                // Process any remaining data in the buffer
                if !this.buffer.is_empty() {
                    let remaining = String::from_utf8_lossy(this.buffer.as_slice()).to_string();
                    this.buffer.clear();
                    if let Some(event) = process_line(&*this.parser, &remaining) {
                        return Poll::Ready(Some(event));
                    }
                }
                return Poll::Ready(None);
            }
            Poll::Pending => {
                return Poll::Pending;
            }
        }

        this.buffer.extend_from_slice(&chunk);

        // Try to extract complete lines from the buffer
        let text = String::from_utf8_lossy(this.buffer.as_slice());

        // Find the last complete line (ending with \n)
        if let Some(last_newline) = text.rfind('\n') {
            let complete = text[..last_newline].to_string();
            // Keep the remainder in the buffer
            *this.buffer = text[last_newline + 1..].as_bytes().to_vec();

            // Process each line
            for line in complete.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Some(event) = process_line(&*this.parser, line) {
                    return Poll::Ready(Some(event));
                }
            }

            // Need more data
            Poll::Pending
        } else {
            // No complete line yet; wait for more data
            Poll::Pending
        }
    }
}

/// Tool call buffer: accumulates partial JSON tool call arguments across
/// multiple SSE deltas.
#[derive(Debug, Default)]
pub struct ToolCallBuffer {
    /// Accumulated partial JSON arguments keyed by tool call ID.
    pub buffers: std::collections::HashMap<String, String>,
    /// Tool names keyed by tool call ID.
    pub names: std::collections::HashMap<String, String>,
}

impl ToolCallBuffer {
    /// Accumulate a tool call delta.
    ///
    /// Returns `Some(ToolCall)` if the accumulated arguments form complete
    /// JSON, or `None` if more data is needed.
    pub fn accumulate(
        &mut self,
        id: &str,
        name: &str,
        arguments: &str,
    ) -> Option<opensquilla_core::types::ToolCall> {
        if !id.is_empty() {
            self.buffers
                .entry(id.to_string())
                .or_default()
                .push_str(arguments);
            if !name.is_empty() {
                self.names.insert(id.to_string(), name.to_string());
            }
        }

        // Try to parse the accumulated JSON
        if let Some(buf) = self.buffers.get(id) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(buf) {
                let tool_name = self.names.get(id).cloned().unwrap_or_default();
                let tc = opensquilla_core::types::ToolCall::new(id, tool_name, parsed);
                self.buffers.remove(id);
                self.names.remove(id);
                return Some(tc);
            }
        }

        None
    }

    /// Flush all remaining buffers, returning any tool calls that have
    /// names (even if JSON is incomplete).
    pub fn flush(&mut self) -> Vec<opensquilla_core::types::ToolCall> {
        let mut result = Vec::new();
        let ids: Vec<String> = self
            .buffers
            .keys()
            .chain(self.names.keys())
            .map(|k| k.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for id in ids {
            let buf = self.buffers.remove(&id).unwrap_or_default();
            let name = self.names.remove(&id).unwrap_or_default();
            if !name.is_empty() {
                let parsed = serde_json::from_str(&buf)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                result.push(opensquilla_core::types::ToolCall::new(id, name, parsed));
            }
        }

        result
    }
}

/// Extract reasoning content from a stream of events.
///
/// Collects all `StreamEvent::Reasoning` events and returns the concatenated
/// text, useful for extracting model thinking traces.
pub fn extract_reasoning(events: &[ProviderResult<StreamEvent>]) -> String {
    events
        .iter()
        .filter_map(|e| {
            if let Ok(StreamEvent::Reasoning { reasoning }) = e {
                Some(reasoning.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_buffer_accumulate_incomplete() {
        let mut buf = ToolCallBuffer::default();
        let result = buf.accumulate("call1", "get_weather", r#"{"loc"#);
        assert!(result.is_none());
    }

    #[test]
    fn test_tool_call_buffer_accumulate_complete() {
        let mut buf = ToolCallBuffer::default();
        let result = buf.accumulate("call1", "get_weather", r#"{"location":"NYC"}"#);
        assert!(result.is_some());
        let tc = result.unwrap();
        assert_eq!(tc.name, "get_weather");
        assert_eq!(tc.input["location"], "NYC");
    }

    #[test]
    fn test_tool_call_buffer_accumulate_across_chunks() {
        let mut buf = ToolCallBuffer::default();
        assert!(buf.accumulate("call1", "get_weather", r#"{"loc"#).is_none());
        assert!(buf.accumulate("call1", "", r#"ation":"NYC"}"#).is_some());
    }

    #[test]
    fn test_extract_reasoning() {
        let events = vec![
            Ok(StreamEvent::Text {
                text: "Hello".into(),
            }),
            Ok(StreamEvent::Reasoning {
                reasoning: "thinking...".into(),
            }),
            Ok(StreamEvent::Text {
                text: "World".into(),
            }),
            Ok(StreamEvent::Reasoning {
                reasoning: "more thinking".into(),
            }),
        ];
        let reasoning = extract_reasoning(&events);
        assert_eq!(reasoning, "thinking...more thinking");
    }
}
