#!/usr/bin/env node
// Send a chat-completion request for deepseek-v4-flash via tokenrhythm.studio,
// tagged with the same OpenSquilla client markers the py backend attaches
// (provider/tokenrhythm_correlation.py + provider/app_attribution.py).
//
// Usage:
//   TOKENRHYTHM_API_KEY=... node scripts/tokenrhythm-deepseek-v4-flash.mjs \
//     --prompt "summarize deepseek-v4" --call-kind agent.chat
//
// Flags:
//   --model <id>          model id (default deepseek-v4-flash)
//   --prompt <text>       user message (default "Say hello")
//   --call-kind <kind>    agent.chat | subagent.chat | *.ensemble.proposer ...
//   --no-stream           non-streaming JSON response
//   --no-markers          omit all OpenSquilla markers (privacy-sim)
//   --install-id/--session-id/--turn-id/--execution-id <id>  override ids

import crypto from "node:crypto";

// Force UTF-8 on both streams regardless of the Windows console code page so
// reasoning_content / content (Chinese) never round-trips through GBK/CP936.
process.stdout.setDefaultEncoding("utf8");
process.stderr.setDefaultEncoding("utf8");

const ENDPOINT = "https://tokenrhythm.studio/v1/chat/completions";
const APP_REFERER = "https://opensquilla.ai";
const APP_TITLE = "OpenSquilla";
const MODEL_DEFAULT = "deepseek-v4-flash";
const CALL_KIND_DEFAULT = "agent.chat";
const CALL_KIND_MAX_LENGTH = 96;

// Header names mirror opensquilla/provider/tokenrhythm_correlation.py.
const HDR = {
  referer: "HTTP-Referer",
  title: "X-Title",
  installId: "X-OpenSquilla-Install-Id",
  sessionId: "X-OpenSquilla-Session-Id",
  turnId: "X-OpenSquilla-Turn-Id",
  executionId: "X-OpenSquilla-Execution-Id",
  callKind: "X-OpenSquilla-Call-Kind",
};

const CORRELATION_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const ABSENT_IDS = new Set(["none", "null", "unknown"]);
const AUX_ROLES = new Set([
  "meta", "vision_gate", "session_flush", "media", "naming",
  "compaction", "image_generation", "other",
]);
const ENSEMBLE_PHASES = new Set(["proposer", "aggregator", "fallback_single"]);
const TRUE_VALUES = new Set(["1", "true", "yes", "on"]);

// _safe_correlation_id: mirrors tokenrhythm_correlation.py:96
function safeId(value) {
  const s = String(value ?? "").trim();
  if (!s || ABSENT_IDS.has(s.toLowerCase()) || !CORRELATION_ID_RE.test(s)) return "";
  return s;
}

// _safe_call_kind: mirrors tokenrhythm_correlation.py:107 (agent.chat /
// subagent.chat / auxiliary.* / {agent,subagent}.ensemble.<phase>).
function safeCallKind(value) {
  const candidate = String(value ?? "").trim();
  if (!candidate || candidate.length > CALL_KIND_MAX_LENGTH) return "";
  let parts = candidate.split(".");
  if (parts[parts.length - 1] === "provider_fallback") parts = parts.slice(0, -1);
  if (parts.length === 2 && parts[0] === "auxiliary" && AUX_ROLES.has(parts[1])) return candidate;
  if (parts.length === 2 && (parts[0] === "agent" || parts[0] === "subagent") && parts[1] === "chat") return candidate;
  if (parts.length === 3 && (parts[0] === "agent" || parts[0] === "subagent") && parts[1] === "ensemble" && ENSEMBLE_PHASES.has(parts[2])) return candidate;
  return "";
}

function parseArgs(argv) {
  const args = {
    model: MODEL_DEFAULT,
    prompt: "Say hello",
    callKind: CALL_KIND_DEFAULT,
    stream: true,
    markers: true,
    installId: crypto.randomUUID(),
    sessionId: crypto.randomUUID(),
    turnId: crypto.randomUUID(),
    executionId: crypto.randomUUID(),
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    switch (a) {
      case "--model": args.model = argv[++i]; break;
      case "--prompt": args.prompt = argv[++i]; break;
      case "--call-kind": args.callKind = argv[++i]; break;
      case "--no-stream": args.stream = false; break;
      case "--no-markers": args.markers = false; break;
      case "--install-id": args.installId = argv[++i]; break;
      case "--session-id": args.sessionId = argv[++i]; break;
      case "--turn-id": args.turnId = argv[++i]; break;
      case "--execution-id": args.executionId = argv[++i]; break;
      case "--help":
      case "-h":
        console.log(`Usage: node ${process.argv[1]} [--model <id>] [--prompt <text>] [--call-kind <kind>] [--no-stream] [--no-markers]`);
        process.exit(0);
      default:
        console.error(`Unknown flag: ${a}`);
        process.exit(1);
    }
  }
  return args;
}

// Marker assembly mirrors openai.py:3330-3344: attribution headers always
// when the host matches, install-id on its own, correlation as all-or-nothing.
function markerHeaders(args) {
  const headers = {
    [HDR.referer]: APP_REFERER,
    [HDR.title]: APP_TITLE,
  };

  const installId = safeId(args.installId);
  if (installId) headers[HDR.installId] = installId;

  const sessionId = safeId(args.sessionId);
  const turnId = safeId(args.turnId);
  const executionId = safeId(args.executionId);
  const callKind = safeCallKind(args.callKind);
  if (sessionId && turnId && executionId && callKind) {
    Object.assign(headers, {
      [HDR.sessionId]: sessionId,
      [HDR.turnId]: turnId,
      [HDR.executionId]: executionId,
      [HDR.callKind]: callKind,
    });
  } else {
    console.error("! correlation id(s) invalid -> correlation headers omitted (all-or-nothing)");
  }
  return headers;
}

function buildHeaders(args, apiKey) {
  const headers = {
    "Content-Type": "application/json",
    Accept: args.stream ? "text/event-stream" : "application/json",
    Authorization: `Bearer ${apiKey}`,
  };
  if (args.markers) Object.assign(headers, markerHeaders(args));
  return headers;
}

async function consumeStream(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      const t = line.trim();
      if (!t.startsWith("data:")) continue;
      const data = t.slice(5).trim();
      if (data === "[DONE]") return;
      try {
        const json = JSON.parse(data);
        const delta = json.choices?.[0]?.delta ?? {};
        // DeepSeek-style reasoning_content streams alongside content.
        if (delta.reasoning_content) process.stderr.write(delta.reasoning_content);
        if (delta.content) process.stdout.write(delta.content);
      } catch {
        // keep-alive / comment lines
      }
    }
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const apiKey = process.env.TOKENRHYTHM_API_KEY;
  if (!apiKey) {
    console.error("Set TOKENRHYTHM_API_KEY first.");
    process.exit(1);
  }

  const body = {
    model: args.model,
    messages: [{ role: "user", content: args.prompt }],
    stream: args.stream,
  };
  const headers = buildHeaders(args, apiKey);

  console.error(`POST ${ENDPOINT}`);
  console.error(`  model=${args.model} stream=${args.stream} markers=${args.markers} call-kind=${args.callKind}`);
  for (const [name, value] of Object.entries(headers)) {
    if (name === "Authorization") continue;
    console.error(`  ${name}: ${value}`);
  }

  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    console.error(`HTTP ${res.status}: ${await res.text()}`);
    process.exit(1);
  }

  if (args.stream) {
    console.error("\n--- stream ---");
    await consumeStream(res);
    process.stdout.write("\n");
  } else {
    console.log(JSON.stringify(await res.json(), null, 2));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
