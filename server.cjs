const express = require("express");
const path = require("path");
const { once } = require("node:events");
const app = express();
const RAG_URL = (process.env.RAG_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const timeoutMs = Number(process.env.RAG_TIMEOUT_MS || 23000);
app.use(express.json({ limit: "100kb" }));
app.use(express.static(path.join(__dirname, "public"), { maxAge: 0 }));

function failure(res, status, error) {
  if (res.destroyed || res.writableEnded) return;
  if (res.headersSent) {
    res.end(JSON.stringify({ type: "error", status, error }) + "\n");
  } else {
    res.status(status).json({ error });
  }
}

function proxy(streaming) {
  return async (req, res) => {
    const { question, history = [] } = req.body || {};
    if (typeof question !== "string" || !question.trim() || question.length > 4000 ||
        !Array.isArray(history) || !history.every(m => m && ["user", "assistant"].includes(m.role) && typeof m.content === "string")) {
      return res.status(400).json({ error: "Invalid question or history." });
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new DOMException("RAG deadline", "TimeoutError")), timeoutMs);
    const onClose = () => { if (!res.writableEnded) controller.abort(); };
    res.on("close", onClose);
    let reader;
    try {
      const response = await fetch(`${RAG_URL}/answer${streaming ? "/stream" : ""}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({ question: question.trim(), history: history.slice(-4).map(m => ({ role: m.role, content: m.content.slice(0, 1500) })) })
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        const detail = typeof data.detail === "string" ? data.detail : "Python rejected the request. Check its terminal.";
        return failure(res, response.status, detail);
      }
      if (!streaming) {
        const result = await response.json();
        if (typeof result.answer !== "string" || !result.answer.trim()) return failure(res, 502, "Python returned an invalid answer.");
        return res.json(result);
      }
      if (!response.headers.get("content-type")?.includes("application/x-ndjson") || !response.body) {
        return failure(res, 502, "Python is running an older version. Replace all optimized files and restart both services.");
      }
      res.set({ "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no" });
      res.flushHeaders();
      reader = response.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!res.write(Buffer.from(value))) {
          await once(res, "drain", { signal: controller.signal });
        }
      }
      res.end();
    } catch (error) {
      if (res.destroyed) return;
      const timedOut = controller.signal.aborted || error.name === "TimeoutError" || error.name === "AbortError";
      failure(res, timedOut ? 504 : 503, timedOut
        ? "The answer service timed out. Please try again shortly."
        : "Python RAG service is unreachable. Start Python on port 8000 and check its terminal.");
    } finally {
      clearTimeout(timer);
      res.off("close", onClose);
      if (reader) await reader.cancel().catch(() => {});
      controller.abort();
    }
  };
}
app.post("/api/chat", proxy(false));
app.post("/api/chat/stream", proxy(true));
app.get("/api/health", async (req, res) => {
  try {
    const upstream = await fetch(`${RAG_URL}/health`, { signal: AbortSignal.timeout(2500) });
    res.status(upstream.status).json(await upstream.json());
  } catch {
    res.status(503).json({ error: "Start the Python service on port 8000." });
  }
});
if (require.main === module) {
  app.listen(Number(process.env.PORT || 3000), () => console.log(`Fast RAG v2: http://localhost:${process.env.PORT || 3000}`));
}
module.exports = app;
