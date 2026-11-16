/* Decode NDJSON even when UTF-8 characters and JSON records cross network chunks. */
async function readAnswerStream(body, onEvent) {
  if (!body) throw new Error("The server returned no response stream.");
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line) onEvent(JSON.parse(line));
      }
      if (buffer.length > 200000) throw new Error("Invalid response stream.");
      if (done) break;
    }
    if (buffer.trim()) onEvent(JSON.parse(buffer));
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
if (typeof module !== "undefined") module.exports = { readAnswerStream };
