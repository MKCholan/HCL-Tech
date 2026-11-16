# Fast RAG v2: start this copy

Extract this archive into a NEW folder. Stop the old Node and Python terminals
with Ctrl+C, then open the extracted HCLTech folder in VS Code. This update changes
Python, Node and the frontend together; run all files from the same extracted copy.

Use Node.js 20+ and Python 3.11+ (3.11/3.12 are convenient for package compatibility).
The existing data, Chroma index, and API key have been retained. You do not need to
run embed.py again.

## Terminal 1 — Python

From the extracted HCLTech folder:

```powershell
cd hcl-tech-retrival
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete`. Keep this terminal running.

Open http://127.0.0.1:8000/health. This exact version should show:

- `build`: `fast-rag-stream-v2`
- `model`: `gemini-2.5-flash-lite`
- `request_deadline_seconds`: `20`
- `streaming`: `true`

The health check confirms local readiness, not Gemini access or API-key validity.
If you see the old health output, a previous Python process is still running or
you started it from the old folder. Close that process before restarting.

## Terminal 2 — Node

Open another terminal in the extracted HCLTech folder:

```powershell
npm install
npm start
```

Open http://localhost:3000 and press Ctrl+F5 to load the updated JavaScript.
Ask a question. Text should appear progressively as Gemini sends it; the button
becomes Stop while working. Stopping or timing out permits another submission.

## What was optimized

- Removed the extra generation call that rewrote every follow-up question.
  Retrieval uses the previous topic for explicit references. For ambiguous short
  follow-ups, restate the topic to improve retrieval.
- Set the included .env chat model to gemini-2.5-flash-lite and disabled its
  optional thinking. If access to that model is unavailable for your key, use a
  supported Flash-Lite/Flash model from your model list. The code normalizes an
  optional `models/` prefix and uses low thinking for Gemini 3 Flash variants.
- Streamed Gemini text through Python and Node to the page. The old page waited
  until the entire answer was generated before displaying anything.
- Removed hidden SDK retry loops from interactive requests. The answer API uses
  Gemini's documented REST endpoints through one reusable asynchronous HTTPX
  connection pool. Offline ingestion remains unchanged.
- Added a 20-second total Python request deadline; embedding alone is limited to
  6 seconds. Node has a 23-second guard and the browser has a 26-second guard.
  These limits stop long waits; a timeout is an error, not a successful answer.
- A browser Stop/disconnect cancels the HTTP stream. Slow local Chroma work runs
  in a background thread; cancellation cannot forcibly terminate a native local
  query already running, but it stops waiting and cancels outbound generation.
- Opened the existing index once at Python startup. Each answer uses up to three
  chunks, limited recent history, and a request for a concise response.
- Limited model output to 1024 tokens. Length-limited output is labelled in the
  page and excluded from both the answer cache and successful conversation history.
- Cached query embeddings for five minutes. Completed answers are cached for
  five minutes only for the SAME question, recent history, and model in the same
  Python process. A new conversation context may therefore generate a fresh answer.
  Errors and interrupted responses are never cached. Restart Python after any
  index rebuild so its collection handles and caches match the new index.
- Kept source references and document grounding; failures now identify quota,
  unavailable model/service, timeouts, and local processing errors.

## What speed should you expect?

A few-second answer is a target, not a measured result or guarantee. Streaming
reduces time until the user sees text; it does not remove Gemini processing or
provider queue time. The 20-second deadline prevents another five-minute Python
request. It can instead return a clear error if Gemini is slow or unavailable.

The Python terminal now prints separate times for embedding, document search,
first answer text, generation, and total request time. Use these real timings to
judge performance on your connection and API quota. If fresh queries still time
out in generation, Google's model availability remains the bottleneck. Further
local prompt changes cannot guarantee provider latency.

## Configuration and troubleshooting

The API reads `hcl-tech-retrival/.env`. Existing system environment variables take
precedence; /health and the Ready log show the ACTUAL model being used. Restart
Python after changing its configuration. Keep the embedding model, dimensions,
and task type aligned with the supplied index. Changing the chat model does not
require rebuilding embeddings.

If Node and Python run in separate containers/machines, configure Node's RAG_URL
to reach Python. The default http://127.0.0.1:8000 requires the same network namespace.
A hosting reverse proxy must allow streaming and disable response buffering.

Quota errors: check the API quota and wait; there are no automatic retry storms.
Model unavailable: check the exact model ID and access for your API key.
Unreachable Python: check Terminal 1 and http://127.0.0.1:8000/health.
Missing npm start: confirm you are in THIS extracted HCLTech folder.

## Validation

20 offline regression checks passed:

```powershell
npm test
python -m unittest discover -s tests -p "test_*.py" -v
```

The checks cover streamed text arriving before completion, fragmented Unicode,
successive questions, context-safe caching, quota/overload failures, cancellation,
request deadlines, source references, partial-answer handling, and recovery.
Browser event-handler checks use a small DOM stand-in; a real browser launch was
unavailable in the repair environment. Live Google requests and the Chroma runtime
were not benchmarked here. The original index files are preserved byte-for-byte.

## API references

- https://ai.google.dev/api/generate-content
- https://ai.google.dev/api/embeddings
- https://www.python-httpx.org/async/
