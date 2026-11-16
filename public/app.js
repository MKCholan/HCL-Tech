const API_URL = "/api/chat/stream";
const form = document.getElementById("question-form");
const input = document.getElementById("question");
const submitButton = document.getElementById("submit-button");
const chat = document.getElementById("chat");
const status = document.getElementById("status");
const history = [];
let isWaiting = false;
let activeController;

function updateSubmitButton() {
  submitButton.disabled = !isWaiting && !input.value.trim();
  submitButton.textContent = isWaiting ? "Stop" : "Submit";
}
function scrollToLatestMessage() { chat.scrollTop = chat.scrollHeight; }
function addSources(message, sources) {
  if (!Array.isArray(sources) || !sources.length) return;
  const references = document.createElement("div");
  references.className = "sources";
  references.textContent = "Sources:\n" + sources.map((source, index) => {
    if (typeof source === "string") return source;
    return `[${source.reference || index + 1}] ${source.source || "Document"}` + (source.page != null ? ` — Page ${source.page}` : "");
  }).join("\n");
  message.appendChild(references);
}
function addMessage(role, text) {
  document.getElementById("welcome")?.remove();
  const message = document.createElement("article");
  message.className = `message ${role}`;
  const sender = document.createElement("div");
  sender.className = "sender";
  sender.textContent = role === "user" ? "You" : "Assistant";
  const content = document.createElement("div");
  content.className = "message-text";
  content.textContent = text;
  message.append(sender, content);
  chat.appendChild(message);
  scrollToLatestMessage();
  return { message, content, sender };
}
function addError(text) {
  const error = document.createElement("div");
  error.className = "message error";
  error.textContent = text;
  chat.appendChild(error);
  scrollToLatestMessage();
}
async function submitQuestion(event) {
  event.preventDefault();
  if (isWaiting) { activeController?.abort(); return; }
  const question = input.value.trim();
  if (!question) return;
  isWaiting = true;
  input.value = "";
  activeController = new AbortController();
  const controller = activeController;
  const timeout = setTimeout(() => controller.abort(new DOMException("Deadline", "TimeoutError")), 26000);
  const started = performance.now();
  let stage = "Connecting…";
  const progress = setInterval(() => {
    status.textContent = `${stage} (${((performance.now() - started) / 1000).toFixed(1)}s)`;
  }, 250);
  status.textContent = stage;
  updateSubmitButton();
  addMessage("user", question);
  let assistant, answer = "", completed = false;
  try {
    const response = await fetch(API_URL, {
      method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
      body: JSON.stringify({ question, history: history.slice(-4) })
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `Request failed (${response.status}).`);
    }
    await readAnswerStream(response.body, (event) => {
      if (event.type === "status") stage = event.message;
      if (event.type === "error") throw new Error(event.error);
      if (event.type === "token") {
        assistant ||= addMessage("assistant", "");
        answer += event.text;
        assistant.content.textContent = answer;
        stage = "Receiving answer…";
        scrollToLatestMessage();
      }
      if (event.type === "done") {
        answer = event.answer;
        if (typeof answer !== "string" || !answer.trim()) throw new Error("Empty answer returned.");
        assistant ||= addMessage("assistant", "");
        assistant.content.textContent = answer;
        addSources(assistant.message, event.sources);
        if (event.truncated) assistant.sender.textContent = "Assistant — answer reached its length limit";
        completed = true;
        stage = event.cached ? "Answer from cache" : "Answer received";
      }
    });
    if (!completed) throw new Error("The response was interrupted before the answer finished.");
    // Incomplete answers must never become factual context for subsequent questions.
    if (!assistant.sender.textContent.includes("length limit")) {
      history.push({ role: "user", content: question }, { role: "assistant", content: answer.slice(0, 1500) });
      history.splice(0, Math.max(0, history.length - 4));
    }
    status.textContent = `${stage} in ${((performance.now() - started) / 1000).toFixed(1)}s`;
  } catch (error) {
    controller.abort();
    if (assistant && !completed) assistant.sender.textContent = "Assistant — incomplete answer";
    const reason = controller.signal.reason;
    addError(error.name === "TimeoutError" || reason?.name === "TimeoutError"
      ? "The request timed out. Please try again shortly."
      : error.name === "AbortError" ? "Request stopped."
      : error instanceof TypeError ? "Could not connect to the server. Check both terminals."
      : error.message);
    if (!input.value.trim()) input.value = question;
    status.textContent = "";
  } finally {
    clearTimeout(timeout);
    clearInterval(progress);
    activeController = null;
    isWaiting = false;
    updateSubmitButton();
    input.focus();
  }
}
input.addEventListener("input", updateSubmitButton);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!isWaiting && !submitButton.disabled) form.requestSubmit();
  }
});
form.addEventListener("submit", submitQuestion);
