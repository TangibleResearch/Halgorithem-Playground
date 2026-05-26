const els = {
  serverState: document.querySelector("#serverState"),
  sourceText: document.querySelector("#sourceText"),
  responseText: document.querySelector("#responseText"),
  sourceUrls: document.querySelector("#sourceUrls"),
  sourceName: document.querySelector("#sourceName"),
  threshold: document.querySelector("#threshold"),
  chatgptPrompt: document.querySelector("#chatgptPrompt"),
  openaiKey: document.querySelector("#openaiKey"),
  generateButton: document.querySelector("#generateButton"),
  fileInput: document.querySelector("#fileInput"),
  dropzone: document.querySelector("#dropzone"),
  fileList: document.querySelector("#fileList"),
  verifyButton: document.querySelector("#verifyButton"),
  clearButton: document.querySelector("#clearButton"),
  loadExample: document.querySelector("#loadExample"),
  notice: document.querySelector("#notice"),
  metrics: document.querySelector("#metrics"),
  results: document.querySelector("#results"),
  generated: document.querySelector("#generated"),
  sourcePills: document.querySelector("#sourcePills"),
  runMeta: document.querySelector("#runMeta")
};

let files = [];

const sample = {
  source: "BASIC was created in 1964 by John Kemeny at Dartmouth College. BASIC was designed for students. The sample has a mass of 10 kilograms. As of 2024, Project Helios is active.",
  response: "BASIC was created in 1964 by John Kemeny at Dartmouth College. The sample has a mass of 10 pounds. Project Helios is currently active.",
  prompt: "Explain what BASIC is, who created it, where it was created, and include one claim about the sample mass."
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setNotice(message) {
  els.notice.textContent = message || "";
  els.notice.classList.toggle("visible", Boolean(message));
}

function setBusy(isBusy) {
  els.verifyButton.disabled = isBusy;
  els.generateButton.disabled = isBusy;
  els.verifyButton.textContent = isBusy ? "Running..." : "Run verification";
  if (isBusy) els.runMeta.textContent = "Working";
}

function setGenerating(isGenerating) {
  els.generateButton.disabled = isGenerating;
  els.verifyButton.disabled = isGenerating;
  els.generateButton.textContent = isGenerating ? "Asking..." : "Ask ChatGPT";
  if (isGenerating) els.runMeta.textContent = "Asking ChatGPT";
}

function renderFiles() {
  els.fileList.innerHTML = files.map((file) => `<span>${escapeHtml(file.name)} · ${Math.ceil(file.size / 1024)} KB</span>`).join("");
}

function setFiles(nextFiles) {
  files = [...files, ...Array.from(nextFiles)];
  renderFiles();
}

function badgeClass(status) {
  if (status === "SUPPORTED") return "good";
  if (status === "WEAK_SUPPORT" || status === "UNVERIFIABLE_DENIAL") return "warn";
  if (["HALLUCINATION", "CONTRADICTION", "ERROR"].includes(status)) return "bad";
  return "";
}

function displayStatus(status) {
  if (status === "HALLUCINATION") return "UNSUPPORTED";
  if (status === "WEAK_SUPPORT") return "WEAK SUPPORT";
  if (status === "UNVERIFIABLE_DENIAL") return "UNVERIFIABLE";
  return status || "UNKNOWN";
}

function renderMetrics(summary) {
  const confidence = Math.round((summary?.confidence || 0) * 100);
  els.metrics.innerHTML = `
    <div><strong>${summary?.total_claims || 0}</strong><span>Claims</span></div>
    <div><strong>${summary?.supported || 0}</strong><span>Supported</span></div>
    <div><strong>${summary?.issues || 0}</strong><span>Issues</span></div>
    <div><strong>${confidence}%</strong><span>Confidence</span></div>
  `;
}

function renderSources(sources = []) {
  els.sourcePills.innerHTML = sources.map((source) => (
    `<span>${escapeHtml(source.name)} · ${source.characters.toLocaleString()} chars</span>`
  )).join("");
}

function renderResults(results = []) {
  if (!results.length) {
    els.results.innerHTML = `<div class="empty">No meaningful claims were found.</div>`;
    return;
  }
  els.results.innerHTML = results.map((result) => {
    const score = typeof result.score === "number" ? result.score.toFixed(3) : "n/a";
    const confidence = typeof result.confidence === "number" ? Math.round(result.confidence * 100) : 0;
    const reason = result.reason ? `<div class="claim-meta">Reason: ${escapeHtml(result.reason)}</div>` : "";
    const unsupported = result.unsupported_terms?.length
      ? `<div class="claim-meta">Unsupported: ${escapeHtml(result.unsupported_terms.join(", "))}</div>`
      : "";
    return `
      <article class="claim-card">
        <div class="claim-head">
          <div class="claim-text">${escapeHtml(result.claim)}</div>
          <span class="badge ${badgeClass(result.status)}">${escapeHtml(displayStatus(result.status))}</span>
        </div>
        <div class="evidence">${escapeHtml(result.chunk_text || result.matched_chunk || "No evidence returned.")}</div>
        <div class="claim-meta">Score ${score} · Confidence ${confidence}% · Source ${escapeHtml(result.matched_source || "n/a")}</div>
        ${reason}
        ${unsupported}
      </article>
    `;
  }).join("");
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("offline");
    els.serverState.textContent = "Django online";
  } catch {
    els.serverState.textContent = "Server offline";
  }
}

async function verify() {
  setNotice("");
  setBusy(true);
  const form = new FormData();
  form.append("source_text", els.sourceText.value);
  form.append("response_text", els.responseText.value);
  form.append("source_urls", els.sourceUrls.value);
  form.append("source_name", els.sourceName.value);
  form.append("threshold", els.threshold.value || "0.30");
  for (const file of files) form.append("files", file);

  try {
    const response = await fetch("/api/verify", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || data.detail || "Verification failed.");
    renderMetrics(data.summary);
    renderSources(data.sources);
    renderResults(data.results);
    els.generated.textContent = "";
    els.generated.classList.remove("visible");
    els.runMeta.textContent = `${data.summary.total_claims} claims checked`;
  } catch (error) {
    setNotice(error.message);
    els.runMeta.textContent = "Needs attention";
  } finally {
    setBusy(false);
  }
}

async function generateChatgpt() {
  setNotice("");
  const prompt = els.chatgptPrompt.value.trim();
  const apiKey = els.openaiKey.value.trim();
  if (!prompt) {
    setNotice("Add a ChatGPT instruction first.");
    return;
  }
  if (!apiKey) {
    setNotice("Add your OpenAI API key, or restart Django with OPENAI_API_KEY set.");
    return;
  }
  setGenerating(true);
  try {
    const response = await fetch("/api/chatgpt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, api_key: apiKey })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || data.detail || "ChatGPT request failed.");
    els.responseText.value = data.response_text || "";
    els.generated.textContent = `ChatGPT response, not yet verified:\n\n${data.response_text || ""}`;
    els.generated.classList.add("visible");
    els.runMeta.textContent = "ChatGPT response ready";
  } catch (error) {
    setNotice(error.message);
    els.runMeta.textContent = "Needs attention";
  } finally {
    setGenerating(false);
  }
}

els.fileInput.addEventListener("change", (event) => setFiles(event.target.files));

["dragenter", "dragover"].forEach((name) => {
  els.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((name) => {
  els.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove("dragover");
  });
});

els.dropzone.addEventListener("drop", (event) => setFiles(event.dataTransfer.files));
els.generateButton.addEventListener("click", generateChatgpt);
els.verifyButton.addEventListener("click", verify);
els.clearButton.addEventListener("click", () => {
  files = [];
  els.sourceText.value = "";
  els.responseText.value = "";
  els.sourceUrls.value = "";
  els.chatgptPrompt.value = "";
  els.openaiKey.value = "";
  els.generated.textContent = "";
  els.generated.classList.remove("visible");
  renderFiles();
  renderSources([]);
  renderMetrics(null);
  renderResults([]);
  setNotice("");
});
els.loadExample.addEventListener("click", () => {
  els.sourceText.value = sample.source;
  els.responseText.value = sample.response;
  els.chatgptPrompt.value = sample.prompt;
});

checkHealth();
