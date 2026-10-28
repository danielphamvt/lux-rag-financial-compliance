/* ==========================================================================
   Evaluation & RAG Audit Dashboard Client Controller (/static/js/evaluation.js)
   Standalone script — zero coupling with the main Gradio Chatbot UI.
   ========================================================================== */

(function () {
  "use strict";

  const state = {
    history: [],
    activeFilename: null,
    matrix: [],
    selectedScenarioId: null,
    activeFilter: "all",
    isRunning: false,
    eventSource: null,
  };

  const SVG_SEARCH =
    '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>';
  const SVG_CALC =
    '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="2" width="16" height="20" rx="2"></rect><line x1="8" y1="6" x2="16" y2="6"></line><line x1="8" y1="11" x2="16" y2="11"></line><line x1="8" y1="16" x2="12" y2="16"></line></svg>';

  function escapeHtml(val) {
    if (val === null || val === undefined) return "";
    return String(val)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatNum(val) {
    if (val === null || val === undefined || val === "") return "—";
    const num = Number(val);
    if (Number.isNaN(num)) return escapeHtml(val);
    return num.toLocaleString("en-US", { maximumFractionDigits: 6 });
  }

  function splitReasoningAndConclusion(rawText) {
    const text = String(rawText || "");
    if (!text.trim()) {
      return { reasoning: "", conclusion: "" };
    }
    const concIdx = text.indexOf("[CONCLUSION]");
    if (concIdx !== -1) {
      const rPart = text
        .slice(0, concIdx)
        .replace(/^\s*\[REASONING\]\s*/i, "")
        .trim();
      const cPart = text.slice(concIdx + "[CONCLUSION]".length).trim();
      return { reasoning: rPart, conclusion: cPart };
    }
    const hIdx = text.search(/###\s*CONCLUSION\b/i);
    if (hIdx > 0) {
      const rPart = text
        .slice(0, hIdx)
        .replace(/^\s*\[REASONING\]\s*/i, "")
        .trim();
      const cPart = text.slice(hIdx).trim();
      return { reasoning: rPart, conclusion: cPart };
    }
    return { reasoning: "", conclusion: text.trim() };
  }

  function renderInlineMarkdown(rawLine) {
    if (!rawLine) return "";
    const codeTokens = [];
    const withPlaceholders = String(rawLine).replace(/`([^`\n]+)`/g, (_, code) => {
      const idx = codeTokens.length;
      codeTokens.push(`<code class="md-inline-code">${escapeHtml(code)}</code>`);
      return `\u0000CODE_${idx}\u0000`;
    });

    let safe = escapeHtml(withPlaceholders);
    safe = safe.replace(/\*\*\*([^*]+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    safe = safe.replace(/\*\*([^*]+?)\*\*/g, "<strong>$1</strong>");
    // Only italicize *word* when preceded by start/whitespace and starting with a letter (preserves Footnote (*1))
    safe = safe.replace(
      /(^|\s)\*([A-Za-z][^*\n]*?)\*(?=$|[\s.,;:!?)\]])/g,
      "$1<em>$2</em>"
    );

    return safe.replace(/\u0000CODE_(\d+)\u0000/g, (_, idx) => codeTokens[Number(idx)] || "");
  }

  function parseTableCells(line) {
    let trimmed = line.trim();
    if (trimmed.startsWith("|")) trimmed = trimmed.slice(1);
    if (trimmed.endsWith("|")) trimmed = trimmed.slice(0, -1);
    return trimmed.split("|").map((c) => renderInlineMarkdown(c.trim()));
  }

  function isTableSeparator(line) {
    if (!line || !line.includes("|") || !line.includes("-")) return false;
    return /^\s*\|?(\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$/.test(line);
  }

  function renderMarkdown(rawText) {
    const text = String(rawText || "").replace(/\r\n/g, "\n").trim();
    if (!text) return '<p class="md-p">No response recorded.</p>';

    const lines = text.split("\n");
    const blocks = [];
    let i = 0;

    while (i < lines.length) {
      const raw = lines[i];
      const trimmed = raw.trim();

      if (!trimmed) {
        i += 1;
        continue;
      }

      // Fenced code block
      if (trimmed.startsWith("```")) {
        const codeLines = [];
        i += 1;
        while (i < lines.length && !lines[i].trim().startsWith("```")) {
          codeLines.push(lines[i]);
          i += 1;
        }
        if (i < lines.length) i += 1;
        blocks.push(`<pre class="md-pre"><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        continue;
      }

      // Horizontal Rule
      if (/^(?:-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
        blocks.push('<hr class="md-hr">');
        i += 1;
        continue;
      }

      // Headings (# .. ####)
      const hMatch = trimmed.match(/^(#{1,4})\s+(.*)$/);
      if (hMatch) {
        const level = Math.min(4, Math.max(2, hMatch[1].length));
        blocks.push(
          `<h${level} class="md-heading md-h${level}">${renderInlineMarkdown(hMatch[2])}</h${level}>`
        );
        i += 1;
        continue;
      }

      // GFM Pipe Table
      if (trimmed.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
        const headers = parseTableCells(trimmed);
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].trim().includes("|") && lines[i].trim() !== "") {
          if (!isTableSeparator(lines[i])) {
            rows.push(parseTableCells(lines[i]));
          }
          i += 1;
        }
        const thead = `<thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead>`;
        const tbody = `<tbody>${rows
          .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`)
          .join("")}</tbody>`;
        blocks.push(
          `<div class="eval-table-wrap md-table-wrap"><table class="eval-table md-table">${thead}${tbody}</table></div>`
        );
        continue;
      }

      // Blockquote
      if (trimmed.startsWith(">")) {
        const qLines = [];
        while (i < lines.length && lines[i].trim().startsWith(">")) {
          qLines.push(renderInlineMarkdown(lines[i].trim().replace(/^>\s?/, "")));
          i += 1;
        }
        blocks.push(`<blockquote class="md-blockquote">${qLines.join("<br>")}</blockquote>`);
        continue;
      }

      // Ordered List (with support for nested sub-bullets inside items)
      if (/^\d+\.\s+/.test(trimmed)) {
        const items = [];
        while (i < lines.length) {
          const lineTrim = lines[i].trim();
          const olMatch = lineTrim.match(/^(\d+)\.\s+(.*)$/);
          if (olMatch) {
            const itemNum = Number(olMatch[1]);
            const itemBody = renderInlineMarkdown(olMatch[2]);
            const subItems = [];
            i += 1;
            while (i < lines.length && /^\s{2,}[-*•]\s+/.test(lines[i])) {
              const subMatch = lines[i].trim().match(/^[-*•]\s+(.*)$/);
              if (subMatch) subItems.push(`<li>${renderInlineMarkdown(subMatch[1])}</li>`);
              i += 1;
            }
            const subHtml = subItems.length
              ? `<ul class="md-ul md-sublist">${subItems.join("")}</ul>`
              : "";
            items.push(`<li value="${itemNum}">${itemBody}${subHtml}</li>`);
          } else if (!lineTrim && i + 1 < lines.length && /^\d+\.\s+/.test(lines[i + 1].trim())) {
            i += 1;
          } else {
            break;
          }
        }
        blocks.push(`<ol class="md-ol">${items.join("")}</ol>`);
        continue;
      }

      // Unordered List
      if (/^[-*•]\s+/.test(trimmed)) {
        const items = [];
        while (i < lines.length) {
          const lineTrim = lines[i].trim();
          const ulMatch = lineTrim.match(/^[-*•]\s+(.*)$/);
          if (ulMatch) {
            items.push(`<li>${renderInlineMarkdown(ulMatch[1])}</li>`);
            i += 1;
          } else {
            break;
          }
        }
        blocks.push(`<ul class="md-ul">${items.join("")}</ul>`);
        continue;
      }

      // Standard Paragraph
      const pLines = [];
      while (
        i < lines.length &&
        lines[i].trim() !== "" &&
        !lines[i].trim().startsWith("```") &&
        !/^(?:-{3,}|\*{3,}|_{3,})$/.test(lines[i].trim()) &&
        !/^(#{1,4})\s+/.test(lines[i].trim()) &&
        !lines[i].trim().startsWith(">") &&
        !/^\d+\.\s+/.test(lines[i].trim()) &&
        !/^[-*•]\s+/.test(lines[i].trim()) &&
        !(lines[i].trim().includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1]))
      ) {
        pLines.push(renderInlineMarkdown(lines[i].trim()));
        i += 1;
      }
      if (pLines.length) {
        blocks.push(`<p class="md-p">${pLines.join(" ")}</p>`);
      } else {
        i += 1;
      }
    }

    return blocks.join("\n");
  }

  function updateStatusUI(statusObj) {
    const running = Boolean(statusObj && statusObj.is_running);
    state.isRunning = running;

    const pill = document.getElementById("benchmark-status-pill");
    const statusText = document.getElementById("benchmark-status-text");
    const rerunBtn = document.getElementById("rerun-btn");
    const rerunLabel = document.getElementById("rerun-btn-label");
    const iconPlay = document.getElementById("rerun-icon-play");
    const iconSpin = document.getElementById("rerun-icon-spin");
    const cancelBtn = document.getElementById("cancel-run-btn");
    const termMeta = document.getElementById("terminal-meta-status");

    pill.className = "eval-status-pill";
    if (running) {
      pill.classList.add("is-running");
      const elapsedStr = statusObj.elapsed_sec
        ? ` (${statusObj.elapsed_sec.toFixed(0)}s)`
        : "";
      statusText.textContent = `Running${elapsedStr}...`;
      rerunBtn.disabled = true;
      rerunLabel.textContent = "Running...";
      iconPlay.style.display = "none";
      iconSpin.style.display = "inline-block";
      cancelBtn.hidden = false;
      termMeta.textContent = `PID ${statusObj.pid || "active"} · Running`;
    } else {
      rerunBtn.disabled = false;
      rerunLabel.textContent = "Re-run Evaluation";
      iconPlay.style.display = "inline-block";
      iconSpin.style.display = "none";
      cancelBtn.hidden = true;

      const st = (statusObj && statusObj.status) || "idle";
      if (st === "completed") {
        pill.classList.add("is-completed");
        statusText.textContent = "Completed";
        termMeta.textContent = "Completed (exit 0)";
      } else if (st === "failed" || st === "timeout" || st === "cancelled") {
        pill.classList.add("is-failed");
        statusText.textContent = st.charAt(0).toUpperCase() + st.slice(1);
        termMeta.textContent = st.toUpperCase();
      } else {
        statusText.textContent = "Ready";
        termMeta.textContent = "Idle";
      }
    }
  }

  function renderRunHistoryBar() {
    const container = document.getElementById("run-history-bar");
    if (!state.history.length) {
      container.innerHTML =
        '<div style="font-size:12.5px;color:#64748b;">No benchmark runs found.</div>';
      return;
    }

    container.innerHTML = state.history
      .map((run) => {
        const isActive = run.filename === state.activeFilename;
        const rate = Number(run.pass_rate_pct || 0);
        const badgeClass =
          rate >= 95 ? "high" : rate >= 80 ? "medium" : "low";
        const sdr = Number(run.calculation_delegation_rate_pct || 0).toFixed(1);
        const nar = Number(run.numerical_accuracy_rate_pct || 0).toFixed(1);
        const rgr = Number(run.rag_grounding_rate_pct || 0).toFixed(1);
        const rfr = Number(run.rag_retrieval_fidelity_rate_pct || 0).toFixed(1);
        const pcr = Number(run.protocol_compliance_rate_pct || 0).toFixed(1);
        const lat = Number(run.avg_latency_sec || 0).toFixed(2);
        const conc = Number(run.concurrency || 1);

        return `
          <div class="run-chip-wrap">
            <button type="button" class="run-chip ${isActive ? "active" : ""}" role="tab"
              aria-selected="${isActive ? "true" : "false"}"
              data-filename="${escapeHtml(run.filename)}">
              <span class="run-chip-time">${escapeHtml(run.display_timestamp)}</span>
              <span class="run-pass-badge ${badgeClass}">
                ${rate.toFixed(1)}% (${run.passed_scenarios}/${run.total_scenarios})
              </span>
            </button>
            <div class="run-chip-tooltip" role="tooltip">
              <div class="tooltip-title">Executive Summary · ${escapeHtml(run.display_timestamp)}</div>
              <div class="tooltip-row"><span>Pass Rate</span><strong class="pass-val">${rate.toFixed(1)}% (${run.passed_scenarios}/${run.total_scenarios})</strong></div>
              <div class="tooltip-row"><span>Calc Delegation (SDR)</span><strong>${sdr}%</strong></div>
              <div class="tooltip-row"><span>Numerical Accuracy (NAR)</span><strong>${nar}%</strong></div>
              <div class="tooltip-row"><span>RAG Grounding</span><strong>${rgr}%</strong></div>
              <div class="tooltip-row"><span>Retrieval Fidelity</span><strong>${rfr}%</strong></div>
              <div class="tooltip-row"><span>Protocol Compliance</span><strong>${pcr}%</strong></div>
              <div class="tooltip-row"><span>Avg Latency / Threads</span><strong>${lat}s · ${conc}x</strong></div>
            </div>
          </div>
        `;
      })
      .join("");

    container.querySelectorAll(".run-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        const fname = btn.getAttribute("data-filename");
        if (fname && fname !== state.activeFilename) {
          loadRunDetails(fname);
        }
      });
    });
  }

  function matchesFilter(sc, filter) {
    if (filter === "all") return true;
    if (filter === "passed") return sc.executed !== false && sc.passed === true;
    if (filter === "failed") return sc.executed !== false && sc.passed === false;
    if (filter === "calculation") return sc.category === "calculation";
    if (filter === "rag_grounding") return sc.category === "rag_grounding";
    return true;
  }

  function renderMatrixGrid() {
    const grid = document.getElementById("scenario-matrix-grid");
    if (!state.matrix.length) {
      grid.innerHTML = "";
      return;
    }

    grid.innerHTML = state.matrix
      .map((sc) => {
        const isSelected = sc.id === state.selectedScenarioId;
        const executed =
          sc.executed !== false &&
          sc.passed !== null &&
          sc.passed !== undefined;
        let statusClass = "is-unrun";
        if (executed) {
          statusClass = sc.passed ? "is-pass" : "is-fail";
        }
        const dimmedClass = matchesFilter(sc, state.activeFilter)
          ? ""
          : "is-dimmed";

        const tools = Array.isArray(sc.tools_called) ? sc.tools_called : [];
        const hasSearch = tools.some(
          (t) => t === "search_documents" || t === "internet_search"
        );
        const hasCalc = tools.some(
          (t) => t === "calculate_decimal" || t === "run_nav_audit"
        );

        const elapsedStr = executed
          ? `${Number(sc.elapsed_sec || 0).toFixed(1)}s`
          : "—";
        const tooltip = `${sc.short_code} (${sc.id}): ${sc.title}${
          executed ? " · " + elapsedStr : ""
        }`;

        return `
          <button type="button"
            class="scenario-cell ${statusClass} ${dimmedClass} ${isSelected ? "active" : ""}"
            data-scenario-id="${escapeHtml(sc.id)}"
            title="${escapeHtml(tooltip)}"
            aria-label="${escapeHtml(tooltip)}">
            <div class="cell-top">
              <span class="cell-code">${escapeHtml(sc.short_code)}</span>
              <span class="cell-tools">
                ${hasSearch ? `<span class="cell-tool-icon">${SVG_SEARCH}</span>` : ""}
                ${hasCalc ? `<span class="cell-tool-icon">${SVG_CALC}</span>` : ""}
              </span>
            </div>
            <div class="cell-bottom">
              <span>${escapeHtml(sc.id)}</span>
              <span>${escapeHtml(elapsedStr)}</span>
            </div>
          </button>
        `;
      })
      .join("");

    grid.querySelectorAll(".scenario-cell").forEach((cell) => {
      cell.addEventListener("click", () => {
        const sid = cell.getAttribute("data-scenario-id");
        if (sid) {
          state.selectedScenarioId = sid;
          renderMatrixGrid();
          renderTraceInspector();
        }
      });
    });
  }

  function renderTraceInspector() {
    const container = document.getElementById("trace-inspector-content");
    const sc =
      state.matrix.find((item) => item.id === state.selectedScenarioId) ||
      state.matrix[0];
    if (!sc) {
      container.innerHTML = "";
      return;
    }

    const retrievedDocs = Array.isArray(sc.retrieved_documents)
      ? sc.retrieved_documents
      : [];
    const searchOutputs = Array.isArray(sc.search_tool_outputs)
      ? sc.search_tool_outputs
      : [];
    const toolCalls = Array.isArray(sc.tool_calls) ? sc.tool_calls : [];
    const calcOutputs = Array.isArray(sc.calc_tool_outputs)
      ? sc.calc_tool_outputs
      : [];
    const numAudit = Array.isArray(sc.number_audit) ? sc.number_audit : [];
    const failureReasons = Array.isArray(sc.failure_reasons)
      ? sc.failure_reasons
      : [];
    const splitResp = splitReasoningAndConclusion(sc.final_text || "");

    // Only render Failure Diagnostics when test failed
    const failureHtml =
      sc.passed === false && failureReasons.length > 0
        ? `
          <div class="failure-diagnostics-box">
            <h4>Failure Diagnostics</h4>
            <ul class="failure-list">
              ${failureReasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}
            </ul>
          </div>
        `
        : "";

    const retrievedListHtml = retrievedDocs.length
      ? `<div class="doc-list">
          ${retrievedDocs
            .map(
              (doc) => `
                <div class="doc-row">
                  <code>${escapeHtml(doc.file_name || doc.doc_id || "Document")}</code>
                  <span>${escapeHtml(doc.section || "")}</span>
                </div>
              `
            )
            .join("")}
        </div>`
      : '<div style="font-size:12px;color:#64748b;">No corpus documents retrieved for this scenario.</div>';

    const toolCallsHtml = toolCalls.length
      ? toolCalls
          .map(
            (tc) =>
              `<pre class="tool-args-pre"><strong>${escapeHtml(
                tc.name || "tool"
              )}:</strong> ${escapeHtml(JSON.stringify(tc.args || {}, null, 2))}</pre>`
          )
          .join("")
      : '<div style="font-size:12px;color:#64748b;">No symbolic tools invoked.</div>';

    const calcOutputsHtml = calcOutputs.length
      ? calcOutputs
          .map((out) => `<pre class="calc-output-box">${escapeHtml(out)}</pre>`)
          .join("")
      : "";

    const numAuditHtml = numAudit.length
      ? `
        <div class="eval-table-wrap">
          <table class="eval-table" aria-label="Quantitative Number Audit">
            <thead>
              <tr>
                <th>Expected</th>
                <th>Tool Output</th>
                <th>Final Answer</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              ${numAudit
                .map(
                  (row) => `
                    <tr>
                      <td><strong>${formatNum(row.expected)}</strong></td>
                      <td>${row.in_tool ? "✓" : "✗"}</td>
                      <td>${row.in_final ? "✓" : "✗"}</td>
                      <td>${escapeHtml(row.status || "")}</td>
                    </tr>
                  `
                )
                .join("")}
            </tbody>
          </table>
        </div>
      `
      : "";

    container.innerHTML = `
      <div class="trace-header">
        <div class="trace-title-row">
          <span class="trace-code-tag">${escapeHtml(sc.short_code)} · ${escapeHtml(sc.id)}</span>
          <h2>${escapeHtml(sc.title)}</h2>
        </div>
        <span class="trace-meta-time">${Number(sc.elapsed_sec || 0).toFixed(2)}s</span>
      </div>

      <blockquote class="prompt-quote-box"><strong>Question:</strong> ${escapeHtml(sc.prompt)}</blockquote>

      ${failureHtml}

      <div class="trace-grid-2col">
        <!-- Block 1: RAG Retrieval -->
        <div class="trace-block">
          <h3 class="trace-block-title">
            <span>RAG Retrieval (${retrievedDocs.length} chunks)</span>
          </h3>
          ${retrievedListHtml}
          ${
            searchOutputs.length
              ? `
                <details class="trace-collapsible">
                  <summary>
                    <span>Raw Retrieved Chunks (${searchOutputs.length})</span>
                    <span>▼</span>
                  </summary>
                  <pre class="trace-raw-pre">${escapeHtml(
                    searchOutputs.join("\n\n----------------------------------------\n\n")
                  )}</pre>
                </details>
              `
              : ""
          }
        </div>

        <!-- Block 2: Tool Calls & Quantitative Audit -->
        <div class="trace-block">
          <h3 class="trace-block-title">
            <span>Tool Calls &amp; Decimal Engine</span>
          </h3>
          ${toolCallsHtml}
          ${calcOutputsHtml}
          ${numAuditHtml}
        </div>
      </div>

      <!-- Agent Output -->
      ${
        splitResp.reasoning
          ? `
            <details class="trace-collapsible">
              <summary>
                <span>[REASONING] Trace</span>
                <span>▼</span>
              </summary>
              <div class="response-pane-content md-prose">${renderMarkdown(splitResp.reasoning)}</div>
            </details>
          `
          : ""
      }
      <details class="trace-collapsible">
        <summary>
          <span>[CONCLUSION] Agent Response</span>
          <span>▼</span>
        </summary>
        <div class="response-pane-content md-prose">${renderMarkdown(
          splitResp.conclusion || sc.final_text || "No response recorded."
        )}</div>
      </details>
    `;
  }

  function applyRunData(data, fallbackFilename) {
    if (!data) return;
    state.activeFilename = data.filename || fallbackFilename || state.activeFilename;
    state.matrix =
      Array.isArray(data.matrix) && data.matrix.length
        ? data.matrix
        : data.results || [];

    const hasCurrent = state.matrix.some(
      (s) => s.id === state.selectedScenarioId && s.executed !== false
    );
    if (!hasCurrent) {
      const firstExec =
        state.matrix.find((s) => s.executed !== false) || state.matrix[0];
      state.selectedScenarioId = firstExec ? firstExec.id : null;
    }

    renderRunHistoryBar();
    renderMatrixGrid();
    renderTraceInspector();
  }

  async function loadRunDetails(filename) {
    try {
      const resp = await fetch(
        `/api/evaluation/run/${encodeURIComponent(filename)}`
      );
      if (!resp.ok) {
        throw new Error(`Failed to load run ${filename} (HTTP ${resp.status})`);
      }
      const data = await resp.json();
      applyRunData(data, filename);
    } catch (err) {
      console.error(err);
    }
  }

  async function refreshHistoryAndSelect(preferredFilename) {
    try {
      const resp = await fetch("/api/evaluation/history?limit=5");
      if (!resp.ok) return;
      const payload = await resp.json();
      state.history = Array.isArray(payload.runs) ? payload.runs : [];
      updateStatusUI(payload.status || {});

      if (state.history.length > 0) {
        const target =
          preferredFilename || state.activeFilename || state.history[0].filename;
        const exists = state.history.some((r) => r.filename === target);
        const chosen = exists ? target : state.history[0].filename;
        state.activeFilename = chosen;
        renderRunHistoryBar();
        await loadRunDetails(chosen);
      } else {
        renderRunHistoryBar();
      }
    } catch (err) {
      console.error("Failed to load evaluation history:", err);
    }
  }

  function appendTerminalLine(line) {
    const out = document.getElementById("terminal-output");
    if (!out) return;
    out.textContent += (out.textContent ? "\n" : "") + line;
    out.scrollTop = out.scrollHeight;
  }

  function connectLogStream() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    const drawer = document.getElementById("terminal-drawer");
    drawer.hidden = false;
    const out = document.getElementById("terminal-output");
    out.textContent = "";

    const es = new EventSource("/api/evaluation/stream");
    state.eventSource = es;

    es.addEventListener("status", (evt) => {
      try {
        updateStatusUI(JSON.parse(evt.data));
      } catch (_) {}
    });

    es.addEventListener("log", (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.line !== undefined) {
          appendTerminalLine(data.line);
        }
      } catch (_) {}
    });

    es.addEventListener("done", async (evt) => {
      es.close();
      state.eventSource = null;
      let latestFile = null;
      try {
        const doneData = JSON.parse(evt.data);
        updateStatusUI(doneData);
        latestFile = doneData.latest_filename || null;
      } catch (_) {
        updateStatusUI({ is_running: false, status: "completed" });
      }
      await refreshHistoryAndSelect(latestFile);
    });

    es.onerror = () => {
      es.close();
      state.eventSource = null;
      fetch("/api/evaluation/status")
        .then((r) => r.json())
        .then((st) => updateStatusUI(st))
        .catch(() => {});
    };
  }

  async function handleRerunClick() {
    if (state.isRunning) return;
    const drawer = document.getElementById("terminal-drawer");
    drawer.hidden = false;
    updateStatusUI({ is_running: true, status: "running" });

    try {
      const resp = await fetch("/api/evaluation/re-run", { method: "POST" });
      if (resp.status === 409) {
        appendTerminalLine(
          "[INFO] Benchmark is already running — attaching to active stream..."
        );
      } else if (!resp.ok) {
        appendTerminalLine(
          `[ERROR] Failed to start benchmark (HTTP ${resp.status}).`
        );
        updateStatusUI({ is_running: false, status: "failed" });
        return;
      }
      connectLogStream();
    } catch (err) {
      appendTerminalLine(`[ERROR] Network error starting benchmark: ${err}`);
      updateStatusUI({ is_running: false, status: "failed" });
    }
  }

  async function handleCancelClick() {
    try {
      const resp = await fetch("/api/evaluation/cancel", { method: "POST" });
      if (resp.ok) {
        const st = await resp.json();
        updateStatusUI(st);
      }
    } catch (err) {
      console.error("Cancel error:", err);
    }
  }

  function bindClick(id, handler) {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", handler);
  }

  // Wire Controls
  bindClick("rerun-btn", handleRerunClick);
  bindClick("cancel-run-btn", handleCancelClick);

  const drawer = document.getElementById("terminal-drawer");
  bindClick("toggle-terminal-btn", () => {
    if (drawer) drawer.hidden = !drawer.hidden;
  });
  bindClick("collapse-terminal-btn", () => {
    if (drawer) drawer.hidden = true;
  });
  bindClick("clear-terminal-btn", () => {
    const out = document.getElementById("terminal-output");
    if (out) out.textContent = "";
  });

  document.querySelectorAll(".filter-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      document
        .querySelectorAll(".filter-pill")
        .forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.activeFilter = btn.getAttribute("data-filter") || "all";
      renderMatrixGrid();
    });
  });

  function bootstrapInitialState() {
    const initScript = document.getElementById("eval-initial-state");
    if (!initScript || !initScript.textContent) return false;
    try {
      const payload = JSON.parse(initScript.textContent);
      if (Array.isArray(payload.runs) && payload.runs.length > 0) {
        state.history = payload.runs;
        updateStatusUI(payload.status || {});
        if (payload.initial_run) {
          applyRunData(payload.initial_run, payload.runs[0].filename);
          return true;
        }
      }
    } catch (err) {
      console.warn("Initial state bootstrap fallback to API:", err);
    }
    return false;
  }

  // Initial Load: Hydrate synchronously from server bootstrap, then sync via API
  const hydrated = bootstrapInitialState();
  if (!hydrated) {
    refreshHistoryAndSelect().then(() => {
      if (state.isRunning) {
        connectLogStream();
      }
    });
  } else if (state.isRunning) {
    connectLogStream();
  }
})();
