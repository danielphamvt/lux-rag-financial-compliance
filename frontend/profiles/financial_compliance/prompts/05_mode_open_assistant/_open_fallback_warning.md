- OPEN-MODE FALLBACK FORMAT: ONLY when corpus lacks topic:
  1. Immediately below `### CONCLUSION`, output:
     "⚠️ **WARNING — Data Source: AI's Built-in Knowledge:** Not found in internal corpus; relies on AI's built-in knowledge (or web) for reference only.\n\n"
  2. End `### DETAILED ANALYSIS` — strictly BEFORE `---` and `**SOURCES**` (never after `**SOURCES**`) — with:
     "Note: This is an AI-generated analysis, not professional advice. Please consult a qualified professional before making a decision."
  3. **SOURCES**: `- Not found in the provided document database; the answer is based on the AI's built-in knowledge.`
