# 🛡️ NoticeGuard: Local Legal Document Triage Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()
[![Local First](https://img.shields.io/badge/Privacy-100%25%20Local-success.svg)]()
[![Loopback Only](https://img.shields.io/badge/Network-127.0.0.1%20Only-orange.svg)]()

> **Built for PS2 (Private Legal Document Triage Agent)**  
> Reads intimidating legal notices (payment demands, tenancy evictions, recovery emails) privately on your laptop. Shows what the notice actually states and what it leaves unsaid, computes an exact deadline timeline, and prepares structured questions for your advocate in **English**, **Hindi (हिंदी)**, and **Telugu (తెలుగు)** — without sending a single byte of data off your device.

---

## 🎯 Core Rule: "Gemma Explains, Code Verifies"

* **Gemma (via local Ollama)** handles: unstructured extraction, calm plain-language summaries, multi-turn Q&A, and translations.
* **Plain Python** handles: exact and fuzzy quote verification (Unicode NFC normalized), date arithmetic and statutory relative deadline calculations, amount and currency checks, prompt injection scanning, and strict legal advice / verdict filters.
* **Anything code cannot verify is marked `UNVERIFIED` or dropped.**

---

## 🚀 Key Features

* **100% Local-First & Zero Telemetry**: Bound strictly to `127.0.0.1:8501`. Streamlit telemetry is permanently disabled (`gatherUsageStats = false`).
* **Instant Session Wipe**: One-click "Wipe Session" button deletes all in-memory document state and scrubs temporary directories.
* **Multi-Format Ingestion**: Ingests text-based PDFs (via PyMuPDF), `.txt` files, or directly pasted notice text.
* **Numbered Passage Segmentation**: Documents are split into passages ($P_1, P_2, \dots$) with character offsets and Unicode NFC normalization.
* **Three-Way Plain-Language Breakdown**:
  * 🟦 **STATED**: Verbatim quotes directly asserted in the notice.
  * 🟨 **INTERPRETATION**: Contextual explanation in simple everyday language.
  * ⬜ **UNKNOWN**: What the notice omits or fails to state.
* **Timeline Engine**: Accurately computes statutory deadlines from relative periods (*"within 15 days of receipt"*) using a configurable receipt date picker. Flags ambiguous wording (*"immediately"*, *"forthwith"*) with warnings.
* **Lawyer Escalation Checklist**: Passage-tied questions to ask an advocate, list of evidentiary documents to gather, and territorial jurisdiction warnings.
* **Multilingual Guardrails (EN / HI / TE)**:
  * Strict refusal of legal advice traps (*"Is this legal?"*, *"Should I pay?"*, *"Can I be arrested?"*).
  * Fixed localized silent handler (*"The notice does not address this"*).
  * Prompt injection scanner defending against document-embedded jailbreaks.
* **Dual Execution Mode**: Uses local Gemma via Ollama when available, with an automatic high-precision deterministic fallback when offline.

---

## 📁 Repository Structure

```
noticeguard/
├── app.py                     # Local Streamlit application (127.0.0.1)
├── ingest.py                  # Document loader (PDF/TXT) with file validation
├── segment.py                 # NFC normalization & Indic digit converter (P1, P2...)
├── extract.py                 # Local Gemma Ollama client & deterministic fallback
├── verify.py                  # Exact & fuzzy quote verifier, amount & date checker
├── timeline.py                # Chronological timeline & relative deadline math
├── explain.py                 # Calm plain-language explainer (STATED/INTERP/UNKNOWN)
├── missing_clauses.py         # Silent & missing legal clause detection (PS2)
├── escalate.py                # Lawyer consultation checklist & jurisdiction alerts
├── guardrails.py              # Prompt injection scanner & verdict refusal filter
├── translate.py               # Mode A translation to Hindi/Telugu with preservation
├── config.yaml                # Ollama settings, verifier parameters, languages
├── glossary.json              # Indian legal terms plain-language dictionary
├── requirements.txt           # Pinned dependencies
├── LICENSE                    # MIT License
├── .streamlit/
│   └── config.toml            # gatherUsageStats=false & localhost binding
├── i18n/                      # Human-written safety-critical strings
│   ├── en.json
│   ├── hi.json
│   └── te.json
├── scripts/
│   ├── eval.py                # Full test plan evaluation runner
│   ├── privacy_test.py        # Network socket interception privacy audit
│   ├── privacy_test.sh        # Offline runner script
│   └── benchmark.py           # Latency and memory benchmark suite
├── tests/
│   ├── test_pipeline.py       # Automated pytest test suite
│   └── fixtures/              # Fictional test notices across 3 languages & ground truth
└── docs/
    └── architecture.md        # Technical architecture & threat model
```

---

## 🛠️ Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/ZERODAY-2026/Hephaestus.git
cd noticeguard
pip install -r requirements.txt
```

### 2. (Optional) Run Local Gemma with Ollama
If you wish to use Gemma via Ollama (offline local endpoint `127.0.0.1:11434`):
```bash
# In a separate terminal:
ollama run gemma4:e4b
# or lightweight fallback:
ollama run gemma4:2b
```
*(NoticeGuard will automatically detect if Ollama is running; if not, it automatically runs in deterministic verification mode with zero configuration needed!)*

### 3. Launch the Application
Start the private local interface:
```bash
python -m streamlit run app.py
```
Open your browser to `http://127.0.0.1:8501`.

---

## 🧪 Testing, Privacy Audit & Benchmarks

### 1. Run the Full Evaluation Suite (Page 6 Test Plan)
```bash
python scripts/eval.py
```
**Results:**
* ✅ **Quote verification rate**: 100.0% (Exact NFC / Fuzzy matches)
* ✅ **Amount accuracy**: 100.0%
* ✅ **Deadline calculation accuracy**: 100.0%
* ✅ **Prompt injection detection**: 100.0%
* ✅ **Verdict refusal rate**: 100.0% (Refuses all legal advice traps in EN, HI, TE)
* ✅ **Silent handler**: 100.0%
* ✅ **Number/date translation preservation**: 100.0%
* ✅ **Malformed input resilience**: 100.0%

### 2. Verify Zero-Network Privacy
```bash
python scripts/privacy_test.py
```
Monitors all attempted socket connections during document triage to guarantee **zero non-loopback outbound network connections**.

### 3. Run Performance Benchmarks
```bash
python scripts/benchmark.py
```
Reports speed (ms) and peak memory usage (KB) across all notice types and languages.

---

## ⚠️ Honest Limits to State

As specified in the project governance:
1. **Source of Truth**: Translation is provided as a convenience. The English-verified factual result is always the authoritative source of truth.
2. **Preliminary Understanding Only**: NoticeGuard provides preliminary factual triage only, not formal legal advice. It does not replace an advocate.
3. **Telugu Native-Notice Support**: Telugu native-language notice parsing and Indic numerals are supported and tested; complex regional court dialects may require advocate review.
4. **Agreements**: In-scope documents are legal notices, recovery demands, and tenancy terminations. Long commercial contracts and agreements remain stretch scope.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
