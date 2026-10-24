# Resume NLP Lab — Next-Gen AI Resume & ATS Intelligence

A multi-paradigm NLP and machine learning platform that evaluates resumes against job descriptions, diagnoses ATS structural formatting, and benchmarks matching algorithms against human ground truth.

---

## 🥊 Why Resume NLP Lab vs. Jobscan & SkillSyncer?

Commercial resume checkers like **Jobscan** and **SkillSyncer** are widely used, but have significant shortcomings. Here is how **Resume NLP Lab** does things differently:

| Feature / Dimension | Jobscan & SkillSyncer | Resume NLP Lab |
| :--- | :--- | :--- |
| **Core Matching Engine** | **Black-box keyword counter**: Counts literal word frequency; rewards exact string matches. | **Multi-Paradigm NLP Suite**: Evaluates (1) Alias-aware Skill Extraction, (2) Transparent TF-IDF Cosine Vector Space, (3) Okapi BM25 Search Ranking, (4) Semantic Levenshtein Edit Distance, and (5) Adversarial Anti-Stuffing Audit. |
| **Vulnerability to Keyword Stuffing** | **High**: Candidates can stuff buzzwords into footers or lists to artificially trigger 80%+ scores. | **Low**: Built-in **Adversarial Keyword Stuffing Detector** audits keyword-to-verb ratios, buzzword density anomalies, and list dumping to penalize gaming attempts. |
| **Synonym & Acronym Intelligence** | **Rigid**: Often penalizes candidates if exact phrasing differs (e.g., treats `k8s` and `Kubernetes` as distinct). | **Built-in Alias & Taxonomy Engine**: Automatically resolves industry aliases (`k8s` ↔ `Kubernetes`, `AWS` ↔ `Amazon Web Services`, `NLP` ↔ `Natural Language Processing`, `JS` ↔ `JavaScript`). |
| **Explainability & Transparency** | **Opaque black-box**: Single score with proprietary formulas designed to drive paid subscriptions. | **Full Mathematical Decomposition**: Displays sub-scores, weights, cosine similarities, and runtime latency profiling in milliseconds. |
| **Scientific Benchmark Lab** | **None**: No way to validate algorithm accuracy against real hiring decisions. | **Open Evaluation Lab**: Benchmarks matching methods against human ground truth, calculating **Accuracy, Precision, Recall, F1-Score, and Spearman Rank Correlation**. |
| **Cost & Data Privacy** | **Paid SaaS Subscription** (\$40–\$50/month); uploads confidential candidate PII to cloud databases. | **100% Free, Open Source & Local**: Runs completely offline on your machine with zero third-party data tracking. |
| **ATS Structural Hygiene** | Basic surface checks behind paywalls. | **Comprehensive ATS Diagnostic**: Audits section completeness, action verb density, quantified impact metrics (\$, %, scale), and word count health. |

---

## 🌟 Key Features

### 1. Multi-Method Analyzer
- **Input Modes**: Paste raw text or upload selectable PDF documents.
- **⚡ 1-Click Demo Presets**: Includes pre-configured profiles (AI/ML Engineer, Full-Stack Developer, Domain Mismatch) for instant testing.
- **Categorized Skill Breakdown**: Grouped into *Programming Languages*, *Frameworks*, *Data & AI*, *Cloud & DevOps*, *Databases*, and *Soft Skills / Methodologies*.
- **ATS Resume Diagnostic**: Evaluates section completeness, impact action verbs, quantified metrics, and delivers targeted improvement tips.
- **JSON Export**: Download complete analysis results for candidate tracking.

### 2. Scientific Evaluation Lab
- Upload a labeled CSV of resume–job pairs with human relevance ratings (`0` = not relevant, `1` = relevant).
- Compare the four matching methods on **Accuracy, Precision, Recall, and F1**.
- Calculate **Spearman Rank Correlation** to determine which algorithm best preserves human hiring judgment.
- Inspect per-example false positives and false negatives to diagnose matching edge cases.
- Export benchmark summaries and prediction matrices to CSV.

---

## 🚀 Setup & Launch (Windows)

### Requirements
- Python 3.10 or 3.11
- Internet connection on the first run (to cache `sentence-transformers/all-MiniLM-L6-v2`)

### Quick Launch
Run the Streamlit application directly:

```powershell
streamlit run app.py
```

Or run via the automated batch script:
```cmd
setup_and_run.bat
```

### Running Unit Tests
Execute the test suite to verify matching logic, alias resolution, and ATS diagnostics:

```powershell
python -m unittest -v test_resume_nlp.py
```

---

## 📊 The Core Information Retrieval & NLP Methods

1. **Keyword / Skill Coverage**:
   Boundary-aware extraction across a 300+ categorized skill taxonomy with alias resolution, plural handling, and hyphen normalization.
2. **Transparent TF-IDF Cosine Similarity**:
   Measures document-level lexical overlap using Porter-stemmed unigrams, stop-word removal, and inner-product vector cosine geometry with a live word-by-word calculation table.
3. **Okapi BM25 Ranking Score**:
   Industry search standard algorithm incorporating document length normalization ($b=0.75$) and term frequency saturation ($k_1=1.5$).
4. **Semantic Levenshtein Similarity**:
   Fuzzy dynamic programming edit-distance measuring character insertions, deletions, and substitutions to capture morphological variations (e.g. `engineer` ↔ `engineering`).
5. **Adversarial Anti-Stuffing Audit**:
   Audits the grammatical coupling between keywords and past-tense action verbs to catch candidates who dump keywords to fool ATS systems.
