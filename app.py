import json
from pathlib import Path

import pandas as pd
import streamlit as st

from resume_nlp import (
    analyze_pair,
    extract_text_from_pdf,
    validate_pair,
    load_evaluation_data,
    evaluate_dataset,
    SAMPLE_PROFILES,
)

st.set_page_config(
    page_title="Next-Gen Resume Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished appearance
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4361ee;
        margin-bottom: 10px;
    }
    .badge-matched {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 2px;
    }
    .badge-missing {
        background-color: #f8d7da;
        color: #842029;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 AI Resume & ATS Intelligence System")
st.write(
    "A transparent, multi-algorithm NLP evaluation platform featuring **Okapi BM25 Ranking**, "
    "**TF-IDF Vector Space**, **Semantic Levenshtein Distance**, and **Adversarial Anti-Stuffing Audit**."
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Analyzer & ATS Diagnostic", "Evaluation Lab (Benchmarks)"], index=0)
    
    st.divider()
    st.subheader("💡 Core Algorithms Explained")
    
    st.markdown("#### 1. Okapi BM25 Ranking")
    st.markdown(
        """
        - **What it is:** Industry-standard probabilistic ranking function (used in modern search engines & Elasticsearch).
        - **Term Saturation ($k_1=1.5$):** Diminishing returns for repeating a keyword. Mentioning a skill 20 times doesn't give 20× the score.
        - **Length Normalization ($b=0.75$):** Penalizes document bloat, ensuring concise 1-page resumes aren't unfairly beaten by 5-page keyword dumps.
        """
    )
    
    st.markdown("#### 2. Levenshtein Edit Distance")
    st.markdown(
        """
        - **What it is:** Character-level metric measuring the minimum edits (insertions, deletions, substitutions) between words.
        - **Morphological Tolerance:** Bridges grammatical and inflectional variations like `engineer` ↔ `engineering` or `automate` ↔ `automation`.
        - **Spelling Resilience:** Overcomes rigid string mismatches and minor formatting variances that cause traditional scanners to fail.
        """
    )
    
    st.markdown("#### 3. Adversarial Anti-Stuffing")
    st.markdown(
        """
        - **What it is:** Adversarial NLP audit designed to expose keyword-stuffing tricks (such as white-text footers or raw list dumps).
        - **Verb-Keyword Coupling:** Checks whether skills are actively paired with past-tense action verbs (*architected*, *deployed*, *built*).
        - **Density Audit:** Flags excessive skill density and uncontextualized buzzword lists.
        """
    )
    st.caption("Privacy-First • Transparent Mathematical Scoring")

def document_input(label: str, key: str, default_val: str = "") -> str:
    mode = st.radio(
        f"{label} input mode",
        ["Text input", "Upload PDF"],
        horizontal=True,
        key=f"{key}_mode",
    )
    if mode == "Text input":
        return st.text_area(
            label,
            value=default_val,
            height=260,
            placeholder=f"Paste {label.lower()} here...",
            key=f"{key}_text",
        ).strip()

    uploaded = st.file_uploader(
        f"Upload {label.lower()} (selectable PDF)",
        type=["pdf"],
        key=f"{key}_file",
    )
    if not uploaded:
        return default_val
    try:
        text = extract_text_from_pdf(uploaded.getvalue())
        if not text.strip():
            st.warning(
                f"No selectable text found in this PDF. If it is a scanned image, please copy-paste the text."
            )
        return text
    except Exception as exc:
        st.error(f"Could not read PDF: {exc}")
        return default_val

# ==============================================================================
# PAGE 1: ANALYZER & ATS DIAGNOSTIC
# ==============================================================================
if page == "Analyzer & ATS Diagnostic":
    st.subheader("Interactive Resume vs. Job Description Matching")

    # Preset Selector for 1-click demonstration
    preset_choice = st.selectbox(
         "Quick Demo Presets (Select a pre-configured role or choose Custom)",
        ["Custom Input"] + list(SAMPLE_PROFILES.keys()),
        index=0,
        help="Select a realistic profile to test the system immediately without manual copy-pasting.",
    )

    preset_resume = ""
    preset_job = ""
    if preset_choice != "Custom Input":
        preset_resume = SAMPLE_PROFILES[preset_choice]["resume"]
        preset_job = SAMPLE_PROFILES[preset_choice]["job"]
        st.info(f"Loaded demo preset: **{preset_choice}**. Click 'Run Multi-Method Analysis' below to view results.")

    left, right = st.columns(2, gap="large")
    with left:
        resume = document_input("Resume Text", "resume", default_val=preset_resume)
    with right:
        job = document_input("Job Description", "job", default_val=preset_job)

    with st.expander("🔍 Preview extracted / pasted content"):
        a, b = st.columns(2)
        with a:
            st.markdown("**Resume Snippet**")
            st.text(resume[:1000] + ("..." if len(resume) > 1000 else "") if resume else "No resume content.")
        with b:
            st.markdown("**Job Description Snippet**")
            st.text(job[:1000] + ("..." if len(job) > 1000 else "") if job else "No job description content.")

    if st.button("🚀 Run Multi-Method Analysis & ATS Diagnostic", type="primary", use_container_width=True):
        errors = validate_pair(resume, job)
        if errors:
            for err in errors:
                st.error(err)
        else:
            try:
                with st.spinner("Analyzing resume across NLP algorithms and computing ATS hygiene audit..."):
                    result = analyze_pair(resume, job)
                st.session_state["last_result"] = result
                st.success("Analysis and ATS hygiene diagnostic completed successfully!")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("last_result")
    if result:
        st.divider()

        # ----------------------------------------------------------------------
        # 1. FIVE-METHOD COMPARISON & ANTI-STUFFING CARDS
        # ----------------------------------------------------------------------
        st.subheader("1. Multi-Algorithm NLP Comparison & Adversarial Audit")
        st.markdown(
            "Evaluation of the resume across five independent, mathematically rigorous NLP algorithms:"
        )

        scores = result["scores"]
        stuffing = result.get("keyword_stuffing_details", {})

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            "1. Skill Coverage",
            "N/A" if scores["keyword_coverage"] is None else f"{scores['keyword_coverage']:.1f}%",
            help="Fraction of required skills found in resume, with automatic alias/synonym and plural normalization.",
        )
        c2.metric(
            "2. TF-IDF Cosine",
            f"{scores['tfidf_similarity']:.1f}%",
            help="Stemmed n-gram term frequency and inverse document frequency cosine vector angle.",
        )
        c3.metric(
            "3. Okapi BM25 Ranking",
            f"{scores['bm25_score']:.1f}%",
            help="Industry-standard search ranking algorithm with document length normalization (b=0.75) and saturation (k1=1.5).",
        )
        c4.metric(
            "4. Levenshtein Sim",
            f"{scores['levenshtein_similarity']:.1f}%",
            help="Fuzzy edit-distance similarity measuring morphological and orthographic alignments.",
        )
        c5.metric(
            "5. Anti-Stuffing Authenticity",
            f"{scores['authenticity_score']:.0f}%",
            delta=stuffing.get("risk_level", "Authentic"),
            delta_color="normal" if scores["authenticity_score"] >= 80 else "inverse",
            help="Adversarial NLP audit detecting keyword dumping, white-text lists, and verb coupling.",
        )

        # Bar chart comparison
        score_df = pd.DataFrame([
            {"Method": "1. Skill Coverage", "Score (%)": scores["keyword_coverage"] or 0.0},
            {"Method": "2. TF-IDF Cosine", "Score (%)": scores["tfidf_similarity"]},
            {"Method": "3. Okapi BM25", "Score (%)": scores["bm25_score"]},
            {"Method": "4. Levenshtein Sim", "Score (%)": scores["levenshtein_similarity"]},
            {"Method": "5. Authenticity (Anti-Stuffing)", "Score (%)": scores["authenticity_score"]},
        ])
        st.bar_chart(score_df.set_index("Method"), height=290)

        # Adversarial Stuffing Callout
        if scores["authenticity_score"] < 80:
            st.warning(f"⚠️ **Adversarial Audit Alert [{stuffing.get('risk_level', 'Flagged')}]:** {stuffing.get('verdict')}")
            for flag in stuffing.get("flags", []):
                st.caption(f"• {flag}")
        else:
            st.success(f"🛡️ **Adversarial Audit Verified [{stuffing.get('risk_level', 'Authentic')}]:** {stuffing.get('verdict')}")

        # ----------------------------------------------------------------------
        # UNDER-THE-HOOD NLP INSPECTOR (EXACT MATHEMATICAL & ARCHITECTURAL BREAKDOWN)
        # ----------------------------------------------------------------------
        with st.expander("🔬 Mathematical Formulations & Execution Details", expanded=True):
            st.markdown(
                "Exact mathematical formulations, vector geometry, and execution breakdown for each evaluated method:"
            )
            
            uth = result.get("under_the_hood", {})
            tab_kw, tab_tfidf, tab_bm25, tab_lev, tab_stuffing = st.tabs([
                "1. Skill Coverage Math",
                "2. TF-IDF Vector Space",
                "3. Okapi BM25 Formulation",
                "4. Semantic Levenshtein Matrix",
                "5. Adversarial Anti-Stuffing Audit"
            ])
            
            with tab_kw:
                kw_data = uth.get("keyword", {})
                st.markdown("### Method 1: Boundary-Aware Skill Extraction & Coverage")
                st.latex(r"\text{Skill Coverage Score} = \frac{|\text{Matched Canonical Skills}|}{|\text{Detected Job Skills}|} \times 100\%")
                
                c_k1, c_k2, c_k3 = st.columns(3)
                c_k1.metric("Coverage Score", f"{scores['keyword_coverage']:.1f}%" if scores['keyword_coverage'] is not None else "N/A")
                c_k2.metric("Matched Skills", f"{len(result['matched_skills'])}")
                c_k3.metric("Required Job Skills", f"{result['job_skill_count']}")

                st.markdown("#### 📐 Step-by-Step Solving Breakdown:")
                matched_count = len(result['matched_skills'])
                job_count = result['job_skill_count']
                all_detected_skills = sorted(list(set(result['matched_skills'] + result['missing_skills'])))
                st.markdown(
                    f"""
                    1. **Detected Job Skills:** `{', '.join(all_detected_skills) if all_detected_skills else 'None detected'}`
                    2. **Matched Skills in Resume:** `{', '.join(result['matched_skills']) if result['matched_skills'] else 'None'}`
                    3. **Missing Skills:** `{', '.join(result['missing_skills']) if result['missing_skills'] else 'None (100% complete match!)'}`
                    4. **Arithmetic Substitution:** `({matched_count} / {job_count}) × 100 =` **`{scores['keyword_coverage']:.2f}%`**
                    """
                )
                st.info(
                    "💡 **Why Jobscan Fails Here:** Jobscan uses rigid string equality. If a job says 'prompt-engineering' and the resume says 'prompt engineering', or 'APIs' vs 'API', Jobscan rejects them as 0% matches. Our engine normalizes hyphens, aliases, and plural suffixes (`s`/`es`)."
                )
            
            with tab_tfidf:
                tfidf_data = result.get("tfidf_details", {})
                st.markdown("### Method 2: Term Frequency-Inverse Document Frequency (TF-IDF) Vector Space")
                
                st.latex(r"\text{TF-IDF Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{t \in V} \text{TF-IDF}(t, R) \times \text{TF-IDF}(t, J)}{\sqrt{\sum_{t \in V} \text{TF-IDF}(t, R)^2} \times \sqrt{\sum_{t \in V} \text{TF-IDF}(t, J)^2}} \times 100")
                
                # Metrics banner
                c_t1, c_t2, c_t3, c_t4 = st.columns(4)
                c_t1.metric("Resume Stemmed Tokens", f"{tfidf_data.get('resume_token_count', 0)}")
                c_t2.metric("Job Stemmed Tokens", f"{tfidf_data.get('job_token_count', 0)}")
                c_t3.metric("Total Vocabulary (|V|)", f"{tfidf_data.get('vocab_size', 0)}")
                c_t4.metric("Shared Stemmed Terms", f"{tfidf_data.get('shared_term_count', 0)}")

                st.markdown("#### 📐 Mathematical Vector Breakdown:")
                st.markdown(
                    f"""
                    - **Dot Product ($\mathbf{{u}} \cdot \mathbf{{v}}$):** `{tfidf_data.get('dot_product', 0.0):.6f}`
                    - **Resume Vector Magnitude ($\|\mathbf{{u}}\|_2$):** `{tfidf_data.get('resume_norm', 0.0):.6f}`
                    - **Job Vector Magnitude ($\|\mathbf{{v}}\|_2$):** `{tfidf_data.get('job_norm', 0.0):.6f}`
                    - **Calculation:** `({tfidf_data.get('dot_product', 0.0):.6f} / ({tfidf_data.get('resume_norm', 0.0):.6f} × {tfidf_data.get('job_norm', 0.0):.6f})) × 100 =` **`{tfidf_data.get('score', 0.0):.2f}%`**
                    """
                )

                terms_table = tfidf_data.get("terms_table", [])
                if terms_table:
                    st.markdown("#### 🔍 Term-by-Term TF-IDF Calculation Table:")
                    filter_view = st.radio(
                        "Filter Table View:",
                        [f"Shared Terms Only ({tfidf_data.get('shared_term_count', 0)} terms)", f"Complete Vocabulary ({len(terms_table)} terms)"],
                        horizontal=True,
                        key="tfidf_table_filter"
                    )

                    if "Shared Terms Only" in filter_view:
                        display_terms = [t for t in terms_table if t.get("is_shared")]
                    else:
                        display_terms = terms_table

                    df_display = pd.DataFrame(display_terms)
                    rename_cols = {
                        "term": "Stemmed Term",
                        "original_word": "Sample Word",
                        "resume_count": "Resume Count",
                        "resume_tf": "Resume TF",
                        "job_count": "Job Count",
                        "job_tf": "Job TF",
                        "df": "DF",
                        "idf": "IDF Weight",
                        "resume_tfidf": "Resume TF-IDF",
                        "job_tfidf": "Job TF-IDF",
                        "product": "Dot Product (u_i × v_i)",
                        "is_shared": "In Both Docs?"
                    }
                    df_display = df_display.rename(columns=rename_cols)
                    st.dataframe(df_display, hide_index=True, use_container_width=True)
                    st.caption("TF = Count / Total Doc Tokens • IDF = ln(3 / (DF + 1)) + 1 • Dot Product = Resume TF-IDF × Job TF-IDF")

            with tab_bm25:
                bm25_data = uth.get("bm25", {})
                st.markdown("### Method 3: Okapi BM25 Probabilistic Ranking Formulation")
                st.latex(r"\text{BM25}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}")
                
                c_b1, c_b2, c_b3, c_b4 = st.columns(4)
                c_b1.metric("BM25 Normalized Score", f"{bm25_data.get('score', 0.0):.1f}%")
                c_b2.metric("Raw BM25 Score", f"{bm25_data.get('raw_score', 0.0):.4f}")
                c_b3.metric("Doc Length (|D|)", f"{bm25_data.get('doc_length', 0)} tokens")
                c_b4.metric("Length Factor (K)", f"{bm25_data.get('k_factor', 1.5):.4f}")

                st.markdown("#### 📐 Step-by-Step Solving Breakdown:")
                st.markdown(
                    f"""
                    1. **Parameters Applied:** Term saturation `k1 = 1.5`, Length normalization `b = 0.75`, Benchmark document length `avgdl = {bm25_data.get('avgdl', 350.0)}`.
                    2. **Dynamic Length Factor ($K$):**
                       $$K = k_1 \\cdot \\left(1 - b + b \\cdot \\frac{{|D|}}{{\\text{{avgdl}}}}\\right) = 1.5 \\cdot \\left(0.25 + 0.75 \\cdot \\frac{{{bm25_data.get('doc_length', 0)}}}{{{bm25_data.get('avgdl', 350.0)}}}\\right) = \\mathbf{{{bm25_data.get('k_factor', 1.5):.4f}}}$$
                    3. **Term Saturated Component:** Each query term $t$ contributes $\\text{{IDF}}(t) \\cdot \\frac{{f(t, D) \\cdot 2.5}}{{f(t, D) + {bm25_data.get('k_factor', 1.5):.4f}}}$.
                    4. **Sum of Contributions:** Raw Score = `{bm25_data.get('raw_score', 0.0):.4f}` (Maximum theoretical score: `{bm25_data.get('max_score', 0.0):.4f}`).
                    5. **Normalized Score:** `({bm25_data.get('raw_score', 0.0):.4f} / {bm25_data.get('max_score', 0.0):.4f}) × 100 =` **`{bm25_data.get('score', 0.0):.2f}%`**
                    """
                )

                matched_bm25 = bm25_data.get("matched_terms", [])
                if matched_bm25:
                    st.markdown("#### 🔍 Term-by-Term BM25 Contributions Table:")
                    df_bm25 = pd.DataFrame(matched_bm25).rename(columns={
                        "term": "Query Term",
                        "resume_freq": "Resume Freq (f)",
                        "job_freq": "Job Query Freq (q)",
                        "idf": "IDF Weight",
                        "tf_component": "Saturated TF Component",
                        "bm25_contribution": "Term Contribution"
                    })
                    st.dataframe(df_bm25, hide_index=True, use_container_width=True)
                st.info(
                    "💡 **Why BM25 Beats Basic Counters:** Repeating a skill 20 times produces diminishing returns due to $k_1=1.5$ saturation, and 4-page resumes are penalized by $b=0.75$ length normalization."
                )

            with tab_lev:
                lev_data = uth.get("levenshtein", {})
                st.markdown("### Method 4: Semantic Levenshtein Edit Distance Matrix")
                st.latex(r"\text{Levenshtein Similarity}(s_1, s_2) = 1.0 - \frac{\text{EditDistance}(s_1, s_2)}{\max(|s_1|, |s_2|)}")
                
                c_l1, c_l2, c_l3 = st.columns(3)
                c_l1.metric("Levenshtein Similarity", f"{scores['levenshtein_similarity']:.1f}%")
                c_l2.metric("Fuzzy Matched Pairs", f"{lev_data.get('matched_pairs_count', 0)}")
                c_l3.metric("Job Keywords Analyzed", f"{result.get('levenshtein_details', {}).get('job_vocab_size', 0)}")

                st.markdown("#### 📐 Step-by-Step Solving Breakdown:")
                st.markdown(
                    """
                    1. **Dynamic Programming Edit Matrix:** Computes minimum single-character insertions, deletions, and substitutions between Job token $s_1$ and Resume token $s_2$.
                    2. **Morphological Bridging:** Word pairs achieving $\ge 70\%$ edit similarity receive partial credit (e.g. `engineer` ↔ `engineering`, `automate` ↔ `automation`).
                    3. **Normalized Average:** Sum of best-match similarities divided by total unique job keywords.
                    """
                )
                matched_lev = lev_data.get("matched_pairs", [])
                if matched_lev:
                    st.markdown("#### 🔍 Matched Fuzzy & Morphological Word Pairs Table:")
                    df_lev = pd.DataFrame(matched_lev).rename(columns={
                        "job_word": "Job Word (s1)",
                        "resume_word": "Matched Resume Word (s2)",
                        "edit_distance": "Min Edit Operations",
                        "similarity": "Fuzzy Similarity (%)"
                    })
                    st.dataframe(df_lev, hide_index=True, use_container_width=True)
                st.info(
                    "💡 **Why Levenshtein Beats Jobscan:** When a job requires 'automation' and a resume says 'automate' (or 'engineering' vs 'engineer'), Jobscan scores 0%. Levenshtein awards 70%+ similarity for grammatical variants."
                )

            with tab_stuffing:
                st_data = uth.get("anti_stuffing", {})
                st.markdown("### Method 5: Adversarial Keyword Stuffing & Authenticity Audit")
                st.latex(r"\text{Authenticity Score} = \max\left(0, 100 - \sum \text{Penalties}\right)")
                
                c_s1, c_s2, c_s3, c_s4 = st.columns(4)
                c_s1.metric("Authenticity Score", f"{st_data.get('authenticity_score', 100):.0f}%")
                c_s2.metric("Audit Risk Level", f"{st_data.get('risk_level', 'Authentic')}")
                c_s3.metric("Action Verbs Found", f"{st_data.get('action_verbs_found', 0)}")
                c_s4.metric("Skill Density", f"{st_data.get('skill_density_pct', 0.0):.1f}%")

                st.markdown("#### 📐 Step-by-Step Adversarial Audit Rubric:")
                breakdown = st_data.get("penalty_breakdown", [])
                if breakdown:
                    st.dataframe(pd.DataFrame(breakdown), hide_index=True, use_container_width=True)
                
                total_pen = st_data.get('total_penalty', 0)
                st.markdown(
                    f"""
                    - **Total Audit Penalty:** `-{total_pen} points`
                    - **Final Authenticity Calculation:** `max(0, 100 - {total_pen}) =` **`{st_data.get('authenticity_score', 100):.0f}%`**
                    - **Verdict:** {st_data.get('verdict', 'Authentic')}
                    """
                )

                if st_data.get("flags"):
                    st.error("⚠️ **Detected Anomalies / Stuffing Flags:**")
                    for fl in st_data.get("flags", []):
                        st.write(f"• {fl}")
                else:
                    st.success("✅ **Verified Authentic:** Keywords are naturally integrated into descriptive project bullets with strong action verbs.")

                st.info(
                    "💡 **Why This Beats Jobscan:** Candidates can trick Jobscan with 30 keywords in white text or raw comma lists to get 95%+. Our adversarial audit checks verb coupling and flags candidates who dump buzzwords without active accomplishment sentences."
                )

        st.divider()

        # ----------------------------------------------------------------------
        # 2. CATEGORIZED SKILL BREAKDOWN & GAP ANALYSIS
        # ----------------------------------------------------------------------
        st.subheader("2. Categorized Skill Breakdown & Gap Analysis")
        col_matched, col_missing = st.columns(2)

        with col_matched:
            st.markdown(f"#### ✅ Matched Competencies ({len(result['matched_skills'])})")
            if result.get("categorized_matched"):
                for category, skills in result["categorized_matched"].items():
                    st.markdown(f"**{category}:**")
                    badges_html = " ".join([f"<span class='badge-matched'>{sk}</span>" for sk in skills])
                    st.markdown(badges_html, unsafe_allow_html=True)
            elif result["matched_skills"]:
                st.write(", ".join(result["matched_skills"]))
            else:
                st.info("No matching skills detected from the taxonomy.")

        with col_missing:
            st.markdown(f"#### ⚠️ Missing / Unmatched Competencies ({len(result['missing_skills'])})")
            if result.get("categorized_missing"):
                for category, skills in result["categorized_missing"].items():
                    st.markdown(f"**{category}:**")
                    badges_html = " ".join([f"<span class='badge-missing'>{sk}</span>" for sk in skills])
                    st.markdown(badges_html, unsafe_allow_html=True)
            elif result["missing_skills"]:
                st.write(", ".join(result["missing_skills"]))
            else:
                st.success("All detected job skills are represented in the resume!")

        st.divider()

        # ----------------------------------------------------------------------
        # 3. ATS RESUME HYGIENE & STRUCTURAL AUDIT
        # ----------------------------------------------------------------------
        st.subheader("3. ATS Resume Hygiene & Structural Audit")
        hygiene = result.get("ats_hygiene", {})

        col_h1, col_h2, col_h3 = st.columns([1, 1, 2])
        with col_h1:
            st.metric("ATS Structural Health Score", f"{hygiene.get('overall_score', 0)}/100")
            st.caption(f"Status: **{hygiene.get('length_status', 'N/A')}** ({hygiene.get('word_count', 0)} words)")
        with col_h2:
            st.metric("Action Verbs Detected", f"{hygiene.get('action_verb_count', 0)}")
            st.caption(f"Quantifiable Metrics Found: **{hygiene.get('metric_count', 0)}**")
        with col_h3:
            st.markdown("**Core Section Completeness:**")
            sections = hygiene.get("sections", {})
            cols_sec = st.columns(2)
            half = len(sections) // 2
            sec_items = list(sections.items())
            for idx, (sec_name, found) in enumerate(sec_items):
                target_col = cols_sec[0] if idx < half else cols_sec[1]
                icon = "✅" if found else "❌"
                target_col.write(f"{icon} {sec_name}")

        st.markdown("#### 🎯 Actionable Optimization Checklist")
        for rec in hygiene.get("recommendations", []):
            st.info(f"💡 {rec}")

        if hygiene.get("action_verbs_found"):
            st.caption(f"**Detected Action Verbs:** {', '.join(hygiene['action_verbs_found'][:15])}")

        st.divider()

        # ----------------------------------------------------------------------
        # 4. EXECUTION LATENCY & JSON EXPORT
        # ----------------------------------------------------------------------
        st.subheader("4. Runtime Latency Profiling")
        timing_df = pd.DataFrame(
            [{"Method": name, "Execution Time (Seconds)": f"{sec:.4f}s"} for name, sec in result["runtime_seconds"].items()]
        )
        st.dataframe(timing_df, hide_index=True, use_container_width=True)

        st.download_button(
            "📥 Download Full Analysis (JSON)",
            data=json.dumps(result, indent=2),
            file_name="resume_nlp_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

# ==============================================================================
# PAGE 2: EVALUATION LAB
# ==============================================================================
else:
    st.subheader("🧪 Evaluation Lab — Empirical NLP Benchmarking")
    st.markdown(
        "Empirical benchmark evaluating the matching algorithms against human relevance labels using "
        "standard Information Retrieval (IR) metrics: **Accuracy, Precision, Recall, F1-Score, and Spearman Rank Correlation**."
    )

    sample_csv_path = Path(__file__).parent / "data" / "sample_evaluation.csv"
    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_csv = st.file_uploader("Upload labeled evaluation CSV", type=["csv"], key="eval_csv")
    with col_sample:
        use_sample = st.checkbox("Use bundled benchmark dataset", value=True)

    dataset = None
    source_name = None
    if uploaded_csv is not None:
        try:
            dataset = load_evaluation_data(uploaded_csv.getvalue())
            source_name = uploaded_csv.name
        except Exception as exc:
            st.error(str(exc))
    elif use_sample and sample_csv_path.exists():
        try:
            dataset = pd.read_csv(sample_csv_path)
            source_name = "sample_evaluation.csv (Bundled Benchmark)"
        except Exception as exc:
            st.error(f"Could not load bundled dataset: {exc}")

    if dataset is not None:
        st.markdown(f"**Loaded Dataset:** `{source_name}` · **Sample Count:** {len(dataset)} labeled pairs")
        st.dataframe(dataset.head(10), use_container_width=True, hide_index=True)

        threshold = st.slider(
            "Classification Decision Threshold (%)",
            min_value=10,
            max_value=90,
            value=50,
            step=5,
            help="Scores at or above this percentage are classified as relevant (1). Default is 50%.",
        )

        if st.button("📊 Run Benchmark Evaluation Across All 4 Methods", type="primary", use_container_width=True):
            try:
                with st.spinner("Computing inferences and statistical ranking metrics across dataset..."):
                    evaluation = evaluate_dataset(dataset, threshold=float(threshold))
                st.session_state["evaluation_result"] = evaluation
                st.success("Benchmark evaluation completed.")
            except Exception as exc:
                st.error(f"Evaluation error: {exc}")

    evaluation = st.session_state.get("evaluation_result")
    if evaluation:
        st.divider()
        st.subheader("Performance Metrics (Accuracy, Precision, Recall, F1)")
        metrics_df = pd.DataFrame(evaluation["metrics"])
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        st.bar_chart(metrics_df.set_index("Method")[["Precision", "Recall", "F1"]])

        st.subheader("Spearman Rank Correlation (Agreement with Human Grading)")
        st.caption(
            "Spearman's rank correlation coefficient evaluates whether higher algorithm scores consistently "
            "correspond to higher human relevance grades (values closer to +1.0 indicate superior ranking ability)."
        )
        st.dataframe(pd.DataFrame(evaluation["ranking_metrics"]), hide_index=True, use_container_width=True)

        st.subheader("Confusion Matrix Statistics")
        predictions_for_counts = pd.DataFrame(evaluation["predictions"])
        confusion_rows = []
        for method_name, group in predictions_for_counts.groupby("Method", sort=False):
            actual = group["Human label"].astype(int)
            predicted = group["Predicted label"].astype(int)
            confusion_rows.append({
                "Method": method_name,
                "True Positives (TP)": int(((actual == 1) & (predicted == 1)).sum()),
                "True Negatives (TN)": int(((actual == 0) & (predicted == 0)).sum()),
                "False Positives (FP - Type I Error)": int(((actual == 0) & (predicted == 1)).sum()),
                "False Negatives (FN - Type II Error)": int(((actual == 1) & (predicted == 0)).sum()),
            })
        st.dataframe(pd.DataFrame(confusion_rows), hide_index=True, use_container_width=True)

        with st.expander("🔎 Per-Example Predictions & Error Inspection"):
            pred_df = pd.DataFrame(evaluation["predictions"])
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

        c_d1, c_d2 = st.columns(2)
        with c_d1:
            st.download_button(
                "📥 Export Predictions (CSV)",
                data=pd.DataFrame(evaluation["predictions"]).to_csv(index=False).encode("utf-8"),
                file_name="benchmark_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_d2:
            st.download_button(
                "📥 Export Metrics Summary (CSV)",
                data=metrics_df.to_csv(index=False).encode("utf-8"),
                file_name="benchmark_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
