import io
import math
import re
import time
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer

# Support modern pymupdf with fallback to legacy fitz
try:
    import pymupdf as fitz
except ImportError:
    import fitz

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

_stemmer = PorterStemmer()

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should",
    "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

# ==============================================================================
# 1. COMPREHENSIVE CATEGORIZED SKILL TAXONOMY
# ==============================================================================

SKILL_TAXONOMY: Dict[str, List[str]] = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "ruby", "php", "swift", "kotlin", "scala", "r", "sql", "bash", "shell",
        "html", "css", "dart", "matlab", "solidity"
    ],
    "Frameworks & Web Tech": [
        "react", "angular", "vue", "next.js", "node.js", "express", "django",
        "flask", "fastapi", "spring", "spring boot", "asp.net", "laravel",
        "ruby on rails", "tailwind css", "bootstrap", "graphql", "rest api",
        "api", "apis", "api integration", "microservices", "redux", "svelte", "django rest framework"
    ],
    "Data, AI & Machine Learning": [
        "ai", "artificial intelligence", "ai models", "machine learning", "deep learning",
        "natural language processing", "nlp", "computer vision", "large language models",
        "generative ai", "transformers", "sentence transformers", "prompt engineering",
        "prompts", "chatbots", "ai assistants", "recommendation systems", "content generation",
        "scikit-learn", "pandas", "numpy", "scipy", "pytorch", "tensorflow", "keras",
        "spacy", "nltk", "hugging face", "data analysis", "data visualization", "statistics",
        "predictive modeling", "data science", "etl", "data engineering", "apache spark",
        "hadoop", "reinforcement learning", "tf-idf", "langchain", "vector database",
        "power bi", "tableau", "excel"
    ],
    "Cloud, DevOps & Infrastructure": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "git", "github", "gitlab", "ci/cd", "automation", "workflow automation",
        "workflows", "intelligent workflows", "linux", "unix", "nginx", "apache",
        "serverless", "cloud computing", "prometheus", "grafana", "helm", "cloudformation",
        "kafka", "rabbitmq"
    ],
    "Databases & Storage": [
        "database", "databases", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "sqlite", "oracle", "sql server", "cassandra", "dynamodb", "snowflake", "bigquery",
        "nosql", "mariadb", "neo4j", "chromadb", "pinecone"
    ],
    "Prototyping, QA & Soft Skills": [
        "prototyping", "prototypes", "mvp", "mvps", "rapid prototyping", "no-code", "low-code",
        "testing", "debugging", "optimization", "agile", "scrum", "kanban", "jira",
        "unit testing", "automated testing", "selenium", "pytest", "test-driven development",
        "oop", "object-oriented programming", "data structures", "algorithms",
        "system design", "cybersecurity", "problem solving", "communication",
        "leadership", "teamwork", "collaboration", "project management",
        "cross-functional collaboration", "critical thinking", "mentoring", "code review"
    ]
}

# Reverse lookup: skill -> category name
SKILL_TO_CATEGORY: Dict[str, str] = {}
for category, skills in SKILL_TAXONOMY.items():
    for skill in skills:
        SKILL_TO_CATEGORY[skill.lower()] = category

# Flattened list of canonical skills
SKILLS: List[str] = sorted(list(SKILL_TO_CATEGORY.keys()))

# Alias / Synonym Dictionary (maps variations and acronyms to canonical skills)
SKILL_ALIASES: Dict[str, str] = {
    "golang": "go",
    "k8s": "kubernetes",
    "k8": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "llm": "large language models",
    "llms": "large language models",
    "genai": "generative ai",
    "gen ai": "generative ai",
    "prompt-engineering": "prompt engineering",
    "prompt engineering workflows": "prompt engineering",
    "ai prompts": "prompts",
    "prompt design": "prompt engineering",
    "chatbot": "chatbots",
    "ai assistant": "ai assistants",
    "recommender systems": "recommendation systems",
    "content-generation": "content generation",
    "low code": "low-code",
    "no code": "no-code",
    "prototype": "prototyping",
    "prototypes": "prototyping",
    "mvps": "mvp",
    "workflow": "workflows",
    "automated workflows": "workflow automation",
    "test": "testing",
    "tests": "testing",
    "debug": "debugging",
    "optimize": "optimization",
    "optimizing": "optimization",
    "collaborate": "collaboration",
    "team work": "teamwork",
    "amazon web services": "aws",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "microsoft azure": "azure",
    "react.js": "react",
    "reactjs": "react",
    "node": "node.js",
    "nodejs": "node.js",
    "nextjs": "next.js",
    "next": "next.js",
    "vue.js": "vue",
    "vuejs": "vue",
    "postgres": "postgresql",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "continuous integration": "ci/cd",
    "continuous deployment": "ci/cd",
    "tdd": "test-driven development",
    "test driven development": "test-driven development",
    "tf": "tensorflow",
    "sk-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "huggingface": "hugging face",
    "powerbi": "power bi",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "object oriented programming": "object-oriented programming",
    "oop": "object-oriented programming",
    "spark": "apache spark",
    "rest": "rest api",
    "restful": "rest api",
    "restful api": "rest api",
    "mongo": "mongodb",
    "elastic search": "elasticsearch",
    "vector db": "vector database",
    "system architecture": "system design",
}

# Inverted mapping: canonical_skill -> list of alias terms
CANONICAL_TO_ALIASES: Dict[str, List[str]] = {}
for alias, canonical in SKILL_ALIASES.items():
    CANONICAL_TO_ALIASES.setdefault(canonical, []).append(alias)

HYBRID_WEIGHTS = {
    "keyword_coverage": 0.40,
    "tfidf_similarity": 0.25,
    "semantic_similarity": 0.35,
}

# ==============================================================================
# 2. DOCUMENT EXTRACTION & NORMALIZATION
# ==============================================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract selectable text from a PDF. OCR for image-only PDFs is not included."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        return "\n".join(page.get_text("text") for page in document).strip()

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def validate_pair(resume: str, job: str) -> List[str]:
    errors = []
    if not resume or len(resume.strip()) < 30:
        errors.append("Please provide a resume with at least 30 characters of text.")
    if not job or len(job.strip()) < 30:
        errors.append("Please provide a job description with at least 30 characters of text.")
    if resume and len(resume) > 250_000:
        errors.append("Resume text is too long for this prototype (maximum 250,000 characters).")
    if job and len(job) > 250_000:
        errors.append("Job description is too long for this prototype (maximum 250,000 characters).")
    return errors

# ==============================================================================
# 3. BOUNDARY-AWARE & SYNONYM-AWARE SKILL EXTRACTION
# ==============================================================================

def _contains_term(text: str, term: str) -> bool:
    """
    Boundary-aware matching with hyphen normalization and optional plural support.
    Matches 'prompt-engineering' to 'prompt engineering', and 'apis' to 'api'.
    """
    lowered = text.lower()
    norm_space = re.sub(r"[-_/]", " ", lowered)
    term_low = term.lower()
    term_space = re.sub(r"[-_/]", " ", term_low)

    # Guard short, ambiguous terms from generic English matches
    if term_low == "go":
        return bool(re.search(r"\b(golang|go\s+(?:programming|language|developer|code))\b", lowered))
    if term_low == "r":
        return bool(re.search(r"\b(r\s+(?:programming|language|stats|package|script))\b", lowered))
    if term_low == "c":
        return bool(re.search(r"\b(c\s+(?:programming|language|code))\b|\bansi\s+c\b", lowered))
    if term_low == "ai":
        return bool(re.search(r"\b(ai|artificial\s+intelligence)\b", lowered))

    # Pattern with optional plural suffix
    pat_exact = r"(?<![a-z0-9])" + re.escape(term_low) + r"(?:s|es)?(?![a-z0-9])"
    pat_space = r"(?<![a-z0-9])" + re.escape(term_space) + r"(?:s|es)?(?![a-z0-9])"

    return bool(re.search(pat_exact, lowered) or re.search(pat_space, norm_space))

def extract_skills(text: str) -> List[str]:
    """
    Extract canonical skills from text using boundary-aware search
    across canonical terms and their known aliases/synonyms.
    """
    detected: Set[str] = set()
    for canonical in SKILLS:
        # Check canonical term
        if _contains_term(text, canonical):
            detected.add(canonical)
            continue
        # Check any aliases mapped to this canonical skill
        for alias in CANONICAL_TO_ALIASES.get(canonical, []):
            if _contains_term(text, alias):
                detected.add(canonical)
                break
    return sorted(list(detected))

def categorize_skills(skills: List[str]) -> Dict[str, List[str]]:
    """Group a list of skills by their architectural/domain category."""
    categorized: Dict[str, List[str]] = {cat: [] for cat in SKILL_TAXONOMY}
    for skill in sorted(skills):
        category = SKILL_TO_CATEGORY.get(skill, "Other / Emerging")
        if category in categorized:
            categorized[category].append(skill)
        else:
            categorized.setdefault("Other / Emerging", []).append(skill)
    # Filter out empty categories
    return {cat: sk for cat, sk in categorized.items() if sk}

def keyword_method(resume: str, job: str) -> Dict:
    resume_skills = set(extract_skills(resume))
    job_skills = set(extract_skills(job))
    matched = sorted(list(resume_skills & job_skills))
    missing = sorted(list(job_skills - resume_skills))
    score = (len(matched) / len(job_skills) * 100.0) if job_skills else None
    return {
        "score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "job_skill_count": len(job_skills),
        "categorized_matched": categorize_skills(matched),
        "categorized_missing": categorize_skills(missing),
    }

# ==============================================================================
# 4. TF-IDF & DENSE SEMANTIC SIMILARITY
# ==============================================================================

def tokenize_and_stem(text: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Clean text, remove stop words, and stem tokens using PorterStemmer.
    Returns (stemmed_tokens, map_stem_to_sample_original_word).
    """
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    raw_words = clean.split()
    tokens: List[str] = []
    stem_to_original: Dict[str, str] = {}
    for w in raw_words:
        if len(w) > 1 and w not in STOP_WORDS and not w.isdigit():
            stem = _stemmer.stem(w)
            tokens.append(stem)
            if stem not in stem_to_original:
                stem_to_original[stem] = w
    return tokens, stem_to_original

def calculate_detailed_tfidf(resume: str, job: str) -> Dict:
    """
    Step-by-step, transparent TF-IDF calculation with full vector decomposition:
    - Preprocessing with Porter Stemmer and stop words removal
    - Term Frequency: TF(t, d) = count(t, d) / total_tokens(d)
    - Inverse Document Frequency: IDF(t) = ln((N + 1) / (DF(t) + 1)) + 1
    - TF-IDF = TF * IDF
    - Dot Product and Euclidean L2 norms
    - Cosine Similarity = (u . v) / (||u|| * ||v||)
    """
    r_tokens, r_orig = tokenize_and_stem(resume)
    j_tokens, j_orig = tokenize_and_stem(job)

    r_total = len(r_tokens)
    j_total = len(j_tokens)

    if r_total == 0 or j_total == 0:
        return {
            "score": 0.0,
            "resume_token_count": r_total,
            "job_token_count": j_total,
            "vocab_size": 0,
            "vocabulary_size": 0,
            "shared_term_count": 0,
            "dot_product": 0.0,
            "resume_norm": 0.0,
            "job_norm": 0.0,
            "terms_table": [],
            "top_shared_terms": [],
        }

    r_counts = Counter(r_tokens)
    j_counts = Counter(j_tokens)
    vocab = sorted(list(set(r_tokens) | set(j_tokens)))

    terms_table = []
    top_shared = []
    dot_product = 0.0
    r_sq_sum = 0.0
    j_sq_sum = 0.0

    for t in vocab:
        c_r = r_counts.get(t, 0)
        c_j = j_counts.get(t, 0)

        tf_r = c_r / r_total if r_total > 0 else 0.0
        tf_j = c_j / j_total if j_total > 0 else 0.0

        # Document Frequency
        df = (1 if c_r > 0 else 0) + (1 if c_j > 0 else 0)
        # Smooth IDF for 2 documents
        idf = math.log(3.0 / (df + 1.0)) + 1.0

        tfidf_r = tf_r * idf
        tfidf_j = tf_j * idf

        product = tfidf_r * tfidf_j
        dot_product += product
        r_sq_sum += tfidf_r ** 2
        j_sq_sum += tfidf_j ** 2

        is_shared = (c_r > 0 and c_j > 0)
        orig_word = r_orig.get(t, j_orig.get(t, t))

        row_dict = {
            "term": t,
            "original_word": orig_word,
            "resume_count": c_r,
            "resume_tf": round(tf_r, 4),
            "job_count": c_j,
            "job_tf": round(tf_j, 4),
            "df": df,
            "idf": round(idf, 4),
            "resume_tfidf": round(tfidf_r, 4),
            "job_tfidf": round(tfidf_j, 4),
            "product": round(product, 6),
            "is_shared": is_shared,
        }
        terms_table.append(row_dict)
        if is_shared:
            top_shared.append(row_dict)

    r_norm = math.sqrt(r_sq_sum)
    j_norm = math.sqrt(j_sq_sum)

    if r_norm > 0 and j_norm > 0:
        cosine_sim = (dot_product / (r_norm * j_norm)) * 100.0
    else:
        cosine_sim = 0.0

    cosine_sim = float(np.clip(cosine_sim, 0.0, 100.0))

    # Sort top shared by dot product contribution
    top_shared.sort(key=lambda x: x["product"], reverse=True)
    # Sort full table: shared terms first, then highest product, then alphabetical
    terms_table.sort(key=lambda x: (-1 if x["is_shared"] else 1, -x["product"], x["term"]))

    return {
        "score": round(cosine_sim, 2),
        "resume_token_count": r_total,
        "job_token_count": j_total,
        "vocab_size": len(vocab),
        "vocabulary_size": len(vocab),
        "shared_term_count": len(top_shared),
        "dot_product": round(dot_product, 6),
        "resume_norm": round(r_norm, 6),
        "job_norm": round(j_norm, 6),
        "terms_table": terms_table,
        "top_shared_terms": top_shared[:15],
    }

def tfidf_method(resume: str, job: str) -> float:
    """Document-level lexical similarity using stemmed tokens and normalized TF-IDF cosine vector angle."""
    return calculate_detailed_tfidf(resume, job)["score"]

def get_tfidf_details(resume: str, job: str, top_n: int = 15) -> Dict:
    """Extract top overlapping stemmed tokens and vocabulary statistics for explainability."""
    return calculate_detailed_tfidf(resume, job)

def calculate_okapi_bm25(resume: str, job: str, k1: float = 1.5, b: float = 0.75, avgdl: float = 350.0) -> Dict:
    """
    Industry-standard Okapi BM25 ranking algorithm:
    BM25(D, Q) = sum_{t in Q} IDF(t) * (f(t, D) * (k1 + 1)) / (f(t, D) + k1 * (1 - b + b * (|D| / avgdl)))
    - k1 (1.5): Term frequency saturation parameter
    - b (0.75): Document length normalization parameter
    - avgdl (350.0): Reference document length benchmark
    """
    r_tokens, _ = tokenize_and_stem(resume)
    j_tokens, _ = tokenize_and_stem(job)

    if not r_tokens or not j_tokens:
        return {
            "score": 0.0,
            "norm_score": 0.0,
            "raw_score": 0.0,
            "k1": k1,
            "b": b,
            "doc_length": len(r_tokens),
            "query_length": len(j_tokens),
            "matched_terms": []
        }

    doc_len = len(r_tokens)
    doc_counts = Counter(r_tokens)
    query_counts = Counter(j_tokens)

    raw_score = 0.0
    max_score = 0.0
    matched_terms = []
    K = k1 * (1.0 - b + b * (doc_len / avgdl))

    for t, q_count in query_counts.items():
        f = doc_counts.get(t, 0)
        df = 1 if f > 0 else 0
        idf = math.log((2.0 - df + 0.5) / (df + 0.5) + 1.0)
        
        denom = f + K
        tf_comp = (f * (k1 + 1.0)) / denom if denom > 0 else 0.0
        
        term_score = idf * tf_comp
        raw_score += term_score
        max_score += idf * (k1 + 1.0)
        
        if f > 0:
            matched_terms.append({
                "term": t,
                "resume_freq": f,
                "job_freq": q_count,
                "idf": round(idf, 4),
                "tf_component": round(tf_comp, 4),
                "bm25_contribution": round(term_score, 4)
            })

    norm_score = (raw_score / max_score) * 100.0 if max_score > 0 else 0.0
    norm_score = float(np.clip(norm_score, 0.0, 100.0))
    matched_terms.sort(key=lambda x: x["bm25_contribution"], reverse=True)

    return {
        "raw_score": round(raw_score, 4),
        "score": round(norm_score, 2),
        "norm_score": round(norm_score, 2),
        "max_score": round(max_score, 4),
        "k_factor": round(K, 4),
        "avgdl": avgdl,
        "k1": k1,
        "b": b,
        "doc_length": doc_len,
        "query_length": len(j_tokens),
        "matched_terms": matched_terms[:15]
    }

def levenshtein_distance(s1: str, s2: str) -> int:
    """Dynamic programming Levenshtein minimum edit distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

def calculate_levenshtein_similarity(resume: str, job: str) -> Dict:
    """
    Fuzzy Levenshtein Edit-Distance Similarity:
    Measures morphological and orthographic alignment between query tokens and candidate tokens.
    Sim(s1, s2) = 1.0 - (LevenshteinDistance(s1, s2) / max(len(s1), len(s2)))
    """
    r_words = list(set(re.sub(r"[^a-zA-Z0-9\s]", " ", resume.lower()).split()))
    j_words = list(set(re.sub(r"[^a-zA-Z0-9\s]", " ", job.lower()).split()))
    
    r_filtered = [w for w in r_words if w not in STOP_WORDS and len(w) > 2]
    j_filtered = [w for w in j_words if w not in STOP_WORDS and len(w) > 2]

    if not r_filtered or not j_filtered:
        return {"score": 0.0, "matched_pairs": []}

    matched_pairs = []
    total_sim = 0.0

    for jw in j_filtered:
        best_sim = 0.0
        best_rw = None
        for rw in r_filtered:
            max_len = max(len(jw), len(rw))
            dist = levenshtein_distance(jw, rw)
            sim = 1.0 - (dist / max_len)
            if sim > best_sim:
                best_sim = sim
                best_rw = rw
        
        if best_sim >= 0.70:
            matched_pairs.append({
                "job_word": jw,
                "resume_word": best_rw,
                "similarity": round(best_sim * 100.0, 1),
                "edit_distance": levenshtein_distance(jw, best_rw)
            })
            total_sim += best_sim

    overall_score = (total_sim / len(j_filtered)) * 100.0 if j_filtered else 0.0
    overall_score = float(np.clip(overall_score, 0.0, 100.0))
    matched_pairs.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "score": round(overall_score, 2),
        "job_vocab_size": len(j_filtered),
        "matched_pairs_count": len(matched_pairs),
        "matched_pairs": matched_pairs[:15]
    }

def detect_keyword_stuffing(resume_text: str, detected_skills: List[str]) -> Dict:
    """
    Adversarial Keyword Stuffing & Authenticity Detector:
    Detects attempts to game ATS scanners (e.g., hidden keyword dumps, white-text lists).
    Evaluates:
    - Keyword-to-Action-Verb Ratio
    - Keyword Word-Density Anomaly
    - Dense comma-separated keyword dumping without syntactic sentences
    """
    words = resume_text.lower().split()
    total_words = max(1, len(words))

    found_verbs = [w.strip(".,;:()[]{}\"'") for w in words if w.strip(".,;:()[]{}\"'") in ACTION_VERBS]
    verb_count = len(found_verbs)

    skill_count = len(detected_skills)
    skill_word_density = (skill_count / total_words) * 100.0

    flags = []
    penalty = 0
    penalty_breakdown = []

    if skill_count > 8 and verb_count < 2:
        flags.append("Critical: High keyword density with near-zero action verbs (classic keyword stuffing pattern).")
        penalty += 45
        penalty_breakdown.append({"Factor": "1. Verb-Keyword Coupling", "Status": f"Critical ({skill_count} skills vs {verb_count} verbs)", "Penalty": -45})
    elif skill_count > 12 and verb_count < 4:
        flags.append("Warning: Disproportionately high keyword-to-verb ratio (lacks descriptive sentences).")
        penalty += 25
        penalty_breakdown.append({"Factor": "1. Verb-Keyword Coupling", "Status": f"Warning ({skill_count} skills vs {verb_count} verbs)", "Penalty": -25})
    else:
        penalty_breakdown.append({"Factor": "1. Verb-Keyword Coupling", "Status": f"Passed ({verb_count} verbs coupling {skill_count} skills)", "Penalty": 0})

    if skill_word_density > 25.0:
        flags.append(f"Warning: Excessive keyword density ({skill_word_density:.1f}% of total words are buzzwords).")
        penalty += 20
        penalty_breakdown.append({"Factor": "2. Skill Word Density", "Status": f"High ({skill_word_density:.1f}% > 25.0% threshold)", "Penalty": -20})
    else:
        penalty_breakdown.append({"Factor": "2. Skill Word Density", "Status": f"Normal ({skill_word_density:.1f}% <= 25.0% threshold)", "Penalty": 0})

    list_patterns = re.findall(r"(?:[a-zA-Z0-9+#.-]+,\s*){5,}[a-zA-Z0-9+#.-]+", resume_text)
    if list_patterns:
        flags.append("Detected dense comma-separated keyword dump without sentence context.")
        penalty += 15
        penalty_breakdown.append({"Factor": "3. Buzzword List Dump Pattern", "Status": "Detected 5+ comma-separated keyword list", "Penalty": -15})
    else:
        penalty_breakdown.append({"Factor": "3. Buzzword List Dump Pattern", "Status": "Passed (No raw list dump patterns)", "Penalty": 0})

    authenticity_score = max(0, 100 - penalty)
    if authenticity_score >= 80:
        risk_level = "Authentic / Low Risk"
        verdict = "Resume naturally weaves competencies into descriptive project accomplishments."
    elif authenticity_score >= 50:
        risk_level = "Moderate Risk"
        verdict = "Contains skill lists; consider expanding bullet points with strong action verbs."
    else:
        risk_level = "High Stuffing Risk (Likely Gaming ATS)"
        verdict = "Severe buzzword dumping detected. Fails human review despite triggering naive keyword counters."

    return {
        "authenticity_score": authenticity_score,
        "risk_level": risk_level,
        "verdict": verdict,
        "action_verbs_found": verb_count,
        "skill_count": skill_count,
        "skill_density_pct": round(skill_word_density, 1),
        "total_penalty": penalty,
        "penalty_breakdown": penalty_breakdown,
        "flags": flags
    }


@lru_cache(maxsize=1)
def _load_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _chunks(text: str, words_per_chunk: int = 100) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)] or [""]

def semantic_method(resume: str, job: str) -> float:
    """
    Compare normalized mean embeddings over short chunks to reduce long-document truncation.
    Captures dense conceptual similarity, intent, and conceptual equivalence beyond literal keywords.
    """
    model = _load_sentence_model()
    resume_chunks = _chunks(resume)
    job_chunks = _chunks(job)
    resume_vectors = model.encode(resume_chunks, normalize_embeddings=True, show_progress_bar=False)
    job_vectors = model.encode(job_chunks, normalize_embeddings=True, show_progress_bar=False)
    resume_vector = np.mean(resume_vectors, axis=0, keepdims=True)
    job_vector = np.mean(job_vectors, axis=0, keepdims=True)
    cosine = float(cosine_similarity(resume_vector, job_vector)[0, 0])
    return float(np.clip(cosine * 100.0, 0.0, 100.0))

# ==============================================================================
# 5. ATS RESUME HYGIENE & STRUCTURAL DIAGNOSTIC
# ==============================================================================

ACTION_VERBS = {
    "accelerated", "achieved", "architected", "automated", "built", "centralized",
    "championed", "collaborated", "constructed", "created", "decreased", "delivered",
    "deployed", "designed", "developed", "directed", "engineered", "established",
    "executed", "expanded", "expedited", "formulated", "founded", "generated",
    "guided", "implemented", "improved", "increased", "initiated", "innovated",
    "installed", "instituted", "integrated", "launched", "lead", "led", "managed",
    "maximized", "mentored", "modernized", "negotiated", "optimized", "orchestrated",
    "overhauled", "pioneered", "reduced", "refactored", "resolved", "restructured",
    "revamped", "saved", "scaled", "simplified", "spearheaded", "standardized",
    "streamlined", "strengthened", "transformed", "upgraded"
}

STANDARD_SECTIONS = {
    "Contact Info": [r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b", r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", r"linkedin\.com", r"github\.com"],
    "Summary / Objective": [r"\b(?:professional\s+summary|summary|profile|about\s+me|objective)\b"],
    "Experience / Work History": [r"\b(?:experience|work\s+experience|employment\s+history|professional\s+experience|career\s+history)\b"],
    "Education": [r"\b(?:education|academic\s+background|degrees?|university|college|bachelor|master|phd|b\.s\.|m\.s\.)\b"],
    "Skills & Competencies": [r"\b(?:skills|technical\s+skills|core\s+competencies|technologies|proficiencies|expertise)\b"],
    "Projects & Portfolio": [r"\b(?:projects|key\s+projects|personal\s+projects|academic\s+projects|portfolio)\b"],
    "Certifications": [r"\b(?:certifications?|certificates?|licenses?|credentials?|accreditations?)\b"],
}

def analyze_ats_hygiene(resume_text: str) -> Dict:
    """
    Comprehensive ATS structural and quality diagnostics:
    - Section presence detection
    - Word count & length classification
    - Action verb density & impact verbs
    - Quantifiable metrics & achievements detection
    - Actionable optimization recommendations
    """
    words = resume_text.split()
    word_count = len(words)
    
    # 1. Section Presence
    sections_status = {}
    found_sections_count = 0
    for section_name, patterns in STANDARD_SECTIONS.items():
        found = False
        for pat in patterns:
            if re.search(pat, resume_text, re.IGNORECASE):
                found = True
                break
        sections_status[section_name] = found
        if found:
            found_sections_count += 1

    # 2. Length & Readability Health
    if word_count < 200:
        length_status = "Too Sparse (< 200 words)"
        length_score = 30
        length_note = "Your resume is too brief. Typical ATS systems expect 400–900 words for comprehensive context."
    elif 200 <= word_count < 400:
        length_status = "Concise (200–400 words)"
        length_score = 75
        length_note = "A bit on the shorter side; ensure all relevant technical competencies and impacts are listed."
    elif 400 <= word_count <= 950:
        length_status = "Optimal Length (400–950 words)"
        length_score = 100
        length_note = "Ideal length for 1–2 page professional ATS readability."
    elif 950 < word_count <= 1400:
        length_status = "Detailed (950–1,400 words)"
        length_score = 85
        length_note = "Acceptable for senior candidates (2+ pages), but be mindful of brevity."
    else:
        length_status = "Overly Long (> 1,400 words)"
        length_score = 55
        length_note = "Excessively long for automated screening; consider condensing to focus on high-impact results."

    # 3. Action Verb Detection
    lowered_words = [w.lower().strip(".,;:()[]{}\"'") for w in words]
    found_action_verbs = sorted(list(set(lowered_words) & ACTION_VERBS))
    action_verb_count = len(found_action_verbs)
    action_verb_score = min(100, int((action_verb_count / 8.0) * 100))

    # 4. Quantifiable Impact & Metrics Detection
    # Look for %, $, numbers with k/M/+, or metrics like "reduced by 25%"
    metric_matches = re.findall(r"\b(?:\$?\d+(?:\.\d+)?%?|\d+\s*(?:k|m|million|billion|\+))\b", resume_text, re.IGNORECASE)
    metric_count = len(metric_matches)
    metric_score = min(100, int((metric_count / 6.0) * 100))

    # 5. Composite ATS Hygiene Score (0-100)
    section_score = int((found_sections_count / len(STANDARD_SECTIONS)) * 100)
    composite_score = round(
        0.35 * section_score +
        0.25 * action_verb_score +
        0.25 * metric_score +
        0.15 * length_score
    )

    # 6. Actionable Recommendations
    recommendations = []
    if not sections_status.get("Contact Info"):
        recommendations.append("Add clear contact information (Email, Phone, and LinkedIn/GitHub links).")
    if not sections_status.get("Experience / Work History"):
        recommendations.append("Include an explicit 'Work Experience' or 'Professional Experience' section header.")
    if not sections_status.get("Education"):
        recommendations.append("Add an explicit 'Education' section with your degrees or relevant institutions.")
    if not sections_status.get("Skills & Competencies"):
        recommendations.append("Add a dedicated 'Skills' section so ATS scanners can parse keywords easily.")
    if action_verb_count < 5:
        recommendations.append("Incorporate more strong action verbs (e.g., 'Spearheaded', 'Architected', 'Optimized', 'Automated').")
    if metric_count < 3:
        recommendations.append("Quantify your accomplishments with measurable metrics (e.g., 'reduced query latency by 40%', 'managed $50k budget').")
    if word_count < 350:
        recommendations.append("Elaborate on your project contributions and technologies to reach the ideal 400–900 word window.")

    if not recommendations:
        recommendations.append("Great job! Your resume demonstrates strong ATS structural hygiene and impact phrasing.")

    return {
        "overall_score": composite_score,
        "word_count": word_count,
        "length_status": length_status,
        "length_note": length_note,
        "sections": sections_status,
        "action_verbs_found": found_action_verbs,
        "action_verb_count": action_verb_count,
        "metric_count": metric_count,
        "recommendations": recommendations,
    }

# ==============================================================================
# 6. COMPREHENSIVE PAIR ANALYSIS & PIPELINE
# ==============================================================================

def analyze_pair(resume: str, job: str) -> Dict:
    errors = validate_pair(resume, job)
    if errors:
        raise ValueError(" ".join(errors))

    start = time.perf_counter()
    keyword = keyword_method(resume, job)
    keyword_time = time.perf_counter() - start

    start = time.perf_counter()
    tfidf_details = calculate_detailed_tfidf(resume, job)
    tfidf_score = tfidf_details["score"]
    tfidf_time = time.perf_counter() - start

    start = time.perf_counter()
    bm25_details = calculate_okapi_bm25(resume, job)
    bm25_score = bm25_details["score"]
    bm25_time = time.perf_counter() - start

    start = time.perf_counter()
    levenshtein_details = calculate_levenshtein_similarity(resume, job)
    lev_score = levenshtein_details["score"]
    lev_time = time.perf_counter() - start

    start = time.perf_counter()
    stuffing_details = detect_keyword_stuffing(resume, keyword["matched_skills"])
    stuffing_time = time.perf_counter() - start

    # ATS hygiene analysis on the resume
    ats_hygiene = analyze_ats_hygiene(resume)

    skill_score = keyword["score"]

    under_the_hood = {
        "keyword": {
            "matched_count": len(keyword["matched_skills"]),
            "job_count": keyword["job_skill_count"],
            "score": None if skill_score is None else round(skill_score, 2),
            "formula": f"({len(keyword['matched_skills'])} / {keyword['job_skill_count']}) * 100 = {skill_score:.1f}%" if keyword["job_skill_count"] else "N/A",
            "explanation": "Extracts hard competencies using boundary-aware regex + alias normalization. This is the primary algorithm commercial tools like Jobscan use.",
            "jobscan_flaw": "Jobscan stops here. If you miss exact keywords or use different phrasing, Jobscan rejects the resume even if you have equivalent experience."
        },
        "tfidf": {
            "score": round(tfidf_score, 2),
            "vocab_size": tfidf_details["vocab_size"],
            "shared_term_count": tfidf_details["shared_term_count"],
            "dot_product": tfidf_details["dot_product"],
            "resume_norm": tfidf_details["resume_norm"],
            "job_norm": tfidf_details["job_norm"],
            "top_terms": tfidf_details["top_shared_terms"],
            "formula": "Cosine(u, v) = (u · v) / (||u|| * ||v||)",
            "explanation": "Constructs high-dimensional stemmed n-gram term frequency-inverse document frequency vectors across both documents and measures the geometric angle.",
            "why_different": "Penalizes keyword stuffing because repeated words get sublinear dampening and words common to both documents receive lower IDF weighting."
        },
        "bm25": {
            "score": round(bm25_score, 2),
            "raw_score": bm25_details["raw_score"],
            "max_score": bm25_details.get("max_score", 0.0),
            "k_factor": bm25_details.get("k_factor", 0.0),
            "avgdl": bm25_details.get("avgdl", 350.0),
            "k1": bm25_details["k1"],
            "b": bm25_details["b"],
            "doc_length": bm25_details["doc_length"],
            "matched_terms": bm25_details["matched_terms"],
            "formula": "BM25(D, Q) = sum IDF(t) * (f(t, D) * (k1 + 1)) / (f(t, D) + k1 * (1 - b + b * (|D| / avgdl)))",
            "explanation": "Industry-standard search ranking algorithm with document length normalization (b=0.75) and term frequency saturation (k1=1.5).",
            "why_different": "Prevents longer resumes from artificially dominating and stops repeated keywords once term saturation is reached."
        },
        "levenshtein": {
            "score": round(lev_score, 2),
            "matched_pairs_count": levenshtein_details["matched_pairs_count"],
            "matched_pairs": levenshtein_details["matched_pairs"],
            "formula": "Sim(s1, s2) = 1 - (LevenshteinDistance(s1, s2) / max(len(s1), len(s2)))",
            "explanation": "Dynamic programming minimum edit-distance measuring character insertions, deletions, and substitutions between query and resume tokens.",
            "why_different": "Catches morphological variants and spelling variations (e.g. 'engineer'/'engineering', 'automate'/'automation') without rigid exact matching."
        },
        "anti_stuffing": {
            "authenticity_score": stuffing_details["authenticity_score"],
            "risk_level": stuffing_details["risk_level"],
            "verdict": stuffing_details["verdict"],
            "action_verbs_found": stuffing_details["action_verbs_found"],
            "skill_count": stuffing_details.get("skill_count", 0),
            "skill_density_pct": stuffing_details["skill_density_pct"],
            "total_penalty": stuffing_details.get("total_penalty", 0),
            "penalty_breakdown": stuffing_details.get("penalty_breakdown", []),
            "flags": stuffing_details["flags"],
            "explanation": "Audits the grammatical coupling between keywords and action verbs. Flags keyword dumping, dense lists, and white-text tricks that fool Jobscan.",
            "why_different": "Jobscan is easily fooled by pasting 20 keywords in a footer. Our algorithm detects the lack of action verbs and penalizes the stuffing attempt!"
        }
    }

    return {
        "scores": {
            "keyword_coverage": None if skill_score is None else round(skill_score, 2),
            "tfidf_similarity": round(tfidf_score, 2),
            "bm25_score": round(bm25_score, 2),
            "levenshtein_similarity": round(lev_score, 2),
            "authenticity_score": round(stuffing_details["authenticity_score"], 2),
        },
        "matched_skills": keyword["matched_skills"],
        "missing_skills": keyword["missing_skills"],
        "job_skill_count": keyword["job_skill_count"],
        "categorized_matched": keyword["categorized_matched"],
        "categorized_missing": keyword["categorized_missing"],
        "ats_hygiene": ats_hygiene,
        "tfidf_details": tfidf_details,
        "bm25_details": bm25_details,
        "levenshtein_details": levenshtein_details,
        "keyword_stuffing_details": stuffing_details,
        "under_the_hood": under_the_hood,
        "runtime_seconds": {
            "Keyword / skill coverage": round(keyword_time, 6),
            "TF-IDF cosine similarity": round(tfidf_time, 6),
            "Okapi BM25 ranking": round(bm25_time, 6),
            "Semantic Levenshtein similarity": round(lev_time, 6),
            "Adversarial anti-stuffing audit": round(stuffing_time, 6),
        },
        "notes": [
            "Keyword score reflects detected required-skill coverage with alias resolution.",
            "TF-IDF measures document-wide lexical term frequency & inverse document frequency angle.",
            "Okapi BM25 ranks relevance using document length normalization and term saturation.",
            "Levenshtein similarity measures fuzzy sub-word and morphological edit distances.",
            "Anti-Stuffing score detects buzzword dumping and evaluates authentic action-verb coupling.",
        ],
    }

# ==============================================================================
# 7. EVALUATION LAB ENGINE
# ==============================================================================

def load_evaluation_data(csv_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    required = {"resume_text", "job_description", "relevance_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("CSV is missing required column(s): " + ", ".join(sorted(missing)))

    df = df[["resume_text", "job_description", "relevance_label"]].copy()
    df = df.dropna(subset=["resume_text", "job_description", "relevance_label"])
    df["resume_text"] = df["resume_text"].astype(str).str.strip()
    df["job_description"] = df["job_description"].astype(str).str.strip()

    labels = pd.to_numeric(df["relevance_label"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("relevance_label must contain only 0 or 1 (0 = not relevant, 1 = relevant).")
    df["relevance_label"] = labels.astype(int)
    df = df[(df["resume_text"].str.len() >= 30) & (df["job_description"].str.len() >= 30)]
    if len(df) < 4:
        raise ValueError("After removing empty/short rows, provide at least 4 valid labeled pairs.")
    if df["relevance_label"].nunique() < 2:
        raise ValueError("Evaluation requires both label classes: 0 (not relevant) and 1 (relevant).")
    return df.reset_index(drop=True)

def _safe_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }

def evaluate_dataset(df: pd.DataFrame, threshold: float = 50.0) -> Dict:
    required = {"resume_text", "job_description", "relevance_label"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain resume_text, job_description, relevance_label.")
    if not 0.0 <= float(threshold) <= 100.0:
        raise ValueError("Threshold must be between 0 and 100.")
    if df.empty:
        raise ValueError("The evaluation dataset is empty.")
    if df["relevance_label"].isna().any() or not set(pd.to_numeric(df["relevance_label"], errors="coerce").dropna().unique()).issubset({0, 1}):
        raise ValueError("relevance_label must contain only 0 or 1.")
    if df["relevance_label"].nunique() < 2:
        raise ValueError("Evaluation requires both label classes: 0 (not relevant) and 1 (relevant).")

    predictions = []
    metric_rows = []
    ranking_rows = []

    method_scores = {
        "Keyword / skill coverage": [],
        "TF-IDF cosine similarity": [],
        "Okapi BM25 ranking": [],
        "Semantic Levenshtein similarity": [],
    }
    y_true = df["relevance_label"].astype(int).tolist()

    for idx, row in df.reset_index(drop=True).iterrows():
        resume = str(row["resume_text"])
        job = str(row["job_description"])
        if len(resume.strip()) < 30 or len(job.strip()) < 30:
            raise ValueError(f"Row {idx + 1} has resume/job text shorter than 30 characters.")

        keyword = keyword_method(resume, job)
        tfidf_score = tfidf_method(resume, job)
        bm25_res = calculate_okapi_bm25(resume, job)
        bm25_score = bm25_res["score"]
        lev_res = calculate_levenshtein_similarity(resume, job)
        lev_score = lev_res["score"]

        row_scores = {
            "Keyword / skill coverage": keyword["score"],
            "TF-IDF cosine similarity": tfidf_score,
            "Okapi BM25 ranking": bm25_score,
            "Semantic Levenshtein similarity": lev_score,
        }
        for method_name, score in row_scores.items():
            method_scores[method_name].append(0.0 if score is None else float(score))
            predicted = int((0.0 if score is None else score) >= threshold)
            predictions.append({
                "Row": idx + 1,
                "Method": method_name,
                "Human label": int(row["relevance_label"]),
                "Predicted label": predicted,
                "Score (%)": None if score is None else round(float(score), 2),
                "Correct?": predicted == int(row["relevance_label"]),
                "Matched skills": ", ".join(keyword["matched_skills"]),
                "Missing skills": ", ".join(keyword["missing_skills"]),
            })

    pred_df = pd.DataFrame(predictions)
    for method_name in method_scores:
        sub = pred_df[pred_df["Method"] == method_name]
        metric_rows.append({
            "Method": method_name,
            **_safe_metrics(
                sub["Human label"].astype(int).tolist(),
                sub["Predicted label"].astype(int).tolist(),
            ),
            "Threshold (%)": threshold,
        })
        scores = method_scores[method_name]
        if len(set(y_true)) > 1 and len(set(scores)) > 1:
            rho = float(spearmanr(scores, y_true).statistic)
        else:
            rho = float("nan")
        ranking_rows.append({
            "Method": method_name,
            "Spearman correlation": None if not np.isfinite(rho) else round(rho, 4),
            "Interpretation": "Higher positive values indicate stronger agreement with human relevance ordering.",
        })

    return {
        "metrics": metric_rows,
        "ranking_metrics": ranking_rows,
        "predictions": predictions,
        "threshold": threshold,
        "sample_count": len(df),
        "evaluation_note": (
            "Metrics benchmark the information retrieval methods against human relevance ground truth."
        ),
    }

    pred_df = pd.DataFrame(predictions)
    for method_name in method_scores:
        sub = pred_df[pred_df["Method"] == method_name]
        metric_rows.append({
            "Method": method_name,
            **_safe_metrics(
                sub["Human label"].astype(int).tolist(),
                sub["Predicted label"].astype(int).tolist(),
            ),
            "Threshold (%)": threshold,
        })
        scores = method_scores[method_name]
        if len(set(y_true)) > 1 and len(set(scores)) > 1:
            rho = float(spearmanr(scores, y_true).statistic)
        else:
            rho = float("nan")
        ranking_rows.append({
            "Method": method_name,
            "Spearman correlation": None if not np.isfinite(rho) else round(rho, 4),
            "Interpretation": "Higher positive values indicate stronger agreement with human relevance ordering.",
        })

    return {
        "metrics": metric_rows,
        "ranking_metrics": ranking_rows,
        "predictions": predictions,
        "threshold": threshold,
        "sample_count": len(df),
        "evaluation_note": (
            "Metrics are only as trustworthy as the labels and dataset. The 50% threshold is a prototype default; "
            "do not tune and report it on the same test data. Use separate validation and test sets for final claims."
        ),
    }

# ==============================================================================
# 8. BUILT-IN DEMO PRESET PROFILES
# ==============================================================================

SAMPLE_PROFILES: Dict[str, Dict[str, str]] = {
    "AI Prompt & Automation Engineer (Your Target Role)": {
        "resume": (
            "Rohan Sharma | rohan.sharma@email.com | (555) 789-0123 | github.com/rohansharma-ai\n\n"
            "PROFESSIONAL SUMMARY\n"
            "Creative AI Engineer & Solutions Architect with 3+ years of experience designing generative AI workflows, "
            "prompt-engineering pipelines, and automation solutions. Proven track record in rapid prototyping, API integrations, "
            "and deploying conversational chatbots.\n\n"
            "TECHNICAL SKILLS\n"
            "AI Models, Generative AI, Prompt Engineering, LLMs, Chatbots, AI Assistants, REST APIs, Database Integration, "
            "Python, SQL, Low-Code / No-Code, Workflow Automation, Rapid Prototyping, MVPs, Debugging, Testing, Optimization, "
            "Recommendation Systems, Git, Collaboration.\n\n"
            "PROFESSIONAL EXPERIENCE\n"
            "AI Solutions Developer | Nexus Automation Labs (2022 – Present)\n"
            "• Designed and tested AI prompts and prompt-engineering workflows across LLM models, improving response accuracy by 35%.\n"
            "• Built intelligent automation workflows connecting third-party platforms via REST APIs and webhooks, reducing repetitive manual tasks by 45 hours/week.\n"
            "• Integrated relational and vector databases with custom AI models to power interactive customer support chatbots and AI assistants.\n"
            "• Built rapid MVPs and prototypes using low-code tools and Python microservices, shipping 8 production prototypes from concept to launch.\n"
            "• Collaborated with creative and product teams to experiment with AI-assisted development tools and content-generation systems.\n"
            "• Debugged, profiled, and optimized existing backend applications, improving system throughput by 28%.\n\n"
            "Junior Automation Engineer | FlowTech Innovations (2020 – 2022)\n"
            "• Automated recurring business processes using Python scripting, API integrations, and database triggers.\n"
            "• Tested and deployed recommendation systems and intelligent workflows for internal operations.\n\n"
            "EDUCATION\n"
            "B.S. in Computer Science & Engineering | Tech University (2016 – 2020)"
        ),
        "job": (
            "Job Title: AI & Automation Development Specialist\n\n"
            "Key Responsibilities:\n"
            "• Design and test AI prompts and prompt-engineering workflows\n"
            "• Build automation workflows to reduce repetitive manual tasks\n"
            "• Integrate APIs, databases, AI models, and third-party tools\n"
            "• Experiment with no-code / low-code and AI-assisted development tools\n"
            "• Create rapid MVPs and prototypes from ideas and concepts\n"
            "• Explore and implement new AI/technology tools\n"
            "• Work on chatbots, AI assistants, recommendation systems, content-generation tools, and intelligent workflows\n"
            "• Connect different platforms and services through APIs and automation\n"
            "• Debug, improve, and optimize existing applications and workflows\n"
            "• Collaborate with creative teams to bring technology-driven ideas to life"
        )
    },
    "AI / Machine Learning Engineer (High Match)": {
        "resume": (
            "Alex Morgan | alex.morgan@email.com | (555) 234-5678 | github.com/alexmorgan-ai\n\n"
            "PROFESSIONAL SUMMARY\n"
            "Experienced Machine Learning Engineer with 4+ years of expertise in NLP, transformer models, "
            "and end-to-end MLOps pipeline deployment on AWS cloud infrastructure.\n\n"
            "CORE SKILLS\n"
            "Python, PyTorch, TensorFlow, Hugging Face, Transformers, NLP, Scikit-Learn, Pandas, NumPy, "
            "Docker, Kubernetes, AWS, FastAPI, SQL, PostgreSQL, Git, CI/CD, Agile.\n\n"
            "WORK EXPERIENCE\n"
            "Senior Machine Learning Engineer | DataCore Solutions (2021 – Present)\n"
            "• Architected and deployed production NLP search service using Sentence Transformers and FastAPI, reducing search latency by 42% for 200k daily queries.\n"
            "• Fine-tuned large language models (LLMs) and transformer classifiers using PyTorch, achieving 94.2% F1 score across 1.2M document datasets.\n"
            "• Spearheaded containerized deployment using Docker and Kubernetes (k8s) on AWS EKS, establishing CI/CD automation pipelines.\n"
            "• Optimized database indexing and vector storage with PostgreSQL and ChromaDB, saving $18,000 annually in infrastructure costs.\n\n"
            "Machine Learning Specialist | CloudScale Analytics (2019 – 2021)\n"
            "• Developed predictive modeling pipelines using Python, scikit-learn, and pandas to analyze customer churn.\n"
            "• Collaborated with cross-functional engineering teams in an Agile scrum framework to deliver model endpoints.\n\n"
            "EDUCATION\n"
            "B.S. in Computer Science | University of Technology (2015 – 2019)\n"
            "• Dean's Honor List, coursework in Data Structures, Algorithms, Statistics, and Machine Learning."
        ),
        "job": (
            "Job Title: Senior Machine Learning Engineer\n"
            "Location: Remote / Hybrid\n\n"
            "About the Role:\n"
            "We are seeking an experienced Machine Learning Engineer with strong expertise in Natural Language Processing (NLP) "
            "and deep learning to design, train, and deploy production AI models at scale.\n\n"
            "Key Responsibilities:\n"
            "• Build and optimize NLP, LLM, and transformer pipelines using Python and PyTorch or TensorFlow.\n"
            "• Package and deploy microservices with FastAPI and Docker into AWS cloud environments.\n"
            "• Manage container orchestration with Kubernetes and automate testing via CI/CD pipelines.\n"
            "• Query relational data using SQL and design scalable vector search solutions.\n"
            "• Collaborate across teams using Agile methodologies and conduct peer code reviews.\n\n"
            "Required Qualifications:\n"
            "• 3+ years of professional experience in Machine Learning and Python.\n"
            "• Demonstrated proficiency with PyTorch, Transformers, and Scikit-Learn.\n"
            "• Hands-on experience with Docker, AWS, and Git version control.\n"
            "• Solid foundation in SQL, data structures, and REST API development."
        )
    },
    "Full-Stack Web Developer (Partial Match)": {
        "resume": (
            "Jordan Lee | jordan.lee@email.com | (555) 876-5432 | github.com/jordanlee-dev\n\n"
            "PROFESSIONAL SUMMARY\n"
            "Dynamic Full-Stack Web Developer specializing in modern frontend ecosystems, React, TypeScript, and Node.js microservices.\n\n"
            "TECHNICAL SKILLS\n"
            "JavaScript, TypeScript, React, Next.js, Node.js, Express, HTML, CSS, Tailwind CSS, PostgreSQL, MongoDB, Git, REST API, Agile, Jira.\n\n"
            "PROFESSIONAL EXPERIENCE\n"
            "Full-Stack Developer | WebCraft Digital (2022 – Present)\n"
            "• Built responsive e-commerce web applications using React, Next.js, and TypeScript, serving 50,000+ active monthly shoppers.\n"
            "• Engineered backend REST API endpoints using Node.js and Express, integrating with PostgreSQL databases.\n"
            "• Automated front-end testing using Jest and unit testing suites, decreasing production bug tickets by 30%.\n"
            "• Collaborated with product designers using Figma and Agile sprints to deliver 12 major feature releases.\n\n"
            "Frontend Engineer | AppVibe Media (2020 – 2022)\n"
            "• Designed interactive user dashboards using JavaScript, React, and Tailwind CSS.\n"
            "• Managed Git version control workflows and performed regular code reviews.\n\n"
            "EDUCATION\n"
            "B.S. in Information Technology | State University (2016 – 2020)"
        ),
        "job": (
            "Job Title: Senior Full-Stack Cloud Engineer\n"
            "Location: San Francisco, CA (or Remote)\n\n"
            "Role Overview:\n"
            "We are looking for a Senior Full-Stack Engineer to architect enterprise web applications. The candidate should be "
            "proficient with React and TypeScript on the frontend, and Python or Go backend services hosted on AWS and Kubernetes.\n\n"
            "Requirements:\n"
            "• Strong hands-on experience with React, TypeScript, and modern JavaScript.\n"
            "• Backend proficiency with Python (FastAPI or Django) or Go.\n"
            "• Production experience with AWS, Docker, Kubernetes, and CI/CD pipelines.\n"
            "• Familiarity with PostgreSQL, Redis caching, and GraphQL.\n"
            "• Strong communication skills and experience with system design."
        )
    },
    "Retail Associate vs Cloud Architect (Mismatch / Negative Case)": {
        "resume": (
            "Chris Miller | chris.miller@email.com | (555) 432-1098\n\n"
            "SUMMARY\n"
            "Customer-oriented Retail Supervisor with 5 years of experience in retail store inventory management, "
            "cash register operations, and customer service excellence.\n\n"
            "KEY SKILLS\n"
            "Inventory Management, Customer Service, Cash Handling, Sales Presentations, Teamwork, Microsoft Excel, Problem Solving.\n\n"
            "EXPERIENCE\n"
            "Shift Supervisor | Metro Retail Store (2021 – Present)\n"
            "• Managed daily inventory counts and prepared weekly replenishment reports in Excel.\n"
            "• Supervised a team of 8 sales associates and trained new cashiers on store policies.\n"
            "• Resolved customer complaints, increasing customer satisfaction scores by 18%.\n\n"
            "Sales Associate | City Mart (2019 – 2021)\n"
            "• Handled daily cash register transactions totaling over $12,000 per shift.\n"
            "• Assisted shoppers with merchandise inquiries and maintained organized display shelves.\n\n"
            "EDUCATION\n"
            "High School Diploma | Central High School (2018)"
        ),
        "job": (
            "Job Title: Principal Cloud Infrastructure Architect\n"
            "Location: New York, NY\n\n"
            "Overview:\n"
            "Seeking an experienced Cloud Architect to lead enterprise migration to AWS and Azure. "
            "You will design secure, multi-region cloud landing zones and automate infrastructure using Terraform.\n\n"
            "Requirements:\n"
            "• 7+ years designing enterprise cloud architectures on AWS and Azure.\n"
            "• Deep mastery of Terraform, Kubernetes, Docker, and Linux systems administration.\n"
            "• Expertise in microservices architecture, network security, and CI/CD automation.\n"
            "• Proficiency in Python or Go for infrastructure scripting."
        )
    }
}
