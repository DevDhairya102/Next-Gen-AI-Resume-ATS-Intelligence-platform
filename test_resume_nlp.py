"""Lightweight tests for core logic; semantic model is mocked to avoid downloads."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import resume_nlp


class FakeSentenceModel:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            lowered = text.lower()
            # Deterministic toy vectors only for smoke testing the pipeline.
            vectors.append(np.array([
                1.0 + lowered.count("python"),
                1.0 + lowered.count("machine learning"),
                1.0 + lowered.count("retail"),
            ], dtype=float))
        vectors = np.asarray(vectors)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        return vectors


class ResumeNlpTests(unittest.TestCase):
    def test_skill_coverage_and_java_boundary(self):
        result = resume_nlp.keyword_method(
            "Built a JavaScript web application and used Python for data work.",
            "Required skills: Java and Python for the project.",
        )
        self.assertIn("python", result["matched_skills"])
        self.assertIn("java", result["missing_skills"])
        self.assertEqual(result["score"], 50.0)

    def test_alias_resolution(self):
        # k8s should map to kubernetes, and aws should match
        result = resume_nlp.keyword_method(
            "Deployed microservices using k8s and amazon web services.",
            "Seeking engineer with Kubernetes and AWS experience.",
        )
        self.assertIn("kubernetes", result["matched_skills"])
        self.assertIn("aws", result["matched_skills"])
        self.assertEqual(result["score"], 100.0)

    def test_tfidf_similarity_range(self):
        score = resume_nlp.tfidf_method(
            "Python data analysis using pandas and machine learning.",
            "Python developer with pandas and machine learning skills.",
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_tfidf_handles_empty_vocabulary(self):
        self.assertEqual(resume_nlp.tfidf_method("the and if", "or but the"), 0.0)

    def test_pair_validation(self):
        errors = resume_nlp.validate_pair("short", "also short")
        self.assertEqual(len(errors), 2)

    def test_ats_hygiene_analysis(self):
        sample_resume = (
            "Jane Doe | jane@example.com | 555-123-4567\n"
            "PROFESSIONAL SUMMARY\n"
            "Experienced software engineer.\n"
            "WORK EXPERIENCE\n"
            "• Spearheaded backend architecture and optimized database queries by 45%.\n"
            "• Built microservices in Python serving 100k daily users.\n"
            "EDUCATION\n"
            "B.S. in Computer Science\n"
            "SKILLS\n"
            "Python, SQL, Docker"
        )
        hygiene = resume_nlp.analyze_ats_hygiene(sample_resume)
        self.assertGreater(hygiene["overall_score"], 50)
        self.assertTrue(hygiene["sections"]["Experience / Work History"])
        self.assertTrue(hygiene["sections"]["Education"])
        self.assertTrue(hygiene["sections"]["Skills & Competencies"])
        self.assertIn("spearheaded", hygiene["action_verbs_found"])
        self.assertGreaterEqual(hygiene["metric_count"], 1)

    def test_categorize_skills(self):
        skills = ["python", "react", "docker", "postgresql", "agile"]
        categorized = resume_nlp.categorize_skills(skills)
        self.assertIn("Programming Languages", categorized)
        self.assertIn("Frameworks & Web Tech", categorized)
        self.assertIn("Cloud, DevOps & Infrastructure", categorized)
        self.assertIn("Databases & Storage", categorized)

    def test_okapi_bm25(self):
        res = resume_nlp.calculate_okapi_bm25("Python automated workflow with APIs", "Python API workflow automation")
        self.assertGreaterEqual(res["score"], 0.0)
        self.assertLessEqual(res["score"], 100.0)
        self.assertGreater(res["raw_score"], 0.0)

    def test_levenshtein_similarity(self):
        res = resume_nlp.calculate_levenshtein_similarity("Software engineering and automation", "Software engineer automating workflows")
        self.assertGreaterEqual(res["score"], 0.0)
        self.assertLessEqual(res["score"], 100.0)

    def test_keyword_stuffing_detection(self):
        authentic = "Architected and deployed microservices using Docker and Python, reducing latency by 35%."
        res_auth = resume_nlp.detect_keyword_stuffing(authentic, ["docker", "python", "microservices"])
        self.assertGreaterEqual(res_auth["authenticity_score"], 70)
        self.assertEqual(res_auth["risk_level"], "Authentic / Low Risk")

        stuffed = "Python, Java, C++, Kubernetes, Docker, AWS, Azure, GCP, React, Node, SQL, MongoDB, Redis, Kafka"
        res_stuffed = resume_nlp.detect_keyword_stuffing(stuffed, ["python", "java", "c++", "docker", "aws"])
        self.assertLess(res_stuffed["authenticity_score"], 80)

    def test_evaluation_pipeline(self):
        data = pd.DataFrame([
            {"resume_text": "Python machine learning data project experience", "job_description": "Python machine learning developer role", "relevance_label": 1},
            {"resume_text": "Retail inventory cashier customer service experience", "job_description": "Python machine learning developer role", "relevance_label": 0},
            {"resume_text": "Python data cleaning pandas project experience", "job_description": "Python data analyst with pandas skills", "relevance_label": 1},
            {"resume_text": "Graphic design marketing social media experience", "job_description": "Python data analyst with pandas skills", "relevance_label": 0},
        ])
        result = resume_nlp.evaluate_dataset(data, threshold=50)
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(len(result["metrics"]), 4)
        self.assertEqual(len(result["predictions"]), 16)

    def test_analyze_pair_enriched_fields(self):
        resume = "Architected and built microservices in Python using Docker, optimizing database query performance."
        job = "Seeking Python engineer with Docker and database optimization experience."
        res = resume_nlp.analyze_pair(resume, job)
        self.assertIn("scores", res)
        self.assertIn("under_the_hood", res)
        uth = res["under_the_hood"]
        self.assertIn("keyword", uth)
        self.assertIn("tfidf", uth)
        self.assertIn("bm25", uth)
        self.assertIn("levenshtein", uth)
        self.assertIn("anti_stuffing", uth)
        self.assertIn("k_factor", uth["bm25"])
        self.assertIn("penalty_breakdown", uth["anti_stuffing"])

    def test_evaluation_rejects_invalid_threshold(self):
        data = pd.DataFrame([
            {"resume_text": "Python machine learning data project experience", "job_description": "Python machine learning developer role", "relevance_label": 1},
            {"resume_text": "Retail inventory cashier customer service experience", "job_description": "Python machine learning developer role", "relevance_label": 0},
        ])
        with self.assertRaises(ValueError):
            resume_nlp.evaluate_dataset(data, threshold=120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
