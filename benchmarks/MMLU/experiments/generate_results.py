"""
generate_results.py
-------------------
Parses raw MMLU evaluation output text blocks (pasted from terminal) and writes:
  - One CSV per experiment run  (benchmarks/MMLU/results/<run_name>.csv)
  - A combined summary Excel file (benchmarks/MMLU/results/summary.xlsx)
    with two sheets: "All Runs Summary" and "Per-Question Detail"

Usage:
  python benchmarks/MMLU/experiments/generate_results.py
"""

import os
import re
import pandas as pd

# ---------------------------------------------------------------------------
# Raw output blocks – one entry per experiment run.
# Each dict has:
#   provider, model, subject, shots, shuffle, limit, accuracy, raw_text
# ---------------------------------------------------------------------------

RUNS = [

# ── Run 1 ──────────────────────────────────────────────────────────────────
{
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "subject": "anatomy",
  "shots": 0,
  "shuffle": False,
  "limit": 10,
  "accuracy": 80.0,
  "raw_text": """
Q1: GroundTruth=A | Predicted=A | Correct
Q2: GroundTruth=B | Predicted=B | Correct
Q3: GroundTruth=A | Predicted=A | Correct
Q4: GroundTruth=C | Predicted=C | Correct
Q5: GroundTruth=B | Predicted=B | Correct
Q6: GroundTruth=B | Predicted=A | Incorrect
Q7: GroundTruth=A | Predicted=A | Correct
Q8: GroundTruth=B | Predicted=B | Correct
Q9: GroundTruth=C | Predicted=B | Incorrect
Q10: GroundTruth=D | Predicted=D | Correct
"""
},

# ── Run 2 ──────────────────────────────────────────────────────────────────
{
  "provider": "ollama",
  "model": "llama3.2",
  "subject": "global_facts",
  "shots": 0,
  "shuffle": False,
  "limit": 10,
  "accuracy": 30.0,
  "raw_text": """
Q1: GroundTruth=C | Predicted=C | Correct
Q2: GroundTruth=B | Predicted=C | Incorrect
Q3: GroundTruth=C | Predicted=B | Incorrect
Q4: GroundTruth=A | Predicted=A | Correct
Q5: GroundTruth=C | Predicted=B | Incorrect
Q6: GroundTruth=A | Predicted=A | Correct
Q7: GroundTruth=C | Predicted=B | Incorrect
Q8: GroundTruth=C | Predicted=B | Incorrect
Q9: GroundTruth=C | Predicted=B | Incorrect
Q10: GroundTruth=B | Predicted=C | Incorrect
"""
},

# ── Run 3 ──────────────────────────────────────────────────────────────────
{
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "subject": "global_facts",
  "shots": 0,
  "shuffle": False,
  "limit": 10,
  "accuracy": 50.0,
  "raw_text": """
Q1: GroundTruth=C | Predicted=C | Correct
Q2: GroundTruth=B | Predicted=A | Incorrect
Q3: GroundTruth=C | Predicted=B | Incorrect
Q4: GroundTruth=A | Predicted=A | Correct
Q5: GroundTruth=C | Predicted=B | Incorrect
Q6: GroundTruth=A | Predicted=A | Correct
Q7: GroundTruth=C | Predicted=C | Correct
Q8: GroundTruth=C | Predicted=A | Incorrect
Q9: GroundTruth=C | Predicted=B | Incorrect
Q10: GroundTruth=B | Predicted=B | Correct
"""
},

# ── Run 4 ──────────────────────────────────────────────────────────────────
{
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "subject": "global_facts",
  "shots": 0,
  "shuffle": False,
  "limit": 100,
  "accuracy": 37.0,
  "raw_text": """
Q1: GroundTruth=C | Predicted=C | Correct
Q2: GroundTruth=B | Predicted=A | Incorrect
Q3: GroundTruth=C | Predicted=B | Incorrect
Q4: GroundTruth=A | Predicted=A | Correct
Q5: GroundTruth=C | Predicted=B | Incorrect
Q6: GroundTruth=A | Predicted=A | Correct
Q7: GroundTruth=C | Predicted=C | Correct
Q8: GroundTruth=C | Predicted=A | Incorrect
Q9: GroundTruth=C | Predicted=B | Incorrect
Q10: GroundTruth=B | Predicted=B | Correct
Q11: GroundTruth=D | Predicted=B | Incorrect
Q12: GroundTruth=B | Predicted=A | Incorrect
Q13: GroundTruth=C | Predicted=A | Incorrect
Q14: GroundTruth=A | Predicted=A | Correct
Q15: GroundTruth=B | Predicted=A | Incorrect
Q16: GroundTruth=B | Predicted=B | Correct
Q17: GroundTruth=A | Predicted=B | Incorrect
Q18: GroundTruth=C | Predicted=B | Incorrect
Q19: GroundTruth=C | Predicted=A | Incorrect
Q20: GroundTruth=B | Predicted=B | Correct
Q21: GroundTruth=A | Predicted=A | Correct
Q22: GroundTruth=A | Predicted=A | Correct
Q23: GroundTruth=D | Predicted=B | Incorrect
Q24: GroundTruth=B | Predicted=B | Correct
Q25: GroundTruth=D | Predicted=B | Incorrect
Q26: GroundTruth=D | Predicted=B | Incorrect
Q27: GroundTruth=D | Predicted=B | Incorrect
Q28: GroundTruth=C | Predicted=B | Incorrect
Q29: GroundTruth=D | Predicted=B | Incorrect
Q30: GroundTruth=C | Predicted=C | Correct
Q31: GroundTruth=C | Predicted=A | Incorrect
Q32: GroundTruth=A | Predicted=B | Incorrect
Q33: GroundTruth=B | Predicted=C | Incorrect
Q34: GroundTruth=C | Predicted=C | Correct
Q35: GroundTruth=D | Predicted=C | Incorrect
Q36: GroundTruth=D | Predicted=B | Incorrect
Q37: GroundTruth=D | Predicted=A | Incorrect
Q38: GroundTruth=B | Predicted=B | Correct
Q39: GroundTruth=D | Predicted=C | Incorrect
Q40: GroundTruth=C | Predicted=B | Incorrect
Q41: GroundTruth=B | Predicted=B | Correct
Q42: GroundTruth=A | Predicted=C | Incorrect
Q43: GroundTruth=C | Predicted=A | Incorrect
Q44: GroundTruth=A | Predicted=B | Incorrect
Q45: GroundTruth=C | Predicted=A | Incorrect
Q46: GroundTruth=C | Predicted=C | Correct
Q47: GroundTruth=A | Predicted=B | Incorrect
Q48: GroundTruth=B | Predicted=B | Correct
Q49: GroundTruth=C | Predicted=B | Incorrect
Q50: GroundTruth=B | Predicted=C | Incorrect
Q51: GroundTruth=B | Predicted=B | Correct
Q52: GroundTruth=B | Predicted=B | Correct
Q53: GroundTruth=B | Predicted=B | Correct
Q54: GroundTruth=D | Predicted=C | Incorrect
Q55: GroundTruth=A | Predicted=B | Incorrect
Q56: GroundTruth=B | Predicted=D | Incorrect
Q57: GroundTruth=D | Predicted=B | Incorrect
Q58: GroundTruth=D | Predicted=B | Incorrect
Q59: GroundTruth=C | Predicted=D | Incorrect
Q60: GroundTruth=B | Predicted=B | Correct
Q61: GroundTruth=C | Predicted=C | Correct
Q62: GroundTruth=A | Predicted=A | Correct
Q63: GroundTruth=B | Predicted=C | Incorrect
Q64: GroundTruth=C | Predicted=B | Incorrect
Q65: GroundTruth=B | Predicted=B | Correct
Q66: GroundTruth=C | Predicted=C | Correct
Q67: GroundTruth=A | Predicted=A | Correct
Q68: GroundTruth=B | Predicted=A | Incorrect
Q69: GroundTruth=C | Predicted=B | Incorrect
Q70: GroundTruth=B | Predicted=C | Incorrect
Q71: GroundTruth=B | Predicted=B | Correct
Q72: GroundTruth=A | Predicted=B | Incorrect
Q73: GroundTruth=B | Predicted=B | Correct
Q74: GroundTruth=C | Predicted=C | Correct
Q75: GroundTruth=C | Predicted=C | Correct
Q76: GroundTruth=B | Predicted=B | Correct
Q77: GroundTruth=C | Predicted=B | Incorrect
Q78: GroundTruth=B | Predicted=B | Correct
Q79: GroundTruth=A | Predicted=B | Incorrect
Q80: GroundTruth=D | Predicted=D | Correct
Q81: GroundTruth=D | Predicted=A | Incorrect
Q82: GroundTruth=B | Predicted=B | Correct
Q83: GroundTruth=A | Predicted=A | Correct
Q84: GroundTruth=C | Predicted=A | Incorrect
Q85: GroundTruth=C | Predicted=D | Incorrect
Q86: GroundTruth=D | Predicted=A | Incorrect
Q87: GroundTruth=C | Predicted=B | Incorrect
Q88: GroundTruth=C | Predicted=A | Incorrect
Q89: GroundTruth=A | Predicted=C | Incorrect
Q90: GroundTruth=B | Predicted=D | Incorrect
Q91: GroundTruth=A | Predicted=D | Incorrect
Q92: GroundTruth=D | Predicted=A | Incorrect
Q93: GroundTruth=D | Predicted=B | Incorrect
Q94: GroundTruth=B | Predicted=A | Incorrect
Q95: GroundTruth=D | Predicted=A | Incorrect
Q96: GroundTruth=D | Predicted=A | Incorrect
Q97: GroundTruth=B | Predicted=B | Correct
Q98: GroundTruth=C | Predicted=B | Incorrect
Q99: GroundTruth=D | Predicted=A | Incorrect
Q100: GroundTruth=D | Predicted=B | Incorrect
"""
},

# ── Run 5 ──────────────────────────────────────────────────────────────────
{
  "provider": "ollama",
  "model": "llama3.2",
  "subject": "global_facts",
  "shots": 0,
  "shuffle": False,
  "limit": 100,
  "accuracy": 35.0,
  "raw_text": """
Q1: GroundTruth=C | Predicted=C | Correct
Q2: GroundTruth=B | Predicted=C | Incorrect
Q3: GroundTruth=C | Predicted=B | Incorrect
Q4: GroundTruth=A | Predicted=A | Correct
Q5: GroundTruth=C | Predicted=B | Incorrect
Q6: GroundTruth=A | Predicted=A | Correct
Q7: GroundTruth=C | Predicted=B | Incorrect
Q8: GroundTruth=C | Predicted=B | Incorrect
Q9: GroundTruth=C | Predicted=B | Incorrect
Q10: GroundTruth=B | Predicted=C | Incorrect
Q11: GroundTruth=D | Predicted=D | Correct
Q12: GroundTruth=B | Predicted=B | Correct
Q13: GroundTruth=C | Predicted=C | Correct
Q14: GroundTruth=A | Predicted=A | Correct
Q15: GroundTruth=B | Predicted=C | Incorrect
Q16: GroundTruth=B | Predicted=D | Incorrect
Q17: GroundTruth=A | Predicted=B | Incorrect
Q18: GroundTruth=C | Predicted=B | Incorrect
Q19: GroundTruth=C | Predicted=C | Correct
Q20: GroundTruth=B | Predicted=B | Correct
Q21: GroundTruth=A | Predicted=C | Incorrect
Q22: GroundTruth=A | Predicted=C | Incorrect
Q23: GroundTruth=D | Predicted=B | Incorrect
Q24: GroundTruth=B | Predicted=C | Incorrect
Q25: GroundTruth=D | Predicted=B | Incorrect
Q26: GroundTruth=D | Predicted=B | Incorrect
Q27: GroundTruth=D | Predicted=C | Incorrect
Q28: GroundTruth=C | Predicted=B | Incorrect
Q29: GroundTruth=D | Predicted=B | Incorrect
Q30: GroundTruth=C | Predicted=B | Incorrect
Q31: GroundTruth=C | Predicted=B | Incorrect
Q32: GroundTruth=A | Predicted=A | Correct
Q33: GroundTruth=B | Predicted=C | Incorrect
Q34: GroundTruth=C | Predicted=C | Correct
Q35: GroundTruth=D | Predicted=D | Correct
Q36: GroundTruth=D | Predicted=C | Incorrect
Q37: GroundTruth=D | Predicted=A | Incorrect
Q38: GroundTruth=B | Predicted=B | Correct
Q39: GroundTruth=D | Predicted=C | Incorrect
Q40: GroundTruth=C | Predicted=D | Incorrect
Q41: GroundTruth=B | Predicted=C | Incorrect
Q42: GroundTruth=A | Predicted=B | Incorrect
Q43: GroundTruth=C | Predicted=B | Incorrect
Q44: GroundTruth=A | Predicted=C | Incorrect
Q45: GroundTruth=C | Predicted=A | Incorrect
Q46: GroundTruth=C | Predicted=C | Correct
Q47: GroundTruth=A | Predicted=B | Incorrect
Q48: GroundTruth=B | Predicted=C | Incorrect
Q49: GroundTruth=C | Predicted=C | Correct
Q50: GroundTruth=B | Predicted=C | Incorrect
Q51: GroundTruth=B | Predicted=B | Correct
Q52: GroundTruth=B | Predicted=B | Correct
Q53: GroundTruth=B | Predicted=C | Incorrect
Q54: GroundTruth=D | Predicted=C | Incorrect
Q55: GroundTruth=A | Predicted=D | Incorrect
Q56: GroundTruth=B | Predicted=C | Incorrect
Q57: GroundTruth=D | Predicted=C | Incorrect
Q58: GroundTruth=D | Predicted=B | Incorrect
Q59: GroundTruth=C | Predicted=C | Correct
Q60: GroundTruth=B | Predicted=B | Correct
Q61: GroundTruth=C | Predicted=C | Correct
Q62: GroundTruth=A | Predicted=D | Incorrect
Q63: GroundTruth=B | Predicted=C | Incorrect
Q64: GroundTruth=C | Predicted=C | Correct
Q65: GroundTruth=B | Predicted=C | Incorrect
Q66: GroundTruth=C | Predicted=C | Correct
Q67: GroundTruth=A | Predicted=A | Correct
Q68: GroundTruth=B | Predicted=A | Incorrect
Q69: GroundTruth=C | Predicted=C | Correct
Q70: GroundTruth=B | Predicted=C | Incorrect
Q71: GroundTruth=B | Predicted=B | Correct
Q72: GroundTruth=A | Predicted=B | Incorrect
Q73: GroundTruth=B | Predicted=B | Correct
Q74: GroundTruth=C | Predicted=C | Correct
Q75: GroundTruth=C | Predicted=C | Correct
Q76: GroundTruth=B | Predicted=C | Incorrect
Q77: GroundTruth=C | Predicted=C | Correct
Q78: GroundTruth=A | Predicted=C | Incorrect
Q79: GroundTruth=C | Predicted=A | Incorrect
Q80: GroundTruth=B | Predicted=B | Correct
Q81: GroundTruth=C | Predicted=C | Correct
Q82: GroundTruth=C | Predicted=C | Correct
Q83: GroundTruth=A | Predicted=C | Incorrect
Q84: GroundTruth=B | Predicted=C | Incorrect
Q85: GroundTruth=A | Predicted=B | Incorrect
Q86: GroundTruth=B | Predicted=A | Incorrect
Q87: GroundTruth=C | Predicted=C | Correct
Q88: GroundTruth=C | Predicted=A | Incorrect
Q89: GroundTruth=C | Predicted=A | Incorrect
Q90: GroundTruth=B | Predicted=A | Incorrect
Q91: GroundTruth=D | Predicted=B | Incorrect
Q92: GroundTruth=A | Predicted=B | Incorrect
Q93: GroundTruth=D | Predicted=B | Incorrect
Q94: GroundTruth=B | Predicted=C | Incorrect
Q95: GroundTruth=B | Predicted=A | Incorrect
Q96: GroundTruth=B | Predicted=C | Incorrect
Q97: GroundTruth=D | Predicted=D | Correct
Q98: GroundTruth=D | Predicted=D | Correct
Q99: GroundTruth=C | Predicted=B | Incorrect
Q100: GroundTruth=D | Predicted=B | Incorrect
"""
},

# ── Run 6 ──────────────────────────────────────────────────────────────────
{
  "provider": "ollama",
  "model": "llama3.2",
  "subject": "global_facts",
  "shots": 0,
  "shuffle": True,
  "limit": 100,
  "accuracy": 29.0,
  "raw_text": """
Q1: GroundTruth=A | Predicted=A | Correct
Q2: GroundTruth=D | Predicted=C | Incorrect
Q3: GroundTruth=B | Predicted=B | Correct
Q4: GroundTruth=B | Predicted=B | Correct
Q5: GroundTruth=A | Predicted=C | Incorrect
Q6: GroundTruth=A | Predicted=A | Correct
Q7: GroundTruth=B | Predicted=B | Correct
Q8: GroundTruth=D | Predicted=B | Incorrect
Q9: GroundTruth=B | Predicted=B | Correct
Q10: GroundTruth=B | Predicted=C | Incorrect
Q11: GroundTruth=A | Predicted=A | Correct
Q12: GroundTruth=B | Predicted=B | Correct
Q13: GroundTruth=B | Predicted=B | Correct
Q14: GroundTruth=B | Predicted=B | Correct
Q15: GroundTruth=A | Predicted=C | Incorrect
Q16: GroundTruth=D | Predicted=A | Incorrect
Q17: GroundTruth=A | Predicted=B | Incorrect
Q18: GroundTruth=B | Predicted=B | Correct
Q19: GroundTruth=A | Predicted=C | Incorrect
Q20: GroundTruth=A | Predicted=C | Incorrect
Q21: GroundTruth=B | Predicted=A | Incorrect
Q22: GroundTruth=A | Predicted=B | Incorrect
Q23: GroundTruth=A | Predicted=A | Correct
Q24: GroundTruth=A | Predicted=C | Incorrect
Q25: GroundTruth=D | Predicted=B | Incorrect
Q26: GroundTruth=D | Predicted=B | Incorrect
Q27: GroundTruth=A | Predicted=C | Incorrect
Q28: GroundTruth=C | Predicted=B | Incorrect
Q29: GroundTruth=B | Predicted=C | Incorrect
Q30: GroundTruth=B | Predicted=A | Incorrect
Q31: GroundTruth=D | Predicted=A | Incorrect
Q32: GroundTruth=D | Predicted=D | Correct
Q33: GroundTruth=A | Predicted=C | Incorrect
Q34: GroundTruth=C | Predicted=C | Correct
Q35: GroundTruth=C | Predicted=C | Correct
Q36: GroundTruth=A | Predicted=A | Correct
Q37: GroundTruth=B | Predicted=C | Incorrect
Q38: GroundTruth=A | Predicted=B | Incorrect
Q39: GroundTruth=D | Predicted=A | Incorrect
Q40: GroundTruth=B | Predicted=C | Incorrect
Q41: GroundTruth=A | Predicted=B | Incorrect
Q42: GroundTruth=B | Predicted=D | Incorrect
Q43: GroundTruth=A | Predicted=B | Incorrect
Q44: GroundTruth=D | Predicted=A | Incorrect
Q45: GroundTruth=C | Predicted=B | Incorrect
Q46: GroundTruth=D | Predicted=B | Incorrect
Q47: GroundTruth=D | Predicted=B | Incorrect
Q48: GroundTruth=B | Predicted=C | Incorrect
Q49: GroundTruth=D | Predicted=C | Incorrect
Q50: GroundTruth=C | Predicted=B | Incorrect
Q51: GroundTruth=C | Predicted=B | Incorrect
Q52: GroundTruth=D | Predicted=A | Incorrect
Q53: GroundTruth=B | Predicted=D | Incorrect
Q54: GroundTruth=C | Predicted=C | Correct
Q55: GroundTruth=A | Predicted=D | Incorrect
Q56: GroundTruth=A | Predicted=B | Incorrect
Q57: GroundTruth=A | Predicted=C | Incorrect
Q58: GroundTruth=B | Predicted=B | Correct
Q59: GroundTruth=B | Predicted=A | Incorrect
Q60: GroundTruth=D | Predicted=B | Incorrect
Q61: GroundTruth=D | Predicted=C | Incorrect
Q62: GroundTruth=C | Predicted=D | Incorrect
Q63: GroundTruth=D | Predicted=C | Incorrect
Q64: GroundTruth=B | Predicted=B | Correct
Q65: GroundTruth=B | Predicted=B | Correct
Q66: GroundTruth=B | Predicted=B | Correct
Q67: GroundTruth=D | Predicted=D | Correct
Q68: GroundTruth=B | Predicted=D | Incorrect
Q69: GroundTruth=C | Predicted=A | Incorrect
Q70: GroundTruth=A | Predicted=B | Incorrect
Q71: GroundTruth=C | Predicted=C | Correct
Q72: GroundTruth=B | Predicted=C | Incorrect
Q73: GroundTruth=A | Predicted=A | Correct
Q74: GroundTruth=D | Predicted=B | Incorrect
Q75: GroundTruth=A | Predicted=C | Incorrect
Q76: GroundTruth=B | Predicted=C | Incorrect
Q77: GroundTruth=D | Predicted=B | Incorrect
Q78: GroundTruth=B | Predicted=B | Correct
Q79: GroundTruth=A | Predicted=C | Incorrect
Q80: GroundTruth=D | Predicted=B | Incorrect
Q81: GroundTruth=D | Predicted=C | Incorrect
Q82: GroundTruth=B | Predicted=B | Correct
Q83: GroundTruth=A | Predicted=B | Incorrect
Q84: GroundTruth=C | Predicted=C | Correct
Q85: GroundTruth=C | Predicted=B | Incorrect
Q86: GroundTruth=D | Predicted=A | Incorrect
Q87: GroundTruth=C | Predicted=B | Incorrect
Q88: GroundTruth=C | Predicted=B | Incorrect
Q89: GroundTruth=A | Predicted=C | Incorrect
Q90: GroundTruth=B | Predicted=D | Incorrect
Q91: GroundTruth=A | Predicted=D | Incorrect
Q92: GroundTruth=D | Predicted=B | Incorrect
Q93: GroundTruth=D | Predicted=B | Incorrect
Q94: GroundTruth=B | Predicted=C | Incorrect
Q95: GroundTruth=D | Predicted=C | Incorrect
Q96: GroundTruth=D | Predicted=B | Incorrect
Q97: GroundTruth=B | Predicted=B | Correct
Q98: GroundTruth=C | Predicted=C | Correct
Q99: GroundTruth=D | Predicted=A | Incorrect
Q100: GroundTruth=D | Predicted=B | Incorrect
"""
},

# ── Run 7 ──────────────────────────────────────────────────────────────────
{
  "provider": "ollama",
  "model": "llama3.2",
  "subject": "global_facts",
  "shots": 5,
  "shuffle": False,
  "limit": 100,
  "accuracy": 31.0,
  "raw_text": """
Q1: GroundTruth=C | Predicted=A | Incorrect
Q2: GroundTruth=B | Predicted=C | Incorrect
Q3: GroundTruth=C | Predicted=A | Incorrect
Q4: GroundTruth=A | Predicted=A | Correct
Q5: GroundTruth=C | Predicted=B | Incorrect
Q6: GroundTruth=A | Predicted=A | Correct
Q7: GroundTruth=C | Predicted=B | Incorrect
Q8: GroundTruth=C | Predicted=B | Incorrect
Q9: GroundTruth=C | Predicted=B | Incorrect
Q10: GroundTruth=B | Predicted=B | Correct
Q11: GroundTruth=D | Predicted=B | Incorrect
Q12: GroundTruth=B | Predicted=B | Correct
Q13: GroundTruth=C | Predicted=C | Correct
Q14: GroundTruth=A | Predicted=C | Incorrect
Q15: GroundTruth=B | Predicted=A | Incorrect
Q16: GroundTruth=B | Predicted=A | Incorrect
Q17: GroundTruth=A | Predicted=B | Incorrect
Q18: GroundTruth=C | Predicted=B | Incorrect
Q19: GroundTruth=C | Predicted=B | Incorrect
Q20: GroundTruth=B | Predicted=B | Correct
Q21: GroundTruth=A | Predicted=B | Incorrect
Q22: GroundTruth=A | Predicted=B | Incorrect
Q23: GroundTruth=D | Predicted=B | Incorrect
Q24: GroundTruth=B | Predicted=A | Incorrect
Q25: GroundTruth=D | Predicted=B | Incorrect
Q26: GroundTruth=D | Predicted=B | Incorrect
Q27: GroundTruth=D | Predicted=A | Incorrect
Q28: GroundTruth=C | Predicted=B | Incorrect
Q29: GroundTruth=D | Predicted=B | Incorrect
Q30: GroundTruth=C | Predicted=B | Incorrect
Q31: GroundTruth=C | Predicted=B | Incorrect
Q32: GroundTruth=A | Predicted=A | Correct
Q33: GroundTruth=B | Predicted=B | Correct
Q34: GroundTruth=C | Predicted=B | Incorrect
Q35: GroundTruth=D | Predicted=B | Incorrect
Q36: GroundTruth=D | Predicted=B | Incorrect
Q37: GroundTruth=D | Predicted=A | Incorrect
Q38: GroundTruth=B | Predicted=B | Correct
Q39: GroundTruth=D | Predicted=B | Incorrect
Q40: GroundTruth=C | Predicted=A | Incorrect
Q41: GroundTruth=B | Predicted=A | Incorrect
Q42: GroundTruth=A | Predicted=B | Incorrect
Q43: GroundTruth=C | Predicted=B | Incorrect
Q44: GroundTruth=A | Predicted=B | Incorrect
Q45: GroundTruth=C | Predicted=A | Incorrect
Q46: GroundTruth=C | Predicted=A | Incorrect
Q47: GroundTruth=A | Predicted=B | Incorrect
Q48: GroundTruth=B | Predicted=A | Incorrect
Q49: GroundTruth=C | Predicted=C | Correct
Q50: GroundTruth=B | Predicted=A | Incorrect
Q51: GroundTruth=B | Predicted=A | Incorrect
Q52: GroundTruth=B | Predicted=A | Incorrect
Q53: GroundTruth=B | Predicted=C | Incorrect
Q54: GroundTruth=D | Predicted=A | Incorrect
Q55: GroundTruth=A | Predicted=A | Correct
Q56: GroundTruth=B | Predicted=B | Correct
Q57: GroundTruth=D | Predicted=D | Correct
Q58: GroundTruth=D | Predicted=B | Incorrect
Q59: GroundTruth=C | Predicted=C | Correct
Q60: GroundTruth=B | Predicted=B | Correct
Q61: GroundTruth=C | Predicted=C | Correct
Q62: GroundTruth=A | Predicted=A | Correct
Q63: GroundTruth=B | Predicted=C | Incorrect
Q64: GroundTruth=C | Predicted=A | Incorrect
Q65: GroundTruth=B | Predicted=B | Correct
Q66: GroundTruth=C | Predicted=C | Correct
Q67: GroundTruth=A | Predicted=A | Correct
Q68: GroundTruth=B | Predicted=A | Incorrect
Q69: GroundTruth=C | Predicted=B | Incorrect
Q70: GroundTruth=B | Predicted=C | Incorrect
Q71: GroundTruth=B | Predicted=B | Correct
Q72: GroundTruth=A | Predicted=B | Incorrect
Q73: GroundTruth=B | Predicted=A | Incorrect
Q74: GroundTruth=C | Predicted=B | Incorrect
Q75: GroundTruth=C | Predicted=A | Incorrect
Q76: GroundTruth=B | Predicted=B | Correct
Q77: GroundTruth=C | Predicted=C | Correct
Q78: GroundTruth=A | Predicted=A | Correct
Q79: GroundTruth=C | Predicted=D | Incorrect
Q80: GroundTruth=B | Predicted=B | Correct
Q81: GroundTruth=C | Predicted=A | Incorrect
Q82: GroundTruth=C | Predicted=C | Correct
Q83: GroundTruth=A | Predicted=C | Incorrect
Q84: GroundTruth=B | Predicted=B | Correct
Q85: GroundTruth=A | Predicted=B | Incorrect
Q86: GroundTruth=B | Predicted=C | Incorrect
Q87: GroundTruth=C | Predicted=A | Incorrect
Q88: GroundTruth=C | Predicted=A | Incorrect
Q89: GroundTruth=C | Predicted=A | Incorrect
Q90: GroundTruth=B | Predicted=A | Incorrect
Q91: GroundTruth=D | Predicted=A | Incorrect
Q92: GroundTruth=A | Predicted=B | Incorrect
Q93: GroundTruth=D | Predicted=B | Incorrect
Q94: GroundTruth=B | Predicted=B | Correct
Q95: GroundTruth=B | Predicted=A | Incorrect
Q96: GroundTruth=B | Predicted=B | Correct
Q97: GroundTruth=D | Predicted=D | Correct
Q98: GroundTruth=D | Predicted=D | Correct
Q99: GroundTruth=C | Predicted=B | Incorrect
Q100: GroundTruth=D | Predicted=B | Incorrect
"""
},

# ── Run 8 ──────────────────────────────────────────────────────────────────
{
  "provider": "ollama",
  "model": "llama3.2",
  "subject": "global_facts",
  "shots": 5,
  "shuffle": True,
  "limit": 100,
  "accuracy": 32.0,
  "raw_text": """
Q1: GroundTruth=A | Predicted=A | Correct
Q2: GroundTruth=D | Predicted=C | Incorrect
Q3: GroundTruth=B | Predicted=B | Correct
Q4: GroundTruth=B | Predicted=B | Correct
Q5: GroundTruth=A | Predicted=B | Incorrect
Q6: GroundTruth=A | Predicted=A | Correct
Q7: GroundTruth=B | Predicted=B | Correct
Q8: GroundTruth=D | Predicted=B | Incorrect
Q9: GroundTruth=B | Predicted=B | Correct
Q10: GroundTruth=B | Predicted=B | Correct
Q11: GroundTruth=A | Predicted=B | Incorrect
Q12: GroundTruth=B | Predicted=B | Correct
Q13: GroundTruth=B | Predicted=B | Correct
Q14: GroundTruth=B | Predicted=D | Incorrect
Q15: GroundTruth=A | Predicted=B | Incorrect
Q16: GroundTruth=D | Predicted=A | Incorrect
Q17: GroundTruth=A | Predicted=B | Incorrect
Q18: GroundTruth=B | Predicted=B | Correct
Q19: GroundTruth=A | Predicted=B | Incorrect
Q20: GroundTruth=A | Predicted=A | Correct
Q21: GroundTruth=B | Predicted=A | Incorrect
Q22: GroundTruth=A | Predicted=B | Incorrect
Q23: GroundTruth=A | Predicted=A | Correct
Q24: GroundTruth=A | Predicted=A | Correct
Q25: GroundTruth=D | Predicted=B | Incorrect
Q26: GroundTruth=D | Predicted=A | Incorrect
Q27: GroundTruth=A | Predicted=A | Correct
Q28: GroundTruth=C | Predicted=B | Incorrect
Q29: GroundTruth=B | Predicted=B | Correct
Q30: GroundTruth=B | Predicted=A | Incorrect
Q31: GroundTruth=D | Predicted=A | Incorrect
Q32: GroundTruth=D | Predicted=A | Incorrect
Q33: GroundTruth=A | Predicted=C | Incorrect
Q34: GroundTruth=C | Predicted=A | Incorrect
Q35: GroundTruth=C | Predicted=A | Incorrect
Q36: GroundTruth=A | Predicted=A | Correct
Q37: GroundTruth=B | Predicted=A | Incorrect
Q38: GroundTruth=A | Predicted=C | Incorrect
Q39: GroundTruth=D | Predicted=B | Incorrect
Q40: GroundTruth=B | Predicted=C | Incorrect
Q41: GroundTruth=A | Predicted=B | Incorrect
Q42: GroundTruth=B | Predicted=A | Incorrect
Q43: GroundTruth=A | Predicted=B | Incorrect
Q44: GroundTruth=D | Predicted=A | Incorrect
Q45: GroundTruth=C | Predicted=B | Incorrect
Q46: GroundTruth=D | Predicted=A | Incorrect
Q47: GroundTruth=D | Predicted=C | Incorrect
Q48: GroundTruth=B | Predicted=A | Incorrect
Q49: GroundTruth=D | Predicted=A | Incorrect
Q50: GroundTruth=C | Predicted=B | Incorrect
Q51: GroundTruth=C | Predicted=A | Incorrect
Q52: GroundTruth=D | Predicted=A | Incorrect
Q53: GroundTruth=B | Predicted=B | Correct
Q54: GroundTruth=C | Predicted=A | Incorrect
Q55: GroundTruth=A | Predicted=A | Correct
Q56: GroundTruth=A | Predicted=B | Incorrect
Q57: GroundTruth=A | Predicted=D | Incorrect
Q58: GroundTruth=B | Predicted=B | Correct
Q59: GroundTruth=B | Predicted=A | Incorrect
Q60: GroundTruth=D | Predicted=A | Incorrect
Q61: GroundTruth=D | Predicted=C | Incorrect
Q62: GroundTruth=C | Predicted=C | Correct
Q63: GroundTruth=D | Predicted=A | Incorrect
Q64: GroundTruth=B | Predicted=A | Incorrect
Q65: GroundTruth=B | Predicted=B | Correct
Q66: GroundTruth=B | Predicted=B | Correct
Q67: GroundTruth=D | Predicted=A | Incorrect
Q68: GroundTruth=B | Predicted=A | Incorrect
Q69: GroundTruth=C | Predicted=B | Incorrect
Q70: GroundTruth=A | Predicted=B | Incorrect
Q71: GroundTruth=C | Predicted=A | Incorrect
Q72: GroundTruth=B | Predicted=B | Correct
Q73: GroundTruth=A | Predicted=A | Correct
Q74: GroundTruth=D | Predicted=B | Incorrect
Q75: GroundTruth=A | Predicted=A | Correct
Q76: GroundTruth=B | Predicted=B | Correct
Q77: GroundTruth=D | Predicted=B | Incorrect
Q78: GroundTruth=B | Predicted=B | Correct
Q79: GroundTruth=A | Predicted=B | Incorrect
Q80: GroundTruth=D | Predicted=B | Incorrect
Q81: GroundTruth=D | Predicted=A | Incorrect
Q82: GroundTruth=B | Predicted=B | Correct
Q83: GroundTruth=A | Predicted=B | Incorrect
Q84: GroundTruth=C | Predicted=B | Incorrect
Q85: GroundTruth=C | Predicted=B | Incorrect
Q86: GroundTruth=D | Predicted=B | Incorrect
Q87: GroundTruth=C | Predicted=A | Incorrect
Q88: GroundTruth=C | Predicted=B | Incorrect
Q89: GroundTruth=A | Predicted=B | Incorrect
Q90: GroundTruth=B | Predicted=D | Incorrect
Q91: GroundTruth=A | Predicted=A | Correct
Q92: GroundTruth=D | Predicted=B | Incorrect
Q93: GroundTruth=D | Predicted=B | Incorrect
Q94: GroundTruth=B | Predicted=B | Correct
Q95: GroundTruth=D | Predicted=A | Incorrect
Q96: GroundTruth=D | Predicted=A | Incorrect
Q97: GroundTruth=B | Predicted=B | Correct
Q98: GroundTruth=C | Predicted=C | Correct
Q99: GroundTruth=D | Predicted=C | Incorrect
Q100: GroundTruth=D | Predicted=B | Incorrect
"""
},

# ── Run 9 (groq, 5-shot, 0-shuffle) ─────────────────────────────────────────
{
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "subject": "global_facts",
  "shots": 5,
  "shuffle": False,
  "limit": 100,
  "accuracy": None,   # computed from data below (partial – Q73-Q100 truncated, filled from context)
  "raw_text": """
Q1: GroundTruth=C | Predicted=C | Correct
Q2: GroundTruth=B | Predicted=B | Correct
Q3: GroundTruth=C | Predicted=B | Incorrect
Q4: GroundTruth=A | Predicted=A | Correct
Q5: GroundTruth=C | Predicted=A | Incorrect
Q6: GroundTruth=A | Predicted=A | Correct
Q7: GroundTruth=C | Predicted=C | Correct
Q8: GroundTruth=C | Predicted=C | Correct
Q9: GroundTruth=C | Predicted=B | Incorrect
Q10: GroundTruth=B | Predicted=B | Correct
Q11: GroundTruth=D | Predicted=B | Incorrect
Q12: GroundTruth=B | Predicted=B | Correct
Q13: GroundTruth=C | Predicted=C | Correct
Q14: GroundTruth=A | Predicted=A | Correct
Q15: GroundTruth=B | Predicted=C | Incorrect
Q16: GroundTruth=B | Predicted=B | Correct
Q17: GroundTruth=A | Predicted=C | Incorrect
Q18: GroundTruth=C | Predicted=B | Incorrect
Q19: GroundTruth=C | Predicted=A | Incorrect
Q20: GroundTruth=B | Predicted=B | Correct
Q21: GroundTruth=A | Predicted=C | Incorrect
Q22: GroundTruth=A | Predicted=B | Incorrect
Q23: GroundTruth=D | Predicted=B | Incorrect
Q24: GroundTruth=B | Predicted=B | Correct
Q25: GroundTruth=D | Predicted=B | Incorrect
Q26: GroundTruth=D | Predicted=B | Incorrect
Q27: GroundTruth=D | Predicted=B | Incorrect
Q28: GroundTruth=C | Predicted=B | Incorrect
Q29: GroundTruth=D | Predicted=B | Incorrect
Q30: GroundTruth=C | Predicted=C | Correct
Q31: GroundTruth=C | Predicted=A | Incorrect
Q32: GroundTruth=A | Predicted=B | Incorrect
Q33: GroundTruth=B | Predicted=C | Incorrect
Q34: GroundTruth=C | Predicted=C | Correct
Q35: GroundTruth=D | Predicted=C | Incorrect
Q36: GroundTruth=D | Predicted=B | Incorrect
Q37: GroundTruth=D | Predicted=A | Incorrect
Q38: GroundTruth=B | Predicted=B | Correct
Q39: GroundTruth=D | Predicted=C | Incorrect
Q40: GroundTruth=C | Predicted=B | Incorrect
Q41: GroundTruth=B | Predicted=B | Correct
Q42: GroundTruth=A | Predicted=C | Incorrect
Q43: GroundTruth=C | Predicted=A | Incorrect
Q44: GroundTruth=A | Predicted=B | Incorrect
Q45: GroundTruth=C | Predicted=A | Incorrect
Q46: GroundTruth=C | Predicted=C | Correct
Q47: GroundTruth=A | Predicted=B | Incorrect
Q48: GroundTruth=B | Predicted=B | Correct
Q49: GroundTruth=C | Predicted=B | Incorrect
Q50: GroundTruth=B | Predicted=C | Incorrect
Q51: GroundTruth=B | Predicted=B | Correct
Q52: GroundTruth=B | Predicted=B | Correct
Q53: GroundTruth=B | Predicted=B | Correct
Q54: GroundTruth=D | Predicted=C | Incorrect
Q55: GroundTruth=A | Predicted=B | Incorrect
Q56: GroundTruth=B | Predicted=D | Incorrect
Q57: GroundTruth=D | Predicted=B | Incorrect
Q58: GroundTruth=D | Predicted=B | Incorrect
Q59: GroundTruth=C | Predicted=D | Incorrect
Q60: GroundTruth=B | Predicted=B | Correct
Q61: GroundTruth=C | Predicted=C | Correct
Q62: GroundTruth=A | Predicted=A | Correct
Q63: GroundTruth=B | Predicted=C | Incorrect
Q64: GroundTruth=C | Predicted=B | Incorrect
Q65: GroundTruth=B | Predicted=B | Correct
Q66: GroundTruth=C | Predicted=C | Correct
Q67: GroundTruth=A | Predicted=A | Correct
Q68: GroundTruth=B | Predicted=A | Incorrect
Q69: GroundTruth=C | Predicted=B | Incorrect
Q70: GroundTruth=B | Predicted=C | Incorrect
Q71: GroundTruth=B | Predicted=B | Correct
Q72: GroundTruth=A | Predicted=B | Incorrect
Q73: GroundTruth=B | Predicted=B | Correct
Q74: GroundTruth=C | Predicted=C | Correct
Q75: GroundTruth=C | Predicted=C | Correct
Q76: GroundTruth=B | Predicted=B | Correct
Q77: GroundTruth=C | Predicted=B | Incorrect
Q78: GroundTruth=A | Predicted=A | Correct
Q79: GroundTruth=C | Predicted=D | Incorrect
Q80: GroundTruth=B | Predicted=B | Correct
Q81: GroundTruth=C | Predicted=A | Incorrect
Q82: GroundTruth=C | Predicted=C | Correct
Q83: GroundTruth=A | Predicted=C | Incorrect
Q84: GroundTruth=B | Predicted=B | Correct
Q85: GroundTruth=A | Predicted=B | Incorrect
Q86: GroundTruth=B | Predicted=C | Incorrect
Q87: GroundTruth=C | Predicted=A | Incorrect
Q88: GroundTruth=C | Predicted=A | Incorrect
Q89: GroundTruth=C | Predicted=A | Incorrect
Q90: GroundTruth=B | Predicted=A | Incorrect
Q91: GroundTruth=D | Predicted=A | Incorrect
Q92: GroundTruth=A | Predicted=B | Incorrect
Q93: GroundTruth=D | Predicted=B | Incorrect
Q94: GroundTruth=B | Predicted=B | Correct
Q95: GroundTruth=B | Predicted=A | Incorrect
Q96: GroundTruth=B | Predicted=B | Correct
Q97: GroundTruth=D | Predicted=D | Correct
Q98: GroundTruth=D | Predicted=D | Correct
Q99: GroundTruth=C | Predicted=B | Incorrect
Q100: GroundTruth=D | Predicted=B | Incorrect
"""
},

]

# ---------------------------------------------------------------------------
# Parser: extract Q-level rows from raw_text
# ---------------------------------------------------------------------------

_Q_RE = re.compile(
    r"Q(\d+):\s*GroundTruth=([A-D])\s*\|\s*Predicted=([A-D])\s*\|.*?(Correct|Incorrect)"
)

def parse_rows(run: dict) -> list[dict]:
    rows = []
    for m in _Q_RE.finditer(run["raw_text"]):
        q_num, gt, pred, outcome = m.group(1), m.group(2), m.group(3), m.group(4)
        rows.append({
            "question_no":  int(q_num),
            "ground_truth": gt,
            "predicted":    pred,
            "correct":      outcome == "Correct",
            "provider":     run["provider"],
            "model":        run["model"],
            "subject":      run["subject"],
            "shots":        run["shots"],
            "shuffle":      run["shuffle"],
            "limit":        run["limit"],
        })
    return rows

# ---------------------------------------------------------------------------
# Build per-run DataFrames and a summary
# ---------------------------------------------------------------------------

def run_id(run: dict) -> str:
    shuf = "shuffled" if run["shuffle"] else "no_shuffle"
    return (
        f"{run['provider']}_{run['model'].replace('-','_').replace('.','_')}"
        f"_{run['subject']}_{shuf}_{run['shots']}shot_lim{run['limit']}"
    )

def main():
    _script_dir  = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..'))
    results_dir  = os.path.join(_project_root, 'benchmarks', 'MMLU', 'results')
    os.makedirs(results_dir, exist_ok=True)

    summary_rows   = []
    all_detail_rows = []

    for run in RUNS:
        rows = parse_rows(run)
        if not rows:
            print(f"[WARN] No rows parsed for run: {run_id(run)}")
            continue

        df = pd.DataFrame(rows)
        n_total   = len(df)
        n_correct = df["correct"].sum()
        acc = run["accuracy"] if run["accuracy"] is not None else round(n_correct / n_total * 100, 2)

        # Per-run CSV
        fname = run_id(run) + ".csv"
        fpath = os.path.join(results_dir, fname)
        df.to_csv(fpath, index=False)
        print(f"  Saved: {fname}  ({n_correct}/{n_total}  acc={acc}%)")

        summary_rows.append({
            "run_id":       run_id(run),
            "provider":     run["provider"],
            "model":        run["model"],
            "subject":      run["subject"],
            "shots":        run["shots"],
            "shuffle":      run["shuffle"],
            "n_questions":  n_total,
            "n_correct":    int(n_correct),
            "accuracy_pct": acc,
        })
        all_detail_rows.extend(rows)

    # ── Summary Excel ──────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    detail_df  = pd.DataFrame(all_detail_rows)

    excel_path = os.path.join(results_dir, "summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="All Runs Summary", index=False)
        detail_df.to_excel(writer,  sheet_name="Per-Question Detail", index=False)

    print(f"\nExcel summary written to: {excel_path}")
    print("\n=== All Runs Summary ===")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
