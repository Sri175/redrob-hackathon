# Intelligent Candidate Discovery & Ranking — Redrob Hackathon

## Quickstart

```bash
python rank.py --jd job_description.txt --candidates candidates.jsonl --out submission.csv
```

**No external dependencies.** Python standard library only. Runs in ~15 seconds on 100K candidates.

---

## Design Philosophy

> "The right answer involves reasoning about the gap between what the JD says and what the JD means."
> — Challenge JD, Final Note

Most naive systems match keywords from the JD against candidate profiles. This fails in two ways:
1. A candidate who lists every AI keyword but is a Marketing Manager ranks high → wrong
2. A candidate who built a recommendation system at a product company but didn't use the exact word "RAG" → missed

Our system works differently: **it reads the JD first, understands what the role actually needs, then scores candidates against that understanding** — not against a keyword checklist.

---

## Architecture

### Step 1: JD Parser — `parse_jd(jd_text)`

The system reads **any job description** and dynamically extracts:

| Signal | How Extracted | Example (this JD) |
|---|---|---|
| Experience range | Regex on `\d+–\d+ years` | 5–9 years |
| Primary domain | Keyword frequency (6 domain dictionaries) | `nlp_ir` (12 hits) |
| Required skills | Named tools + tech bigrams from "absolutely need" section | Pinecone, NDCG, sentence-transformers |
| Preferred skills | Same from "we'd like" section | LoRA, learning-to-rank |
| Notice preference | Regex on "sub-30-day notice" | 30 days |
| Consulting penalty | Named firms appearing in JD text | TCS, Wipro, Accenture, etc. |
| GitHub valued? | Keyword detection: "open-source", "contributions" | Yes |

**Zero role-specific hardcoding.** Give it a Frontend JD or a Data Engineering JD — the system adapts.

### Step 2: Hard Disqualification Gates

Before scoring anything, candidates are eliminated if they fail:

1. **Too junior** — below 70% of JD's stated minimum experience
2. **Honeypot** — 3+ skills listed as `expert` with `0 months` duration (fabricated profile)
3. **Implausible timeline** — claimed years of experience > 2× the sum of career history
4. **Zero domain relevance** — candidate's skill names have no overlap with JD vocabulary and no domain keyword match in career text

This eliminates ~55% of the 100K candidate pool immediately.

### Step 3: Multi-Dimensional Scoring (0–107 raw)

| Component | Max Pts | Logic |
|---|---|---|
| **Title match** | 20 | Checks current title AND full career history against JD domain keywords |
| **Experience fit** | 20 | Peaks at the ideal range centre; both extremes penalized |
| **Skill match** | 40 | Candidate skills → JD text (not JD words → candidate text) |
| **Career trajectory** | 15 | Product vs consulting ratio + tenure stability + domain role count |
| **Education** | 5 | Tier-1 institution bonus (IIT, BITS, Stanford, etc.) |
| **Market desirability** | 5 | Saved by recruiters, search appearances, connections, endorsements |

**Skill matching detail:**
```
For each candidate skill name:
  ├── Appears in JD's REQUIRED section → 3.0 pts × proficiency × depth
  ├── Appears in JD's PREFERRED section → 1.5 pts × proficiency × depth
  └── Appears anywhere in JD text → 0.8 pts × proficiency × depth

Depth = proficiency_value × 0.6 + duration_years × 0.25 + endorsement_rate × 0.15
```

This direction (candidate → JD) prevents JD prose words from becoming false skill matches.

### Step 4: Behavioral Availability Multiplier (0.25–1.15×)

The raw score is **multiplied** — never added to — by an availability factor. This ensures a ghost candidate (perfect skills but inactive for a year, 5% response rate) ranks below a slightly less skilled but actively available candidate.

Signals:
- Last active date (most weighted — > 1 year = −45% multiplier)
- Recruiter response rate
- Open-to-work flag
- GitHub activity (weighted higher when JD values open-source)
- Notice period vs JD preference
- Interview completion rate
- Profile completeness
- Offer acceptance rate

### Step 5: Top-100 via Min-Heap

Single streaming pass through all candidates: O(N log K), K=100. No loading everything into memory.

---

## Honeypot Resistance

The challenge dataset includes deliberately traps. We catch:

| Trap | Detection |
|---|---|
| Keyword stuffer (Marketing Manager with AI skills) | Domain gate: non-technical title + no ML career evidence |
| Fabricated expertise | Honeypot gate: 3+ `expert` skills with `0 months` |
| Inflated tenure | Timeline gate: claimed years > 2× career history |
| Ghost candidate | Behavioral multiplier: inactive + unresponsive → score crushed |
| Pure CV/Speech expert for NLP role | Domain mismatch gate |

---

## Compute Profile

| Metric | Result |
|---|---|
| Total runtime | ~12–18 seconds |
| Peak memory | < 200 MB |
| GPU | Not required |
| Network calls | Zero |
| Python version | 3.8+ |
| Dependencies | None (stdlib only) |

---

## Repository Structure

```
├── rank.py                  # Main ranking script (works for any JD)
├── jd.txt                   # Job description used for this submission
├── submission.csv           # Our submitted top-100 ranking
├── requirements.txt         # Empty — stdlib only
└── README.md                # This file
```

---

## Why This Beats Keyword Matching

| Scenario | Keyword system | Our system |
|---|---|---|
| Marketing Manager with "Pinecone" in skills | ✅ Ranks high (wrong!) | ❌ Gate 1: disqualified |
| ML Engineer who wrote "recommendation engine" in job description, not "RAG" in skills | ❌ Missed | ✅ Career text search finds it |
| Candidate inactive for 14 months, 5% response rate | ✅ Still top-10 (wrong!) | ❌ Multiplier: score × 0.35 |
| Junior with 1.5 years trying to fake seniority | ✅ Might pass | ❌ Gate 1: disqualified |
| Consultant with all AI skills but 0 product experience | ✅ Ranks high | ❌ Career score: −4 pts |
