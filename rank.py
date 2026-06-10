"""
rank.py  ─  General-Purpose AI Candidate Ranking System
=========================================================
Works for ANY job description. Zero hardcoded role-specific skills.

Usage:
  python rank.py --jd job_description.txt --candidates candidates.jsonl --out submission.csv

No external dependencies. Python stdlib only.
"""

import json, csv, heapq, argparse, re
from datetime import datetime
from collections import Counter

REFERENCE_DATE = datetime(2026, 1, 15)

# ── Universal constants (truly not role-specific) ──────────────────────────────

KNOWN_CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "mindtree", "birlasoft", "l&t infotech", "deloitte", "kpmg",
    "pwc", "booz allen", "mckinsey", "bcg",
}

TIER1_SCHOOLS = {
    "iit ", "iim ", "bits ", "iisc", "nit ", "stanford", "mit ",
    "cmu", "carnegie mellon", "berkeley", "oxford", "cambridge",
}

SENIOR_WORDS = {"senior", "staff", "principal", "lead", "head", "chief"}
JUNIOR_WORDS = {"junior", "associate", "intern", "trainee", "fresher"}

# Extended stop words — includes words that sound technical but are prose
STOP_WORDS = {
    "the","and","or","in","of","to","a","an","for","with","is","are","we",
    "you","your","our","have","has","will","be","this","that","it","at","on",
    "by","as","if","can","not","but","from","all","what","how","who","their",
    "they","both","into","than","been","were","would","its","do","did","done",
    "use","using","used","build","building","built","need","needs","needs",
    "experience","years","role","team","work","company","candidate","candidates",
    "please","most","one","two","three","product","things","people","actually",
    "means","mean","just","about","some","also","without","should","get","got",
    "find","want","make","know","think","think","like","take","see","say","here",
    "there","which","more","these","those","them","has","had","have","etc",
    "systems","system","way","ways","much","many","every","each","part","type",
    "because","before","after","while","when","where","how","why","who","whom",
    "very","really","well","even","still","first","second","third","last","next",
    "new","good","great","best","right","wrong","true","false","real","clear",
    "strong","deep","solid","broad","full","open","high","low","large","small",
    "early","late","long","short","old","young","free","fast","slow","hard","easy",
    "different","same","similar","likely","possible","available","current","previous",
    "following","including","example","examples","case","cases","whether","might",
}


# ── JD Parser ─────────────────────────────────────────────────────────────────

def norm(s):
    return s.strip().lower()

def extract_skill_terms(text: str) -> set:
    """
    Extract only genuine skill-like terms from a block of text.
    Targets: named tools (PascalCase/ALLCAPS), hyphenated tech terms,
    known technical bigrams, metric names, framework names.
    Avoids: prose, verbs, adjectives, generic nouns.
    """
    terms = set()

    # 1. Named tools: words with internal caps or all-caps (FAISS, PyTorch, LlamaIndex)
    named = re.findall(r"\b([A-Z][a-zA-Z0-9]{2,}|[A-Z]{3,})\b", text)
    for t in named:
        tl = t.lower()
        # filter out obvious prose words
        if tl not in STOP_WORDS and len(tl) >= 3:
            terms.add(tl)

    # 2. Hyphenated tech terms (sentence-transformers, learning-to-rank, fine-tuning)
    hyphenated = re.findall(r"\b[a-z][a-z0-9]*(?:-[a-z][a-z0-9]+){1,3}\b", text.lower())
    terms.update(h for h in hyphenated if len(h) > 5 and h not in STOP_WORDS)

    # 3. Technical bigrams: two non-stop technical words together
    # (learning to rank, vector database, hybrid search, embedding drift)
    words = re.findall(r"[a-z][a-z0-9]+", text.lower())
    for i in range(len(words)-1):
        a, b = words[i], words[i+1]
        if (a not in STOP_WORDS and b not in STOP_WORDS
                and len(a) >= 3 and len(b) >= 3
                and not a.isdigit() and not b.isdigit()):
            terms.add(f"{a} {b}")

    # 4. Trigrams for common 3-word skill phrases
    for i in range(len(words)-2):
        a, b, c = words[i], words[i+1], words[i+2]
        if (a not in STOP_WORDS and c not in STOP_WORDS
                and len(a) >= 3 and len(c) >= 3):
            terms.add(f"{a} {b} {c}")

    # Remove obvious non-technical terms
    noise = {"this", "that", "they", "their", "have", "been", "were", "will",
             "would", "could", "should", "which", "what", "when", "where",
             "then", "than", "there", "here", "with", "from", "your", "ours",
             "very", "much", "also", "even", "just", "only", "some", "such"}
    terms = {t for t in terms if t not in noise and len(t) >= 3}

    return terms


def parse_jd(jd_text: str) -> dict:
    """
    Parse any job description into structured requirements.
    No role-specific hardcoding anywhere.

    Matching strategy:
      candidate_skill_name → JD text  (clean, no noise)
    NOT:
      JD_words → candidate_text  (too noisy — prose leaks in)
    """
    text_low = norm(jd_text)

    # ── 1. Experience range ───────────────────────────────────────────────────
    exp_min, exp_max = 3, 15
    m = re.search(r"(\d+)\s*[-–to]+\s*(\d+)\s*years?", text_low)
    if m:
        exp_min, exp_max = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"(\d+)\+\s*years?", text_low)
        if m:
            exp_min = int(m.group(1))
            exp_max = exp_min + 6

    # ── 2. Domain detection ───────────────────────────────────────────────────
    domain_signals = {
        "nlp_ir":   ["retrieval", "ranking", "recommendation", "embedding", "vector",
                     "semantic search", "llm", "language model", "nlp", "rag",
                     "information retrieval", "search"],
        "cv":       ["computer vision", "image classification", "object detection",
                     "segmentation", "yolo", "cnn", "video understanding"],
        "speech":   ["speech recognition", "asr", "text-to-speech", "tts",
                     "audio processing"],
        "mlops":    ["mlops", "model serving", "monitoring", "kubernetes",
                     "docker", "airflow", "deployment pipeline"],
        "data_eng": ["data pipeline", "etl", "spark", "hadoop",
                     "data warehouse", "dbt", "bigquery", "snowflake"],
        "fintech":  ["fraud detection", "credit scoring", "trading",
                     "financial", "banking"],
    }
    domain_scores = {}
    for domain, sigs in domain_signals.items():
        domain_scores[domain] = sum(1 for s in sigs if s in text_low)
    primary_domain = max(domain_scores, key=domain_scores.get)

    # ── 3. Notice period preference ───────────────────────────────────────────
    notice_pref = 60
    m = re.search(r"sub[-\s]?(\d+)[-\s]?day", text_low)
    if m:
        notice_pref = int(m.group(1))
    else:
        m = re.search(r"(\d+)\s*[-\s]?day\s*notice", text_low)
        if m:
            notice_pref = int(m.group(1))

    # ── 4. GitHub / open-source valued? ──────────────────────────────────────
    values_oss = any(kw in text_low for kw in
                     ["github", "open source", "open-source", "contributions",
                      "papers", "publications", "talks"])

    # ── 5. Consulting firms mentioned ─────────────────────────────────────────
    mentioned_firms = {f for f in KNOWN_CONSULTING_FIRMS if f in text_low}

    # ── 6. Parse requirement sections ─────────────────────────────────────────
    req_markers  = ["things you absolutely", "absolutely need", "you must have",
                    "must have", "hard requirement", "required skills"]
    pref_markers = ["things we'd like", "things we would", "nice to have",
                    "like you to have", "would like you", "bonus"]
    disq_markers = ["things we explicitly", "do not want", "won't move forward",
                    "will not move forward", "disqualifier"]

    lines = jd_text.splitlines()
    current = "general"
    section_lines = {"required": [], "preferred": [], "disqualify": [], "general": []}
    for line in lines:
        ll = norm(line)
        if any(mk in ll for mk in req_markers):   current = "required"
        elif any(mk in ll for mk in pref_markers): current = "preferred"
        elif any(mk in ll for mk in disq_markers): current = "disqualify"
        else:
            section_lines[current].append(line)  # keep original case for named tools

    required_text  = " ".join(section_lines["required"])
    preferred_text = " ".join(section_lines["preferred"])
    disq_text      = " ".join(section_lines["disqualify"])
    full_jd_text   = " ".join(section_lines["required"] +
                               section_lines["preferred"] +
                               section_lines["general"])

    required_terms  = extract_skill_terms(required_text)
    preferred_terms = extract_skill_terms(preferred_text)
    disq_terms      = extract_skill_terms(disq_text)
    jd_vocab        = extract_skill_terms(full_jd_text)  # all sections except disqualify

    return {
        "exp_min":          exp_min,
        "exp_max":          exp_max,
        "primary_domain":   primary_domain,
        "domain_scores":    domain_scores,
        "notice_pref_days": notice_pref,
        "values_oss":       values_oss,
        "mentioned_firms":  mentioned_firms,
        "jd_vocab":         jd_vocab,
        "required_terms":   required_terms,
        "preferred_terms":  preferred_terms,
        "disq_terms":       disq_terms,
        "full_text":        text_low,
    }


# ── Candidate utilities ───────────────────────────────────────────────────────

def build_skill_map(skills):
    m = {}
    for s in skills:
        n = norm(s.get("name", ""))
        m[n] = (s.get("proficiency","beginner"),
                max(0, s.get("duration_months", 0)),
                max(0, s.get("endorsements", 0)))
    return m

def prof_val(p):
    return {"beginner":0.25,"intermediate":0.50,"advanced":0.75,"expert":1.0}.get(p,0.25)

def career_text(career):
    return norm(" ".join(j.get("title","")+" "+j.get("description","") for j in career))

def candidate_full_text(candidate):
    """All text about a candidate — for JD relevance check."""
    parts = []
    p = candidate.get("profile", {})
    parts += [p.get("current_title",""), p.get("summary",""), p.get("headline","")]
    for j in candidate.get("career_history", []):
        parts += [j.get("title",""), j.get("description","")]
    for s in candidate.get("skills", []):
        parts.append(s.get("name",""))
    return norm(" ".join(parts))

def all_titles(profile, career):
    return [norm(profile.get("current_title",""))] + [norm(j.get("title","")) for j in career]

def days_since(date_str):
    try:
        return max(0, (REFERENCE_DATE - datetime.strptime(date_str, "%Y-%m-%d")).days)
    except:
        return 9999


# ── Hard Gates ────────────────────────────────────────────────────────────────

def is_disqualified(candidate, jd):
    profile = candidate.get("profile", {})
    skills  = candidate.get("skills", [])
    career  = candidate.get("career_history", [])
    yoe     = profile.get("years_of_experience", 0)
    ct      = candidate_full_text(candidate)
    sm      = build_skill_map(skills)

    # Gate 1: Too junior (allow 70% of JD minimum)
    if yoe < jd["exp_min"] * 0.70:
        return True, f"Too junior: {yoe}yr, JD needs {jd['exp_min']}+"

    # Gate 2: Honeypot — expert + 0 months (>=3 cases = fabricated)
    zero_expert = [s.get("name","") for s in skills
                   if s.get("proficiency")=="expert"
                   and s.get("duration_months",1)==0]
    if len(zero_expert) >= 3:
        return True, f"Honeypot: {len(zero_expert)} expert skills with 0 months"

    # Gate 3: Implausible timeline
    total_mo = sum(j.get("duration_months",0) for j in career)
    if total_mo > 0 and yoe*12 > total_mo*2.0+36:
        return True, f"Implausible: claims {yoe:.1f}yr, history={total_mo/12:.1f}yr"

    # Gate 4: Zero relevance to JD domain
    # Check overlap between candidate's actual SKILL NAMES and JD vocabulary
    cand_skills_norm = {norm(s.get("name","")) for s in skills}
    jd_vocab = jd["jd_vocab"]
    skill_overlap = sum(
        1 for skill in cand_skills_norm
        if skill in jd_vocab or
        any(skill in jdterm or jdterm in skill
            for jdterm in jd_vocab if len(jdterm) > 4)
    )
    # Also check career text overlap with JD domain signals
    domain = jd["primary_domain"]
    domain_signals = {
        "nlp_ir":   ["retrieval","ranking","embedding","vector","semantic","nlp","search","recommend"],
        "cv":       ["vision","image","detection","segmentation","yolo","cnn"],
        "speech":   ["speech","audio","asr","tts","speaker"],
        "mlops":    ["mlops","pipeline","deployment","kubernetes","airflow"],
        "data_eng": ["pipeline","spark","etl","warehouse","dbt"],
        "fintech":  ["fraud","credit","risk","trading","financial"],
    }
    domain_kws = domain_signals.get(domain, [])
    domain_hit = sum(1 for kw in domain_kws if kw in ct)

    if skill_overlap == 0 and domain_hit == 0:
        return True, f"No relevance to JD domain ({domain})"

    return False, ""


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_title(profile, career, jd):
    """0-20 pts. Dynamic title matching using JD domain keywords."""
    titles = all_titles(profile, career)

    domain_title_kws = {
        "nlp_ir":   ["engineer", "scientist", "researcher", "specialist",
                     "nlp", "ml", "ai", "search", "retrieval", "recommendation",
                     "ranking", "deep learning", "applied", "data scientist"],
        "cv":       ["engineer", "scientist", "vision", "image", "perception"],
        "speech":   ["engineer", "scientist", "speech", "audio", "acoustics"],
        "mlops":    ["engineer", "platform", "infrastructure", "devops", "sre"],
        "data_eng": ["engineer", "analyst", "pipeline", "architect"],
        "fintech":  ["engineer", "analyst", "quant", "risk", "data"],
    }
    relevant_kws = domain_title_kws.get(jd["primary_domain"],
                                         ["engineer", "scientist"])

    for t in titles:
        domain_hits = sum(1 for kw in relevant_kws if kw in t)
        if domain_hits >= 2:
            seniority = 3 if any(sw in t for sw in SENIOR_WORDS) else 0
            junior_pen = -4 if any(jw in t for jw in JUNIOR_WORDS) else 0
            return min(20, 15 + seniority + junior_pen)
        elif domain_hits == 1:
            return 10
    return 3


def score_experience(yoe, jd):
    """0-20 pts. Dynamic based on JD's experience range."""
    e_min, e_max = jd["exp_min"], jd["exp_max"]
    # Sweet spot: middle 60% of the range
    lo = e_min + (e_max - e_min) * 0.2
    hi = e_max - (e_max - e_min) * 0.2

    if lo <= yoe <= hi:               return 20
    elif e_min <= yoe < lo:           return 16
    elif hi < yoe <= e_max:           return 15
    elif e_min*0.8 <= yoe < e_min:    return 10
    elif e_max < yoe <= e_max*1.3:    return 10
    elif yoe < e_min*0.8:             return 4
    else:                             return 6

def score_skills(skills, career, signals, jd):
    """
    0-40 pts. The core scoring function.

    KEY DESIGN: We check each CANDIDATE SKILL against the JD text.
    Direction: candidate_skill → jd_text  (NOT jd_words → candidate_text)
    This prevents noisy JD prose words from becoming false skill matches.

    Scoring tiers:
      Required section match   → 3.0 pts/skill (max 18)
      Preferred section match  → 1.5 pts/skill (max 8)
      General JD vocab match   → 0.8 pts/skill (max 7)
      Proficiency × duration × endorsements modulate within each tier
      Platform assessment scores → bonus multiplier
    """
    sm   = build_skill_map(skills)
    ct   = career_text(career)
    jd_text = jd["full_text"]
    required  = jd["required_terms"]
    preferred = jd["preferred_terms"]
    jd_vocab  = jd["jd_vocab"]

    # Platform-validated skill scores
    assessments = signals.get("skill_assessment_scores", {})
    validated = {norm(k): min(v/100.0, 1.0) for k, v in assessments.items() if v > 0}

    found = []
    score = 0.0
    req_score  = 0.0
    pref_score = 0.0
    gen_score  = 0.0

    # Score each candidate skill name
    for skill_name, (prof, dur, end) in sm.items():
        if len(skill_name) < 3:
            continue

        pv       = prof_val(prof)
        dur_pts  = min(dur / 12.0, 3.0)   # up to 3 bonus for duration
        end_pts  = min(end / 100.0, 0.3)  # up to 0.3 bonus for endorsements

        # Check assessment boost
        assess_mult = 1.0
        for ak, av in validated.items():
            if skill_name in ak or ak in skill_name:
                assess_mult = 1.0 + av * 0.25
                break

        # Depth score for this skill (0-1)
        depth = (pv * 0.6 + dur_pts * 0.25 + end_pts * 0.15) * assess_mult

        # Where does this skill appear in the JD?
        in_required  = (skill_name in required or
                        any(skill_name in t or t in skill_name
                            for t in required if len(t) > 4))
        in_preferred = (skill_name in preferred or
                        any(skill_name in t or t in skill_name
                            for t in preferred if len(t) > 4))
        in_jd_text   = (skill_name in jd_text or
                        skill_name in jd_vocab or
                        any(skill_name in v or v in skill_name
                            for v in jd_vocab if len(v) > 4))

        if in_required:
            pts = depth * 3.0
            req_score += pts
            found.append(f"{skill_name}({prof},{dur}mo) [REQ]")
        elif in_preferred:
            pts = depth * 1.5
            pref_score += pts
            found.append(f"{skill_name}({prof},{dur}mo) [PREF]")
        elif in_jd_text:
            pts = depth * 0.8
            gen_score += pts
            found.append(f"{skill_name}({prof},{dur}mo)")

    # Also check career text for evidence of required skills (partial credit)
    for req_term in required:
        if len(req_term) < 5: continue
        if req_term in ct and req_term not in str(found):
            req_score += 0.3
            found.append(f"{req_term}(career)")

    score += min(18, req_score)
    score += min(8,  pref_score)
    score += min(7,  gen_score)

    # Platform assessment bonus (0-4) — external validation is trusted
    if validated:
        score += min(4, sum(validated.values()) / len(validated) * 4)

    return min(40, round(score, 3)), found

def score_career(career, jd):
    """0-15 pts. Dynamic using JD's consulting firms list."""
    if not career: return 0

    total_mo = sum(j.get("duration_months",0) for j in career)
    bad_firms = jd["mentioned_firms"] or KNOWN_CONSULTING_FIRMS
    consulting_mo = sum(
        j.get("duration_months",0) for j in career
        if any(f in norm(j.get("company","")) for f in bad_firms)
    )
    product_mo = max(0, total_mo - consulting_mo)
    score = 0.0

    if total_mo > 0:
        pr = product_mo / total_mo
        if pr >= 0.8:   score += 7
        elif pr >= 0.5: score += 4
        elif pr >= 0.2: score += 1
        else:           score -= 4

    short = sum(1 for j in career
                if j.get("duration_months",12) < 12
                and not j.get("is_current",False))
    if short >= 4:   score -= 4
    elif short >= 2: score -= 1
    elif short == 0: score += 2

    # Count how many past roles were relevant to JD domain
    domain_kws = {
        "nlp_ir":   ["machine learning","ml ","ai engineer","search","retrieval",
                     "ranking","recommendation","nlp","embedding","llm","data scientist"],
        "cv":       ["computer vision","image","object detection","perception","robotics"],
        "speech":   ["speech","audio","asr","tts","acoustic"],
        "mlops":    ["mlops","platform","infrastructure","devops","pipeline"],
        "data_eng": ["data engineer","etl","pipeline","spark","data warehouse"],
        "fintech":  ["finance","trading","risk","fraud","credit"],
    }.get(jd["primary_domain"], ["machine learning","engineer","data"])

    domain_roles = sum(
        1 for j in career
        if sum(1 for kw in domain_kws
               if kw in norm(j.get("title","")+" "+j.get("description",""))) >= 1
    )
    if domain_roles >= 3:   score += 6
    elif domain_roles == 2: score += 4
    elif domain_roles == 1: score += 2

    return max(0, min(15, score))

def score_education(education):
    """0-5 pts. Universal."""
    if not education: return 0
    best = 0
    for ed in education:
        inst  = norm(ed.get("institution",""))
        tier  = ed.get("tier","unknown")
        field = norm(ed.get("field_of_study",""))
        deg   = norm(ed.get("degree",""))
        t_val = {"tier_1":5,"tier_2":3,"tier_3":2,"tier_4":1,"unknown":1}.get(tier,1)
        t1b = 1 if any(t in inst for t in TIER1_SCHOOLS) else 0
        fb  = 1 if any(f in field for f in ["computer","machine learning","data",
                                              "artificial","statistics","mathematics",
                                              "information","electronics"]) else 0
        db  = 1 if any(d in deg for d in ["phd","ph.d","m.tech","m.e.","m.sc"]) else 0
        best = max(best, min(5, t_val+t1b+fb+db-2))
    return max(0, best)

def score_market(signals, jd):
    """0-5 pts. Universal market desirability signals."""
    score  = 0.0
    saved  = signals.get("saved_by_recruiters_30d", 0)
    search = signals.get("search_appearance_30d", 0)
    conns  = signals.get("connection_count", 0)
    endors = signals.get("endorsements_received", 0)

    score += min(2.0, saved / 8.0)
    score += min(1.5, search / 200.0)
    score += min(1.0, conns / 600.0)
    score += min(0.5, endors / 80.0)

    if jd["values_oss"]:
        gh = signals.get("github_activity_score", -1)
        if gh >= 60: score += 0.5

    return min(5, score)

def behavioral_multiplier(signals, jd):
    """
    0.25–1.15×. Universal availability multiplier.
    Applied as multiplier (not additive) to enforce that
    an unreachable candidate ranks below a slightly weaker but available one.
    """
    mult = 1.0

    d = days_since(signals.get("last_active_date","2020-01-01"))
    if   d <= 14:   mult += 0.08
    elif d <= 30:   mult += 0.05
    elif d <= 90:   mult += 0.00
    elif d <= 150:  mult -= 0.08
    elif d <= 270:  mult -= 0.20
    elif d <= 365:  mult -= 0.30
    else:           mult -= 0.45

    rr = signals.get("recruiter_response_rate", 0.5)
    if   rr >= 0.80: mult += 0.07
    elif rr >= 0.60: mult += 0.03
    elif rr >= 0.40: mult += 0.00
    elif rr >= 0.25: mult -= 0.07
    else:            mult -= 0.18

    if signals.get("open_to_work_flag", False):
        mult += 0.05

    gh = signals.get("github_activity_score", -1)
    gh_w = 1.5 if jd["values_oss"] else 1.0
    if   gh >= 80: mult += 0.12 * gh_w
    elif gh >= 60: mult += 0.08 * gh_w
    elif gh >= 40: mult += 0.04 * gh_w
    elif gh >= 15: mult += 0.01
    elif gh == -1: mult -= 0.05

    notice = signals.get("notice_period_days", 60)
    pref   = jd["notice_pref_days"]
    if   notice <= pref * 0.5: mult += 0.05
    elif notice <= pref:        mult += 0.02
    elif notice <= pref * 1.5:  mult -= 0.02
    else:                       mult -= 0.06

    ic = signals.get("interview_completion_rate", 0.5)
    if   ic >= 0.90: mult += 0.04
    elif ic >= 0.70: mult += 0.01
    elif ic < 0.30:  mult -= 0.06

    pc = signals.get("profile_completeness_score", 50)
    if   pc >= 90: mult += 0.03
    elif pc >= 75: mult += 0.01
    elif pc < 40:  mult -= 0.04

    oar = signals.get("offer_acceptance_rate", -1)
    if oar >= 0.8:       mult += 0.02
    elif 0 <= oar < 0.3: mult -= 0.03

    return max(0.25, min(1.15, mult))


# ── Master evaluator ──────────────────────────────────────────────────────────

def evaluate(candidate, jd):
    disq, reason = is_disqualified(candidate, jd)
    if disq:
        return -999.0, reason

    profile   = candidate.get("profile", {})
    skills    = candidate.get("skills", [])
    career    = candidate.get("career_history", [])
    education = candidate.get("education", [])
    signals   = candidate.get("redrob_signals", {})
    yoe       = profile.get("years_of_experience", 0)

    t_score        = score_title(profile, career, jd)
    e_score        = score_experience(yoe, jd)
    s_score, found = score_skills(skills, career, signals, jd)
    c_score        = score_career(career, jd)
    ed_score       = score_education(education)
    m_score        = score_market(signals, jd)
    b_mult         = behavioral_multiplier(signals, jd)

    raw   = t_score + e_score + s_score + c_score + ed_score + m_score
    final = round(raw * b_mult, 4)

    title  = norm(profile.get("current_title",""))
    rr     = signals.get("recruiter_response_rate", 0)
    gh     = signals.get("github_activity_score", -1)
    notice = signals.get("notice_period_days", 60)
    la     = days_since(signals.get("last_active_date","2020-01-01"))
    saved  = signals.get("saved_by_recruiters_30d", 0)

    key = ", ".join(dict.fromkeys(
        f.split(" [")[0] for f in found[:5]  # strip [REQ]/[PREF] tags
    ))
    parts = [
        f"{title.title()} | {yoe:.1f} yrs",
        f"skills: {key}" if key else "limited skill overlap",
        f"resp={rr:.0%}",
        f"active={la}d ago",
        f"GH={gh:.0f}" if gh >= 0 else "no GitHub",
        f"notice={notice}d",
    ]
    if saved >= 5:
        parts.append(f"saved-by-{saved}-recruiters")

    return final, " | ".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def process(jd_file, candidates_file, output_file):
    print(f"Parsing JD: {jd_file}")
    with open(jd_file, "r", encoding="utf-8") as f:
        jd_text = f.read()
    jd = parse_jd(jd_text)

    print(f"  Experience range  : {jd['exp_min']}-{jd['exp_max']} yrs")
    print(f"  Primary domain    : {jd['primary_domain']}")
    print(f"  Notice preference : {jd['notice_pref_days']}d")
    print(f"  Values OSS        : {jd['values_oss']}")
    print(f"  Consulting firms  : {sorted(jd['mentioned_firms'])}")
    print(f"  JD vocab size     : {len(jd['jd_vocab'])} terms")
    print(f"  Required terms    : {len(jd['required_terms'])}")
    print(f"  Preferred terms   : {len(jd['preferred_terms'])}")
    print()

    heap = []
    n = skipped = 0

    print(f"Scoring: {candidates_file}")
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            n += 1
            if n % 10_000 == 0:
                print(f"  {n:,} scanned | {skipped:,} disqualified | heap={len(heap)}")

            c = json.loads(line)
            score, reason = evaluate(c, jd)

            if score < 0:
                skipped += 1
                continue

            item = (score, c["candidate_id"], reason)
            if len(heap) < 100:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    print(f"\nDone: {n:,} total | {skipped:,} disqualified | {len(heap)} finalists")
    results = sorted(heap, reverse=True, key=lambda x: (x[0], x[1]))

    max_s = results[0][0]  if results else 1.0
    min_s = results[-1][0] if results else 0.0
    rng   = max(max_s - min_s, 1e-9)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (raw, cid, reason) in enumerate(results, 1):
            ns = round(0.20 + 0.79 * (raw - min_s) / rng, 4)
            writer.writerow([cid, rank, ns, reason])

    print("\nTop 10:")
    for rank, (raw, cid, reas) in enumerate(results[:10], 1):
        print(f"  #{rank:2d}  {cid}  score={raw:.2f}  {reas[:95]}")
    print(f"\nSaved -> {output_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="General-purpose candidate ranker — reads any JD, scores any candidates."
    )
    ap.add_argument("--jd",         required=True,          help="Path to job description .txt")
    ap.add_argument("--candidates", required=True,          help="Path to candidates.jsonl")
    ap.add_argument("--out",        default="submission.csv", help="Output CSV path")
    args = ap.parse_args()
    process(args.jd, args.candidates, args.out)
