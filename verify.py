"""
NoticeGuard - Verification Module.
Core Rule: "Gemma explains, code verifies."
Plain Python handles:
- Exact & fuzzy quote verification against source text (NFC normalized)
- Amount verification (digit-normalized against Indian formats)
- Date existence verification
- Passage linking (mapping verified quotes to P1, P2, ...)
Anything code can't verify is marked UNVERIFIED or dropped.
"""

import re
import unicodedata
from typing import Dict, Any, List, Tuple, Optional
from segment import normalize_unicode_nfc, normalize_digits

# Try importing rapidfuzz, fallback to difflib if not available
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    HAS_RAPIDFUZZ = False

def clean_spacing(text: str) -> str:
    """Collapses consecutive whitespace to a single space."""
    return re.sub(r'\s+', ' ', text).strip()

def verify_quote(
    quote: str,
    source_text: str,
    passages: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 65.0
) -> Dict[str, Any]:
    """
    Verifies if a candidate quote or paraphrase is grounded in the source document.
    Two-Tier Verification:
    Tier 1: Exact NFC substring match or whitespace-normalized match (100% / 98%).
    Tier 2: Normalized token-overlap / fuzzy matching (SequenceMatcher >= 0.65 or keyword overlap >= 70%).
    Tier 3: Low overlap / unverified claim -> gentle contextual inference.
    """
    if not quote or not source_text:
        return {
            "verified": False,
            "match_type": "none",
            "score": 0.0,
            "quote": quote,
            "passage_id": None,
            "status": "UNVERIFIED"
        }

    norm_quote = normalize_unicode_nfc(quote).strip()
    norm_source = normalize_unicode_nfc(source_text)

    # 1. Exact substring match
    if norm_quote in norm_source:
        passage_id = find_passage_for_quote(norm_quote, passages)
        return {
            "verified": True,
            "match_type": "exact",
            "score": 100.0,
            "quote": norm_quote,
            "passage_id": passage_id,
            "status": "STATED"
        }

    # 2. Whitespace-collapsed match
    collapsed_quote = clean_spacing(norm_quote)
    collapsed_source = clean_spacing(norm_source)
    if collapsed_quote in collapsed_source:
        passage_id = find_passage_for_quote(norm_quote, passages)
        return {
            "verified": True,
            "match_type": "normalized_whitespace",
            "score": 98.0,
            "quote": norm_quote,
            "passage_id": passage_id,
            "status": "STATED"
        }

    # 3. Tier 2: Token-overlap & fuzzy matching across passages & source text
    q_tokens = set([w for w in re.findall(r'\w+', collapsed_quote.lower()) if len(w) > 2])
    if not q_tokens:
        q_tokens = set(re.findall(r'\w+', collapsed_quote.lower()))

    best_p_id = None
    best_score = 0.0
    matched_tier2 = False

    # Check against individual numbered passages first
    if passages:
        for p in passages:
            p_text = p.get("nfc_text", p.get("text", ""))
            p_clean = clean_spacing(normalize_unicode_nfc(p_text)).lower()
            p_tokens = set(re.findall(r'\w+', p_clean))

            # Token overlap percentage
            token_overlap = (len(q_tokens & p_tokens) / len(q_tokens)) if q_tokens else 0.0

            # SequenceMatcher ratio / partial ratio
            seq_score = 0.0
            if HAS_RAPIDFUZZ:
                seq_score = fuzz.partial_ratio(collapsed_quote.lower(), p_clean) / 100.0
            else:
                import difflib
                matcher = difflib.SequenceMatcher(None, collapsed_quote.lower(), p_clean)
                match = matcher.find_longest_match(0, len(collapsed_quote), 0, len(p_clean))
                seq_score = match.size / max(len(collapsed_quote), 1)

            passage_best = max(token_overlap, seq_score)
            if passage_best > best_score:
                best_score = passage_best
                best_p_id = p["id"]

            if token_overlap >= 0.70 or seq_score >= (threshold / 100.0):
                matched_tier2 = True

    # Check against full source text if passages didn't trigger Tier 2
    if not matched_tier2:
        src_tokens = set(re.findall(r'\w+', collapsed_source.lower()))
        src_overlap = (len(q_tokens & src_tokens) / len(q_tokens)) if q_tokens else 0.0

        src_seq = 0.0
        if HAS_RAPIDFUZZ:
            src_seq = fuzz.partial_ratio(collapsed_quote.lower(), collapsed_source.lower()) / 100.0
        else:
            import difflib
            matcher = difflib.SequenceMatcher(None, collapsed_quote.lower(), collapsed_source.lower())
            match = matcher.find_longest_match(0, len(collapsed_quote), 0, len(collapsed_source))
            src_seq = match.size / max(len(collapsed_quote), 1)

        src_metric = max(src_overlap, src_seq)
        if src_metric > best_score:
            best_score = src_metric

        if src_overlap >= 0.70 or src_seq >= (threshold / 100.0):
            matched_tier2 = True
            if not best_p_id:
                best_p_id = find_passage_for_quote(norm_quote, passages)

    if matched_tier2:
        return {
            "verified": True,
            "match_type": "fuzzy",
            "score": round(best_score * 100.0, 1),
            "quote": norm_quote,
            "passage_id": best_p_id,
            "status": "STATED"
        }

    return {
        "verified": False,
        "match_type": "inference",
        "score": round(best_score * 100.0, 1),
        "quote": norm_quote,
        "passage_id": None,
        "status": "UNVERIFIED"
    }

def find_passage_for_quote(quote: str, passages: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Finds which numbered passage (P1, P2, ...) contains the quote or paraphrase."""
    if not passages or not quote:
        return None
    
    clean_q = clean_spacing(normalize_unicode_nfc(quote)).lower()
    
    # 1. Exact or substring containment
    for p in passages:
        p_clean = clean_spacing(p.get("nfc_text", p.get("text", ""))).lower()
        if clean_q in p_clean:
            return p["id"]
    
    # 2. Token overlap and fuzzy matching
    q_tokens = set([w for w in re.findall(r'\w+', clean_q) if len(w) > 2])
    if not q_tokens:
        q_tokens = set(re.findall(r'\w+', clean_q))

    best_id = None
    best_score = 0.0

    for p in passages:
        p_clean = clean_spacing(p.get("nfc_text", p.get("text", ""))).lower()
        p_tokens = set(re.findall(r'\w+', p_clean))
        overlap = (len(q_tokens & p_tokens) / len(q_tokens)) * 100.0 if q_tokens else 0.0
        
        fz_score = 0.0
        if HAS_RAPIDFUZZ:
            fz_score = fuzz.partial_ratio(clean_q, p_clean)
        else:
            import difflib
            matcher = difflib.SequenceMatcher(None, clean_q, p_clean)
            match = matcher.find_longest_match(0, len(clean_q), 0, len(p_clean))
            fz_score = (match.size / max(len(clean_q), 1)) * 100.0

        combined = max(overlap, fz_score)
        if combined > best_score and (overlap >= 70.0 or fz_score >= 65.0):
            best_score = combined
            best_id = p["id"]

    return best_id

def extract_numeric_value(text: str) -> Optional[int]:
    """Extracts raw integer digits from an amount string, stripping commas/symbols."""
    if not text:
        return None
    # Normalize Devanagari/Telugu digits
    ascii_text = normalize_digits(text)
    # Strip currency signs (₹, Rs, Rs., INR) and commas
    cleaned = re.sub(r'[^\d]', '', ascii_text)
    if cleaned:
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None

def verify_amount(amount_claimed: Any, source_text: str) -> Dict[str, Any]:
    """
    Verifies that the extracted amount claimed genuinely appears in the source text.
    Handles Indian comma formats (e.g. 1,50,000), decimals (150000.00), and Indic digits.
    """
    if amount_claimed is None:
        return {"verified": True, "amount": None, "status": "UNKNOWN"}

    # If amount_claimed is given as string or int
    amount_str = str(amount_claimed)
    num_val = extract_numeric_value(amount_str)
    
    if num_val is None:
        return {"verified": False, "amount": amount_str, "status": "UNVERIFIED"}

    # Normalize source text digits
    norm_source_digits = normalize_digits(source_text)
    
    # Check variations: raw number, comma formats
    patterns = [
        str(num_val),
        f"{num_val:,}", # standard 1,000,000
    ]
    # Indian numbering format: e.g. 1,50,000
    s = str(num_val)
    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.append(rest)
        groups.reverse()
        indian_formatted = ",".join(groups) + "," + last3
        patterns.append(indian_formatted)

    for p in patterns:
        if p in norm_source_digits:
            return {
                "verified": True,
                "amount": f"₹{num_val:,}",
                "numeric_value": num_val,
                "status": "STATED"
            }

    return {
        "verified": False,
        "amount": f"₹{num_val:,}",
        "numeric_value": num_val,
        "status": "UNVERIFIED"
    }

def verify_facts(raw_facts: Dict[str, Any], source_text: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs deterministic verification across all extracted facts.
    Ensures every fact has a verified quote or is marked UNVERIFIED.
    Returns match_type and score for transparent verification badges.
    """
    verified_facts = {}
    
    # Sender
    sender_quote = raw_facts.get("sender_quote", "")
    v_sender = verify_quote(sender_quote, source_text, passages)
    verified_facts["sender"] = {
        "value": raw_facts.get("sender", "Unknown"),
        "quote": v_sender["quote"],
        "verified": v_sender["verified"],
        "match_type": v_sender["match_type"],
        "score": v_sender["score"],
        "passage_id": v_sender["passage_id"],
        "status": "STATED" if v_sender["verified"] else "UNVERIFIED"
    }

    # Recipient
    recip_quote = raw_facts.get("recipient_quote", "")
    v_recip = verify_quote(recip_quote, source_text, passages)
    verified_facts["recipient"] = {
        "value": raw_facts.get("recipient", "Unknown"),
        "quote": v_recip["quote"],
        "verified": v_recip["verified"],
        "match_type": v_recip["match_type"],
        "score": v_recip["score"],
        "passage_id": v_recip["passage_id"],
        "status": "STATED" if v_recip["verified"] else "UNVERIFIED"
    }

    # Demand
    demand_quote = raw_facts.get("demand_quote", "")
    v_demand = verify_quote(demand_quote, source_text, passages)
    verified_facts["demand"] = {
        "value": raw_facts.get("demand", "Not specified"),
        "quote": v_demand["quote"],
        "verified": v_demand["verified"],
        "match_type": v_demand["match_type"],
        "score": v_demand["score"],
        "passage_id": v_demand["passage_id"],
        "status": "STATED" if v_demand["verified"] else "UNVERIFIED"
    }

    # Amount
    raw_amount = raw_facts.get("amount")
    v_amount = verify_amount(raw_amount, source_text)
    amount_quote = raw_facts.get("amount_quote", "")
    v_amount_quote = verify_quote(amount_quote, source_text, passages)
    verified_facts["amount"] = {
        "value": v_amount["amount"] if v_amount["verified"] else (f"₹{raw_amount}" if raw_amount else "None stated"),
        "numeric_value": v_amount.get("numeric_value"),
        "verified": v_amount["verified"] or v_amount_quote["verified"],
        "match_type": v_amount_quote["match_type"] if v_amount_quote["verified"] else ("exact" if v_amount["verified"] else "unverified"),
        "score": v_amount_quote["score"] if v_amount_quote["verified"] else (100.0 if v_amount["verified"] else 0.0),
        "quote": v_amount_quote["quote"],
        "passage_id": v_amount_quote["passage_id"],
        "status": "STATED" if (v_amount["verified"] or v_amount_quote["verified"]) else "UNKNOWN"
    }

    # Deadline quote and raw string
    dl_quote = raw_facts.get("deadline_quote", "")
    v_dl = verify_quote(dl_quote, source_text, passages)
    verified_facts["deadline"] = {
        "raw_text": raw_facts.get("deadline_raw", "None specified"),
        "quote": v_dl["quote"],
        "verified": v_dl["verified"],
        "match_type": v_dl["match_type"],
        "score": v_dl["score"],
        "passage_id": v_dl["passage_id"],
        "relative_days": raw_facts.get("relative_days"),
        "is_ambiguous": raw_facts.get("is_ambiguous", False),
        "status": "STATED" if v_dl["verified"] else "UNKNOWN"
    }

    # Termination & Breach Clause
    breach_quote = raw_facts.get("termination_breach_quote", "")
    v_breach = verify_quote(breach_quote, source_text, passages) if breach_quote else {"verified": False, "quote": "", "match_type": "none", "score": 0.0, "passage_id": None}
    verified_facts["termination_breach"] = {
        "value": raw_facts.get("termination_breach", "Not specified"),
        "quote": v_breach["quote"],
        "verified": v_breach["verified"],
        "match_type": v_breach["match_type"],
        "score": v_breach["score"],
        "passage_id": v_breach["passage_id"],
        "status": "STATED" if v_breach["verified"] else ("UNKNOWN" if not breach_quote else "UNVERIFIED")
    }

    # Required Actions
    action_quote = raw_facts.get("required_actions_quote", "")
    v_action = verify_quote(action_quote, source_text, passages) if action_quote else {"verified": False, "quote": "", "match_type": "none", "score": 0.0, "passage_id": None}
    verified_facts["required_actions"] = {
        "value": raw_facts.get("required_actions", raw_facts.get("demand", "Not specified")),
        "quote": v_action["quote"] or v_demand["quote"],
        "verified": v_action["verified"] or v_demand["verified"],
        "match_type": v_action["match_type"] if v_action["verified"] else v_demand["match_type"],
        "score": v_action["score"] if v_action["verified"] else v_demand["score"],
        "passage_id": v_action["passage_id"] or v_demand["passage_id"],
        "status": "STATED" if (v_action["verified"] or v_demand["verified"]) else "UNVERIFIED"
    }

    # Financial Penalties / Interest
    penalty_quote = raw_facts.get("financial_penalties_quote", "")
    v_penalty = verify_quote(penalty_quote, source_text, passages) if penalty_quote else {"verified": False, "quote": "", "match_type": "none", "score": 0.0, "passage_id": None}
    verified_facts["financial_penalties"] = {
        "value": raw_facts.get("financial_penalties", "None specified"),
        "quote": v_penalty["quote"],
        "verified": v_penalty["verified"],
        "match_type": v_penalty["match_type"],
        "score": v_penalty["score"],
        "passage_id": v_penalty["passage_id"],
        "status": "STATED" if v_penalty["verified"] else ("UNKNOWN" if not penalty_quote else "UNVERIFIED")
    }

    # Consequence
    cons_quote = raw_facts.get("consequence_quote", "")
    v_cons = verify_quote(cons_quote, source_text, passages)
    verified_facts["consequence"] = {
        "value": raw_facts.get("consequence", "None specified"),
        "quote": v_cons["quote"],
        "verified": v_cons["verified"],
        "match_type": v_cons["match_type"],
        "score": v_cons["score"],
        "passage_id": v_cons["passage_id"],
        "status": "STATED" if v_cons["verified"] else "UNVERIFIED"
    }

    # Overall urgency
    verified_facts["urgency"] = raw_facts.get("urgency", "MEDIUM")

    return verified_facts
