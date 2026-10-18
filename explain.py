"""
NoticeGuard - Plain-Language Explanation Module.
Produces calm, non-intimidating explanations for each key legal passage.
Every item is strictly tagged with:
- STATED: Verbatim, verified fact stated in the notice.
- INTERPRETATION: Contextual meaning in plain language (not legal advice).
- UNKNOWN: Vital facts the notice omits or fails to address.
"""

import re
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple
from segment import normalize_unicode_nfc
from extract import is_ollama_available, query_local_ollama, extract_json_from_llm_response, DEFAULT_MODEL
from guardrails import (
    scan_for_injection,
    is_verdict_query,
    get_refusal_message,
    get_silent_message,
    sanitize_output_verdicts
)
from translate import translate_text

EXPLAIN_SYSTEM_PROMPT = """You are NoticeGuard's calm legal interpreter.
Your role is to explain legal notice passages in calm, reassuring, plain English for a student or freelancer.
CRITICAL RULES:
1. Label every point strictly as "STATED", "INTERPRETATION", or "UNKNOWN".
2. STATED points must directly reflect what is written.
3. INTERPRETATION points explain what it means in simple terms.
4. UNKNOWN points highlight crucial details that the notice fails to state.
5. NEVER say "you are guilty" or "this notice is illegal". No legal verdicts.
6. Return JSON only.

SCHEMA:
[
  {
    "passage_id": "P1",
    "quote": "verbatim text from passage",
    "stated": "calm plain-language summary of what is directly stated",
    "interpretation": "plain meaning of the clause without fear-mongering",
    "unknown": "what the passage leaves unsaid or does not clarify"
  }
]
"""

def generate_plain_explanation(
    passages: List[Dict[str, Any]],
    verified_facts: Dict[str, Any],
    base_url: str = "http://127.0.0.1:11434",
    model: str = DEFAULT_MODEL,
    force_mock: bool = False
) -> List[Dict[str, Any]]:
    """
    Generates plain-language breakdowns for document passages.
    Uses local Gemma when available; falls back to deterministic rule-based generator
    if Ollama is offline or force_mock is True.
    """
    if not force_mock and is_ollama_available(base_url) and len(passages) > 0:
        # Prepare condensed passages text
        passages_text = "\n\n".join([f"[{p['id']}]: {p['text']}" for p in passages[:8]])
        prompt = f"Explain these legal notice passages for a regular person:\n\n{passages_text}\n\nReturn JSON array only."
        
        response = query_local_ollama(
            prompt=prompt,
            system_prompt=EXPLAIN_SYSTEM_PROMPT,
            model=model,
            base_url=base_url
        )
        
        parsed = extract_json_from_llm_response(response) if response else None
        if parsed and isinstance(parsed, list):
            # Sanitize verdicts from model output
            for item in parsed:
                item["stated"] = sanitize_output_verdicts(item.get("stated", ""))
                item["interpretation"] = sanitize_output_verdicts(item.get("interpretation", ""))
                item["unknown"] = sanitize_output_verdicts(item.get("unknown", ""))
            return parsed

    # Deterministic fallback generator
    explanations = []
    for p in passages:
        p_text = p.get("text", "")
        p_id = p.get("id", "")
        
        stated_text = ""
        interp_text = ""
        unknown_text = ""
        
        p_lower = p_text.lower()
        
        if "under instructions from" in p_lower or "on behalf of" in p_lower or "advocate" in p_lower:
            stated_text = "The sender's advocate states that they are authorized to send this formal notice on behalf of their client."
            interp_text = "This identifies who is making the claim and who their legal representative is."
            unknown_text = "The notice does not prove whether the advocate holds an executed power of attorney or vakalatnama."
        elif "sum of" in p_lower or "rs." in p_lower or "₹" in p_lower or "amount" in p_lower or "balance" in p_lower:
            stated_text = "The sender states that a specific financial balance or debt is outstanding and demands its payment."
            interp_text = "The sender is asserting an unpaid monetary balance. A claim in a notice is not a court judgment."
            unknown_text = "The notice does not provide an itemized audit, ledger statement, or invoices verifying how this exact figure was calculated."
        elif "vacate" in p_lower or "premises" in p_lower or "tenancy" in p_lower or "possession" in p_lower:
            stated_text = "The notice demands that you vacate the premises and hand over physical possession."
            interp_text = "The landlord is issuing a termination warning. Under Indian tenancy law, formal eviction requires due process of law."
            unknown_text = "The notice does not account for the security deposit balance or formal move-out inspection details."
        elif "dishonour" in p_lower or "138" in p_lower or "cheque" in p_lower:
            stated_text = "The sender asserts that a cheque issued by you was returned unpaid by the bank."
            interp_text = "This refers to statutory proceedings under the Negotiable Instruments Act. A 15-day notice period is required by law."
            unknown_text = "The notice does not specify whether the bank memo was received within the statutory 30-day window."
        elif "failing which" in p_lower or "legal proceedings" in p_lower or "court" in p_lower or "police" in p_lower:
            stated_text = "The sender states that if demands are not fulfilled, they intend to institute legal proceedings."
            interp_text = "This is standard legal demand language. It indicates intent to file a suit, not an active court summons or police warrant."
            unknown_text = "The notice does not provide a court case number or summons (none has been filed yet)."
        else:
            first_sentence = p_text.split('.')[0].strip()
            stated_text = f"The notice states: '{first_sentence}'"
            interp_text = "This clause outlines the factual narrative from the sender's viewpoint."
            unknown_text = "The document does not state your side of the dispute or previous correspondence."

        explanations.append({
            "passage_id": p_id,
            "quote": p_text[:200] + ("..." if len(p_text) > 200 else ""),
            "stated": sanitize_output_verdicts(stated_text),
            "interpretation": sanitize_output_verdicts(interp_text),
            "unknown": sanitize_output_verdicts(unknown_text)
        })

    return explanations

def answer_followup_question(
    question: str,
    passages: List[Dict[str, Any]],
    lang: str = "en",
    base_url: str = "http://127.0.0.1:11434",
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Dedicated follow-up Q&A engine for NoticeGuard.
    Passes numbered source passages ([P1], [P2], etc.) and the user's question directly
    to local Gemma 4 (gemma4:e4b) via http://127.0.0.1:11434/api/generate (stream=False, timeout=120).
    Answers directly in plain language using ONLY provided passages, citing [P#],
    and strictly enforces verdict/non-lawyer guardrails and prompt injection checks.
    """
    q_clean = question.strip() if question else ""
    if not q_clean:
        return {
            "answer": "Please enter a question about this notice.",
            "passage_id": None,
            "status": "silent"
        }

    # 1. Prompt Injection Guardrail Scan
    has_inj, patterns = scan_for_injection(q_clean)
    if has_inj:
        return {
            "answer": "⚠️ Security Alert: Query contains disallowed override instructions or adversarial prompts.",
            "passage_id": None,
            "status": "refusal"
        }

    # 2. Judicial Verdict / Enforceability / Payment Advice Guardrail
    if is_verdict_query(q_clean):
        refusal_msg = get_refusal_message(lang)
        return {
            "answer": refusal_msg,
            "passage_id": None,
            "status": "refusal"
        }

    # 3. Format numbered source passages [P1], [P2]...
    passages_context = "\n\n".join([f"[{p.get('id', f'P{i+1}')}]: {p.get('text', '')}" for i, p in enumerate(passages)])

    qa_system_prompt = (
        "You are NoticeGuard's calm, helpful legal document reader assistant.\n"
        "Your role is to answer the user's question directly in plain language using ONLY the provided numbered legal notice passages.\n"
        "RULES:\n"
        "1. Every factual statement must cite the source passage tag, e.g. [P1] or [P2].\n"
        "2. If the passages do not state the answer, reply: 'The notice does not address this.'\n"
        "3. NEVER provide legal opinions, enforceability verdicts, or advice on whether to pay or litigate.\n"
        "4. Keep your answer plain, objective, and reassuring without legal jargon."
    )

    user_prompt = (
        f"Numbered Source Passages from Notice:\n"
        f"{passages_context}\n\n"
        f"User Question: {q_clean}\n\n"
        f"Direct Grounded Answer (citing [P#]):"
    )

    raw_answer = None

    # Call local Ollama gemma4:e4b
    if is_ollama_available(base_url):
        payload = {
            "model": model,
            "prompt": user_prompt,
            "system": qa_system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                ans_candidate = resp_data.get("response", "").strip()
                if ans_candidate:
                    raw_answer = ans_candidate
        except Exception:
            raw_answer = None

    # 4. Fallback if Ollama is offline or returned empty
    if not raw_answer:
        q_tokens = [w for w in re.findall(r'\w+', q_clean.lower()) if len(w) > 3]
        best_p = None
        best_overlap = 0
        for p in passages:
            p_text = p.get("text", "")
            p_words = set(re.findall(r'\w+', p_text.lower()))
            overlap = sum(1 for w in q_tokens if w in p_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_p = p

        if best_p and best_overlap > 0:
            raw_answer = f"According to passage [{best_p['id']}]: \"{best_p['text'][:250]}...\""
        else:
            raw_answer = "The notice does not address this."

    # 5. Sanitize any legal verdicts
    sanitized_answer = sanitize_output_verdicts(raw_answer, language=lang)

    # 6. Check if silent handler applies
    silent_indicators = [
        "not address", "doesn't address", "not mentioned", "not stated",
        "does not clarify", "does not contain", "unaddressed", "not specified"
    ]
    if any(ind in sanitized_answer.lower() for ind in silent_indicators):
        silent_msg = get_silent_message(lang)
        return {
            "answer": silent_msg,
            "passage_id": None,
            "status": "silent"
        }

    # 7. Extract cited passage ID if present
    passage_match = re.search(r'\[(P\d+)\]', sanitized_answer)
    found_p_id = passage_match.group(1) if passage_match else None

    # 8. Decouple Translation: Translate response if language is Hindi or Telugu
    final_answer = sanitized_answer
    if lang in ("hi", "te"):
        try:
            trans_res = translate_text(final_answer, target_lang=lang, base_url=base_url, model=model)
            final_answer = trans_res.get("text", final_answer)
        except Exception:
            pass

    return {
        "answer": final_answer,
        "passage_id": found_p_id,
        "status": "grounded"
    }

# Multilingual Legal Jargon Glossary Definitions
LEGAL_GLOSSARY_DEFINITIONS = {
    "indemnity": {
        "term_en": "Indemnity",
        "meaning_en": "A promise by one party to pay for or protect the other party against financial losses, lawsuits, or damages.",
        "term_hi": "क्षतिपूर्ति (Indemnity)",
        "meaning_hi": "एक पक्ष द्वारा दूसरे पक्ष को होने वाले वित्तीय नुकसान या कानूनी हानि की भरपाई करने का औपचारिक वादा।",
        "term_te": "నష్టపరిహారం (Indemnity)",
        "meaning_te": "ఒక పక్షానికి కలిగే ఆర్థిక నష్టాలను లేదా హానిని భర్తీ చేస్తామని మరొకరు చేసే చట్టపరమైన వాగ్దానం."
    },
    "force majeure": {
        "term_en": "Force Majeure",
        "meaning_en": "Unforeseen catastrophic events (natural disasters, floods, war) that legally excuse parties from fulfilling contractual duties.",
        "term_hi": "अप्रत्याशित घटना (Force Majeure)",
        "meaning_hi": "प्राकृतिक आपदा या युद्ध जैसी असाधारण घटनाएं जो किसी पक्ष को अनुबंध का पालन न करने पर कानूनी छूट देती हैं।",
        "term_te": "అనివార్య పరిస్థితులు (Force Majeure)",
        "meaning_te": "వరదలు, యుద్ధం వంటి ఊహించని విపత్కర పరిస్థితులు తలెత్తినప్పుడు ఒప్పంద బాధ్యతల నుండి మినహాయింపునిచ్చే నిబంధన."
    },
    "cure period": {
        "term_en": "Cure Period",
        "meaning_en": "A formal grace period given to fix an alleged default or mistake before the other party can terminate or sue.",
        "term_hi": "सुधार अवधि (Cure Period)",
        "meaning_hi": "अनुबंध रद्द करने या मुकदमा दर्ज करने से पहले कथित गलती या चूक को सुधारने के लिए दी गई औपचारिक छूट अवधि।",
        "term_te": "సవరణ గడువు (Cure Period)",
        "meaning_te": "చట్టపరమైన చర్యలు తీసుకోవడానికి ముందు జరిగిన పొరపాటును సరిదిద్దుకోవడానికి ఇచ్చే అధికారిక గడువు."
    },
    "liquidated damages": {
        "term_en": "Liquidated Damages",
        "meaning_en": "A pre-agreed, fixed monetary penalty specified in a contract to be paid if a specific breach occurs.",
        "term_hi": "निर्धारित हर्जाना (Liquidated Damages)",
        "meaning_hi": "अनुबंध में पहले से तय की गई निश्चित जुर्माना राशि जो किसी शर्त का उल्लंघन होने पर चुकानी होती है।",
        "term_te": "ముందుగా నిర్ణయించిన నష్టపరిహారం (Liquidated Damages)",
        "meaning_te": "ఒప్పందం ఉల్లంఘించినప్పుడు చెల్లించాల్సినట్లుగా ముందుగానే ఒప్పందంలో నిర్ణయించబడిన నిర్దిష్ట జరిమానా మొత్తం."
    },
    "show cause": {
        "term_en": "Show Cause",
        "meaning_en": "A formal notice requiring you to explain why a proposed legal, financial, or disciplinary penalty should not be taken against you.",
        "term_hi": "कारण बताओ (Show Cause)",
        "meaning_hi": "एक औपचारिक कानूनी नोटिस जिसमें आपसे यह स्पष्टीकरण मांगा जाता है कि आपके खिलाफ दंडात्मक कार्रवाई क्यों न की जाए।",
        "term_te": "కారణం చూపండి (Show Cause)",
        "meaning_te": "మీపై క్రమశిక్షణ లేదా చట్టపరమైన చర్యలు ఎందుకు తీసుకోకూడదో వివరణ ఇవ్వాలని కోరే అధికారిక నోటీసు."
    },
    "without prejudice": {
        "term_en": "Without Prejudice",
        "meaning_en": "A label on legal letters ensuring that statements or compromise offers cannot be used as admissions of guilt in court.",
        "term_hi": "पूर्वाग्रह के बिना (Without Prejudice)",
        "meaning_hi": "पत्र पर लिखा एक कानूनी शब्द जिसका अर्थ है कि इसमें लिखी गई बातों को बाद में अदालत में अपराध की स्वीकृति नहीं माना जा सकता।",
        "term_te": "పక్షపాతం లేకుండా (Without Prejudice)",
        "meaning_te": "ఈ సంభాషణ లేదా ప్రతిపాదనలను భవిష్యత్తులో కోర్టులో నేరం ఒప్పుకున్నట్లుగా సాక్ష్యంగా ఉపయోగించలేరని తెలిపే నిబంధన."
    },
    "subrogation": {
        "term_en": "Subrogation",
        "meaning_en": "The legal right of an entity (like an insurer) to step into another person's shoes to pursue damages against a third party.",
        "term_hi": "प्रत्यास्थापन (Subrogation)",
        "meaning_hi": "किसी संस्था (जैसे बीमा कंपनी) द्वारा तीसरे पक्ष से नुकसान की वसूली के लिए मूल पीड़ित के कानूनी अधिकारों को ग्रहण करना।",
        "term_te": "ప్రత్యామ్నాయ హక్కు (Subrogation)",
        "meaning_te": "మూడవ పక్షం నుండి నష్టాన్ని వసూలు చేయడానికి ఒక సంస్థ బాధితుడి చట్టపరమైన స్థానంలోకి ప్రవేశించే హక్కు."
    },
    "eviction": {
        "term_en": "Eviction",
        "meaning_en": "The formal legal procedure required to lawfully remove a tenant or occupant from rented property.",
        "term_hi": "बेदखली (Eviction)",
        "meaning_hi": "किसी किरायेदार या कब्जेदार को किराये की संपत्ति से कानूनी रूप से बाहर निकालने की औपचारिक प्रक्रिया।",
        "term_te": "ఖాళీ చేయించడం (Eviction)",
        "meaning_te": "అద్దెదారుని లేదా ఆక్రమణదారుని ఆస్తి నుండి చట్టబద్ధంగా తొలగించే అధికారిక ప్రక్రియ."
    },
    "breach": {
        "term_en": "Breach of Contract",
        "meaning_en": "Breaking a promise or failing to perform an obligation explicitly agreed upon in a legally binding agreement.",
        "term_hi": "अनुबंध का उल्लंघन (Breach)",
        "meaning_hi": "कानूनी समझौते में तय की गई किसी शर्त, वादे या कर्तव्य को पूरा न करना या तोड़ना।",
        "term_te": "ఒప్పంద ఉల్లంఘన (Breach)",
        "meaning_te": "ఒప్పందంలో అంగీకరించిన బాధ్యతను నెరవేర్చడంలో విఫలమవడం లేదా షరతును ఉల్లంఘించడం."
    },
    "arbitration": {
        "term_en": "Arbitration",
        "meaning_en": "A private, legally binding dispute-resolution method conducted outside courtrooms by an independent arbitrator.",
        "term_hi": "मध्यस्थता (Arbitration)",
        "meaning_hi": "अदालत जाए बिना किसी स्वतंत्र निष्पक्ष मध्यस्थ (Arbitrator) द्वारा विवाद को कानूनी रूप से सुलझाने की प्रक्रिया।",
        "term_te": "మధ్యవర్తిత్వం (Arbitration)",
        "meaning_te": "కోర్టుకు వెళ్లకుండా ఒక స్వతంత్ర మధ్యవర్తి ద్వారా వివాదాన్ని చట్టబద్ధంగా పరిష్కరించే ప్రక్రియ."
    },
    "liability": {
        "term_en": "Legal Liability",
        "meaning_en": "Financial or legal responsibility enforceable under law to satisfy debts or remedy damages.",
        "term_hi": "कानूनी दायित्व (Liability)",
        "meaning_hi": "कानून के तहत किसी बकाये का भुगतान करने, नुकसान की भरपाई करने या कर्तव्य निभाने की कानूनी जिम्मेदारी।",
        "term_te": "చట్టపరమైన బాధ్యత (Liability)",
        "meaning_te": "నష్టాన్ని భర్తీ చేయడానికి లేదా రుణాన్ని చెల్లించడానికి చట్టం ప్రకారం బాధ్యత వహించడం."
    },
    "pursuant": {
        "term_en": "Pursuant to",
        "meaning_en": "In accordance with, following, or carrying out a specific legal clause, statute, or contract term.",
        "term_hi": "के अनुसरण में (Pursuant to)",
        "meaning_hi": "किसी विशेष कानूनी धारा, अधिनियम या अनुबंध की शर्त के अनुसार या उसके तहत।",
        "term_te": "అనుగుణంగా (Pursuant to)",
        "meaning_te": "ఒక నిర్దిష్ట చట్ట నిబంధన లేదా ఒప్పంద షరతు ప్రకారం అమలు చేయడం."
    },
    "severability": {
        "term_en": "Severability",
        "meaning_en": "A clause stating that if any single term is found illegal, the rest of the contract remains valid and enforceable.",
        "term_hi": "पृथक्करणीयता (Severability)",
        "meaning_hi": "यह नियम कि यदि समझौते का कोई एक खंड अवैध हो जाता है, तो भी शेष समझौता पूरी तरह वैध रहेगा।",
        "term_te": "వేరుచేయగల నిబంధన (Severability)",
        "meaning_te": "ఒప్పందంలో ఏదైనా ఒక నిబంధన చెల్లకపోయినా, మిగిలిన ఒప్పందం మొత్తం చెల్లుబాటు అవుతుందని తెలిపే నిబంధన."
    },
    "mesne profits": {
        "term_en": "Mesne Profits",
        "meaning_en": "Damages that an unlawful occupant of a property must pay to the rightful owner for wrongfully staying after lease expiry.",
        "term_hi": "मध्यवर्ती लाभ / हर्जाना (Mesne Profits)",
        "meaning_hi": "किरायेदारी समाप्त होने के बाद भी संपत्ति पर अवैध कब्जा रखने पर मकान मालिक को चुकाया जाने वाला हर्जाना।",
        "term_te": "అనధికార ఆక్రమణ నష్టపరిహారం (Mesne Profits)",
        "meaning_te": "లీజు ముగిసిన తర్వాత కూడా అక్రమంగా ఆస్తిలో కొనసాగినందుకు యజమానికి చెల్లించాల్సిన రోజువారీ నష్టపరిహారం."
    },
    "vakalatnama": {
        "term_en": "Vakalatnama",
        "meaning_en": "A formal written document authorizing a specific advocate to represent and act on behalf of a client in legal proceedings.",
        "term_hi": "वकालतनामा (Vakalatnama)",
        "meaning_hi": "एक औपचारिक कानूनी अधिकार-पत्र जिसके द्वारा मुवक्किल अपने वकील को अपनी ओर से कानूनी कार्रवाई करने का अधिकार देता है।",
        "term_te": "వకాలత్‌నామా (Vakalatnama)",
        "meaning_te": "కోర్టులో క్లయింట్ తరపున వాదించడానికి మరియు వ్యవహరించడానికి న్యాయవాదికి ఇచ్చే అధికార పత్రం."
    },
    "jurisdiction": {
        "term_en": "Court Jurisdiction",
        "meaning_en": "The geographical territory or official authority of a specific court to hear and adjudicate a legal case.",
        "term_hi": "न्यायालय का क्षेत्राधिकार (Jurisdiction)",
        "meaning_hi": "किसी विशेष अदालत का वह भौगोलिक क्षेत्र या कानूनी दायरा जिसके भीतर वह मुकदमे की सुनवाई कर सकती है।",
        "term_te": "కోర్టు అధికార పరిధి (Jurisdiction)",
        "meaning_te": "ఒక నిర్దిష్ట కోర్టుకు కేసును విచారించడానికి గల భౌగోళిక లేదా చట్టపరమైన అధికార పరిధి."
    },
    "limitation period": {
        "term_en": "Limitation Period",
        "meaning_en": "The legal time limit set by law within which a civil lawsuit or criminal complaint must be initiated.",
        "term_hi": "परिसीमा अवधि (Limitation Period)",
        "meaning_hi": "कानून द्वारा निर्धारित वह अंतिम समय-सीमा जिसके समाप्त होने के बाद अदालत में मुकदमा दायर नहीं किया जा सकता।",
        "term_te": "కాలపరిమితి (Limitation Period)",
        "meaning_te": "ఒక వివాదంపై కోర్టులో కేసు దాఖలు చేయడానికి చట్టం నిర్ణయించిన గరిష్ట సమయ పరిమితి."
    },
    "statutory notice": {
        "term_en": "Statutory Notice",
        "meaning_en": "A formal written demand required by an act of parliament before a party is legally permitted to sue in court.",
        "term_hi": "वैधानिक नोटिस (Statutory Notice)",
        "meaning_hi": "किसी कानून (जैसे चेक बाउंस अधिनियम) के तहत अदालत में केस करने से पहले प्रेषित किया जाने वाला अनिवार्य लिखित नोटिस।",
        "term_te": "చట్టబద్ధమైన నోటీసు (Statutory Notice)",
        "meaning_te": "కోర్టులో దావా వేయడానికి ముందు ఒక చట్టం ప్రకారం తప్పనిసరిగా పంపవలసిన అధికారిక లిఖితపూర్వక నోటీసు."
    }
}

def extract_legal_glossary(text: str, lang: str = "en") -> List[Dict[str, str]]:
    """
    Extracts or matches 4 to 8 difficult legal words found in the notice text,
    with built-in dictionary fallbacks so it always populates immediately.
    Returns list of dicts: [{'term': ..., 'meaning': ...}] localized to target language.
    """
    if not text:
        text = ""
    lower_text = text.lower()
    matched_keys = []

    # Find terms occurring in document
    for key in LEGAL_GLOSSARY_DEFINITIONS:
        if re.search(r'\b' + re.escape(key) + r'\b', lower_text):
            matched_keys.append(key)

    # Ensure at least 4-6 terms are present using foundational legal terms
    fallback_keys = ["breach", "liability", "statutory notice", "cure period", "arbitration", "jurisdiction", "indemnity"]
    for fb in fallback_keys:
        if len(matched_keys) >= 6:
            break
        if fb not in matched_keys:
            matched_keys.append(fb)

    # Build localized output
    glossary_items = []
    for key in matched_keys[:8]:
        spec = LEGAL_GLOSSARY_DEFINITIONS.get(key, {})
        if lang == "hi":
            term = spec.get("term_hi", spec.get("term_en", key.title()))
            meaning = spec.get("meaning_hi", spec.get("meaning_en", ""))
        elif lang == "te":
            term = spec.get("term_te", spec.get("term_en", key.title()))
            meaning = spec.get("meaning_te", spec.get("meaning_en", ""))
        else:
            term = spec.get("term_en", key.title())
            meaning = spec.get("meaning_en", "")

        glossary_items.append({
            "term": term,
            "meaning": meaning,
            "raw_key": key
        })

    return glossary_items

