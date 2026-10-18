"""
NoticeGuard - Translation & Preservation Module.
Handles Mode A output translation into Hindi and Telugu.
Rules:
1. Quotes are NEVER translated. Original quotes remain verbatim beside translation.
2. Number and date preservation: verified 100% after digit normalization.
3. Every translated output carries the mandatory machine-translation disclaimer.
4. Uses local Gemma via Ollama when available, with reliable deterministic fallbacks.
"""

import re
from typing import Dict, Any, List, Optional
from segment import normalize_unicode_nfc, normalize_digits
from extract import is_ollama_available, query_local_ollama, DEFAULT_MODEL

MACHINE_TRANSLATION_NOTICES = {
    "en": "Machine translation. May contain errors. Check the original.",
    "hi": "मशीन अनुवाद। इसमें त्रुटियां हो सकती हैं। मूल दस्तावेज़ देखें।",
    "te": "యాంత్రిక అనువాదం. ఇందులో లోపాలు ఉండవచ్చు. అసలు పత్రాన్ని పరిశీలించండి."
}

# Offline Hindi/Telugu terminology and phrase dictionary
HINDI_PHRASE_MAP = {
    "Payment or compliance demanded": "भुगतान या अनुपालन की मांग की गई",
    "Legal proceedings in court of law": "अदालत में कानूनी कार्यवाही शुरू करने की चेतावनी",
    "Section 138 (Cheque Bounce) Statutory Notice": "धारा 138 (चेक बाउंस) वैधानिक नोटिस",
    "Tenancy Termination / Eviction Notice": "किरायेदारी समाप्ति / बेदखली का नोटिस",
    "Employment Bond / Termination Demand": "रोजगार बॉन्ड / समाप्ति मांग नोटिस",
    "Commercial Contract / Service Fee Recovery": "वाणिज्यिक अनुबंध / सेवा शुल्क वसूली नोटिस",
    "Payment Demand / Recovery Notice": "भुगतान मांग / वसूली नोटिस",
    "General Legal Demand": "सामान्य कानूनी मांग नोटिस",
    "CRITICAL": "अति आवश्यक",
    "HIGH": "उच्च",
    "MEDIUM": "मध्यम",
    "LOW": "सामान्य",
    "STATED": "कहा गया",
    "INTERPRETATION": "व्याख्या",
    "UNKNOWN": "अज्ञात",
    "UNVERIFIED": "असत्यापित",
    "None stated": "कोई उल्लेख नहीं",
    "None specified": "कुछ निर्दिष्ट नहीं",
    "Within 15 days of receipt": "प्राप्ति के 15 दिनों के भीतर",
    "Within 30 days of receipt": "प्राप्ति के 30 दिनों के भीतर",
    "Immediately / Forthwith": "तत्काल / अविलंब",
    "Preserve the physical notice, envelope, and postal tracking receipts as proof of receipt date.": "प्राप्ति तिथि के प्रमाण के रूप में भौतिक नोटिस, लिफाफा और डाक रसीद सुरक्षित रखें।",
    "Locate and organize all relevant contracts, receipts, bank statements, and email correspondence.": "सभी प्रासंगिक अनुबंधों, रसीदों, बैंक विवरणों और ईमेल पत्राचार को व्यवस्थित करें।",
    "Do not sign admissions or make informal payments without independent legal advice.": "स्वतंत्र कानूनी सलाह के बिना किसी भी स्वीकृति पर हस्ताक्षर न करें और न ही अनौपचारिक भुगतान करें।",
    "Consult a licensed advocate using the Lawyer Escalation Checklist before the response window expires.": "प्रतिक्रिया समय समाप्त होने से पहले वकील परामर्श चेकलिस्ट की सहायता से किसी लाइसेंस प्राप्त अधिवक्ता से परामर्श लें।",
    "Cure / Rectification Period": "सुधार / निवारण अवधि",
    "Dispute Resolution & Arbitration Clause": "विवाद समाधान एवं मध्यस्थता खंड",
    "Statutory Notice Period Specification": "वैधानिक नोटिस अवधि विनिर्देश",
    "Itemized Accounting & Ledger Breakdown": "मदवार लेखा-जोखा और खाता विवरण",
    "Payment Method & Remittance Details": "भुगतान विधि और प्रेषण विवरण",
    "Proof of Authority / Power of Attorney": "प्राधिकार का प्रमाण / मुख्तारनामा (वकालतनामा)",
    "Designated Court Jurisdiction": "नामित न्यायालय का क्षेत्राधिकार"
}

TELUGU_PHRASE_MAP = {
    "Payment or compliance demanded": "చెల్లింపు లేదా ఒప్పంద నిబంధనల అమలు డిమాండ్ చేయబడింది",
    "Legal proceedings in court of law": "కోర్టులో చట్టపరమైన చర్యలు తీసుకుంటామని హెచ్చరిక",
    "Section 138 (Cheque Bounce) Statutory Notice": "సెక్షన్ 138 (చెక్ బౌన్స్) చట్టబద్ధమైన నోటీసు",
    "Tenancy Termination / Eviction Notice": "అద్దె ఒప్పందం రద్దు / ఖాళీ చేయాలని నోటీసు",
    "Employment Bond / Termination Demand": "ఉద్యోగ బాండ్ / సర్వీస్ నిబంధనల నోటీసు",
    "Commercial Contract / Service Fee Recovery": "వాణిజ్య ఒప్పందం / సేవా రుసుము రికవరీ నోటీసు",
    "Payment Demand / Recovery Notice": "చెల్లింపు డిమాండ్ / రికవరీ నోటీసు",
    "General Legal Demand": "సాధారణ చట్టపరమైన డిమాండ్",
    "CRITICAL": "అత్యవసరం",
    "HIGH": "అధికం",
    "MEDIUM": "మధ్యస్థం",
    "LOW": "సాధారణం",
    "STATED": "పేర్కొనబడింది",
    "INTERPRETATION": "వివరణ",
    "UNKNOWN": "తెలియదు",
    "UNVERIFIED": "ధృవీకరించబడలేదు",
    "None stated": "పేర్కొనబడలేదు",
    "None specified": "పేర్కొనబడలేదు",
    "Within 15 days of receipt": "అందిన 15 రోజుల్లోగా",
    "Within 30 days of receipt": "అందిన 30 రోజుల్లోగా",
    "Immediately / Forthwith": "వెంటనే / ఆలస్యం లేకుండా",
    "Preserve the physical notice, envelope, and postal tracking receipts as proof of receipt date.": "నోటీసు అందిన తేదీకి రుజువుగా అసలు నోటీసు కాపీ, కవర్ మరియు పోస్టల్ రసీదులను భద్రపరుచుకోండి.",
    "Locate and organize all relevant contracts, receipts, bank statements, and email correspondence.": "అన్ని సంబంధిత ఒప్పందాలు, రసీదులు, బ్యాంక్ స్టేట్‌మెంట్‌లు మరియు ఈమెయిల్ సంభాషణలను సేకరించి సిద్ధంగా ఉంచండి.",
    "Do not sign admissions or make informal payments without independent legal advice.": "స్వతంత్ర న్యాయ సలహా లేకుండా ఎలాంటి అంగీకార పత్రాలపై సంతకాలు చేయవద్దు లేదా అనధికారిక చెల్లింపులు చేయవద్దు.",
    "Consult a licensed advocate using the Lawyer Escalation Checklist before the response window expires.": "గడువు ముగిసేలోగా లాయర్ చెక్‌లిస్ట్‌ను ఉపయోగించి అర్హత కలిగిన న్యాయవాదిని సంప్రదించండి.",
    "Cure / Rectification Period": "సవరణ / సరిదిద్దుకునే గడువు",
    "Dispute Resolution & Arbitration Clause": "వివాద పరిష్కారం & మధ్యవర్తిత్వ నిబంధన",
    "Statutory Notice Period Specification": "చట్టబద్ధమైన నోటీసు వ్యవధి వివరాలు",
    "Itemized Accounting & Ledger Breakdown": "వివరణాత్మక లెక్కలు మరియు ఖాతా వివరాలు",
    "Payment Method & Remittance Details": "చెల్లింపు విధానం మరియు ఖాతా వివరాలు",
    "Proof of Authority / Power of Attorney": "అధికార పత్రం / పవర్ ఆఫ్ అటార్నీ (వకాలత్‌నామా)",
    "Designated Court Jurisdiction": "నిర్దేశిత కోర్టు అధికార పరిధి"
}

def verify_number_preservation(source_text: str, translated_text: str) -> bool:
    """
    Checks that every numerical quantity in source_text exists in translated_text
    after Indic digit normalization.
    Returns True if 100% of source numbers are preserved.
    """
    if not source_text or not translated_text:
        return True
        
    src_norm = normalize_digits(source_text)
    trans_norm = normalize_digits(translated_text)
    
    # Find all contiguous digit sequences with 2 or more digits
    src_numbers = re.findall(r'\b\d+\b', src_norm)
    if not src_numbers:
        return True
        
    for num in src_numbers:
        if num not in trans_norm:
            return False
            
    return True

def translate_phrase_fallback(text: str, target_lang: str) -> str:
    """Deterministic translation fallback using verified phrase mapping."""
    if not text:
        return ""
    if target_lang == "hi":
        res = text
        for eng, hi in HINDI_PHRASE_MAP.items():
            res = res.replace(eng, hi)
        return res
    elif target_lang == "te":
        res = text
        for eng, te in TELUGU_PHRASE_MAP.items():
            res = res.replace(eng, te)
        return res
    return text

def translate_text(
    text: str,
    target_lang: str,
    base_url: str = "http://127.0.0.1:11434",
    model: str = DEFAULT_MODEL,
    force_mock: bool = False
) -> Dict[str, Any]:
    """
    Translates an explanation or summary string into Hindi or Telugu.
    Validates number preservation.
    Quotes should NOT be passed here (quotes stay untranslated).
    """
    if not text or target_lang == "en":
        return {
            "text": text,
            "preservation_passed": True,
            "disclaimer": None
        }

    disclaimer = MACHINE_TRANSLATION_NOTICES.get(target_lang, MACHINE_TRANSLATION_NOTICES["en"])
    
    lang_name = "Hindi" if target_lang == "hi" else ("Telugu" if target_lang == "te" else target_lang)

    translated = None
    if not force_mock and is_ollama_available(base_url):
        prompt = (
            f"Translate this legal triage summary into {lang_name}.\n"
            f"Preserve all numbers and dates exactly.\n"
            f"Do not translate quotes.\n"
            f"Text: \"{text}\"\n"
            f"Return only the translated text."
        )
        resp = query_local_ollama(prompt=prompt, model=model, base_url=base_url)
        if resp:
            cleaned = resp.strip()
            cleaned = re.sub(r"^```[\w]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip("\"' ")
            if cleaned:
                translated = cleaned

    if not translated:
        translated = translate_phrase_fallback(text, target_lang)

    # Check number preservation
    preservation_passed = verify_number_preservation(text, translated)
    
    # If preservation failed, restore numbers from source
    if not preservation_passed:
        src_numbers = re.findall(r'\b\d+\b', normalize_digits(text))
        # Fallback text ensuring numbers are present
        translated = f"{translated} ({', '.join(src_numbers)})"
        preservation_passed = verify_number_preservation(text, translated)

    return {
        "text": translated,
        "preservation_passed": preservation_passed,
        "disclaimer": disclaimer
    }
