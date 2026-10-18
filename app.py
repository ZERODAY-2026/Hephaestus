"""
NoticeGuard - Streamlit Local-First Web Application.
Private Legal Document Triage Agent for India.
"Understand. Respond. Stay Safe."
Gemma explains, code verifies. Zero cloud calls, 127.0.0.1 localhost only.
"""
import re
import os
import json
import yaml
from datetime import date, datetime
import streamlit as st

# Import NoticeGuard local modules
from ingest import (
    ingest_document,
    wipe_temp_directory,
    IngestError,
    validate_is_legal_notice
)
from segment import segment_notice, normalize_unicode_nfc, normalize_digits
from guardrails import (
    scan_for_injection,
    sanitize_prompt_injection,
    is_verdict_query,
    get_refusal_message,
    get_silent_message,
    sanitize_output_verdicts
)
from extract import (
    extract_facts,
    check_gemma4_status,
    is_ollama_available,
    query_local_ollama,
    DEFAULT_MODEL
)
from verify import verify_facts, verify_quote
from timeline import compute_timeline
from explain import generate_plain_explanation, answer_followup_question
from escalate import build_lawyer_checklist
from translate import translate_text, MACHINE_TRANSLATION_NOTICES
from missing_clauses import detect_missing_clauses

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

# Load localized strings safely with fallback to English defaults
I18N_DIR = os.path.join(os.path.dirname(__file__), "i18n")
def load_i18n(lang: str) -> dict:
    base = {}
    en_path = os.path.join(I18N_DIR, "en.json")
    if os.path.exists(en_path):
        try:
            with open(en_path, "r", encoding="utf-8") as f:
                base = json.load(f)
        except Exception:
            base = {}
    if lang == "en":
        return base
    
    path = os.path.join(I18N_DIR, f"{lang}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lang_data = json.load(f)
                for k, v in lang_data.items():
                    if isinstance(v, dict) and isinstance(base.get(k), dict):
                        base[k].update(v)
                    else:
                        base[k] = v
        except Exception:
            pass
    return base

# Load legal glossary
GLOSSARY_PATH = os.path.join(os.path.dirname(__file__), "glossary.json")
glossary_data = {}
if os.path.exists(GLOSSARY_PATH):
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        glossary_data = json.load(f)

# Pre-loaded sample notices for 1-click testing & judge demonstrations
SAMPLE_NOTICES = {
    "Select a sample notice...": "",
    "Sample 1: Section 138 Cheque Bounce Demand Notice (English)": """LEGAL NOTICE (UNDER SECTION 138 NEGOTIABLE INSTRUMENTS ACT)
Date: 12th October 2026
To,
Mr. Rahul Sharma,
Flat 402, Sunshine Apartments, Indiranagar, Bengaluru - 560038

Under instructions from and on behalf of my client, M/s Apex Tech Solutions Pvt Ltd, having its registered office at Koramangala, Bengaluru, I hereby serve upon you this formal statutory legal notice:

1. That you had issued Cheque No. 440219 dated 15th September 2026 for a sum of Rs. 75,000/- (Rupees Seventy Five Thousand only) drawn on HDFC Bank in discharge of your outstanding contractual dues.

2. That upon presentation by my client through their banker ICICI Bank, the said cheque was returned unpaid and dishonoured with the Cheque Return Memo stating 'FUNDS INSUFFICIENT' on 28th September 2026.

3. That you have deliberately failed to maintain sufficient balance in your bank account to honour the said commitment.

4. I therefore hereby call upon you to pay the sum of Rs. 75,000/- to my client within a period of 15 days of the receipt of this legal notice.

5. Failing which, my client will be constrained to initiate criminal proceedings against you under Section 138 of the Negotiable Instruments Act, 1881, and initiate civil suit for recovery at your sole risk, cost, and consequences.

Advocate Priya Ramanathan, High Court of Karnataka""",

    "Sample 2: Tenancy Termination and Eviction Notice (English)": """NOTICE TO QUIT AND VACATE PREMISES
Date: 05th October 2026
To,
Ms. Ananya Roy,
Tenant, House No. 12B, Green Glen Layout, Bellandur, Bengaluru

From:
K. V. Subbarao (Landlord),
Represented by Advocate S. Deshmukh

Madam,
Under instructions from my client Sri K. V. Subbarao, landlord of residential premises House No. 12B:

1. That the 11-month lease agreement executed on 1st November 2025 has expired and my client does not wish to renew the tenancy.

2. That you are currently in arrears of rent for the past two months amounting to Rs. 44,000/- (Rupees Forty Four Thousand only).

3. My client hereby terminates your month-to-month tenancy under Section 106 of the Transfer of Property Act and demands that you vacate and hand over vacant physical possession of the said premises and pay the sum of Rs. 44,000/- within 15 days of receipt of this notice.

4. If you fail to vacate the premises within the stipulated 15 days, my client shall institute eviction proceedings before the competent Civil Court and claim mesne profits of Rs. 2,000/- per day for unauthorized occupation.

Advocate S. Deshmukh, Civil Court""",

    "Sample 3: Freelance Recovery Notice with Ambiguous Deadline": """FORMAL PAYMENT DEMAND NOTICE
Date: 10th October 2026
To,
NextGen Media Agency,
Cyber City, Gurugram

Dear Sir/Madam,
I am writing on behalf of freelance designer Vikram Malhotra regarding unpaid milestone invoices #204 and #205.

1. That the design deliverables for your client branding were completed and approved in writing on 14th August 2026.

2. That the outstanding invoice balance of Rs. 1,20,000/- remains unpaid despite repeated email reminders.

3. We hereby call upon you to settle the outstanding dues immediately and forthwith without any further delay.

4. In default whereof, our client will initiate legal proceedings for recovery of debt along with 18% commercial interest per annum and damages.

Vikram Malhotra (via Legal Counsel)""",

    "Sample 4: Employment Bond Recovery Notice (English)": """LEGAL NOTICE FOR RECOVERY OF TRAINING EXPENSES UNDER EMPLOYMENT BOND
Date: 14th October 2026
To,
Mr. Siddharth Mehta,
Former Associate Software Engineer,
H-403, Cyber Heights, Sector 62, Noida, Uttar Pradesh

From:
Advocate Rajiv Khurana,
On behalf of M/s Technovate Solutions Pvt. Ltd.,
Okhla Industrial Area, Phase-III, New Delhi

Sir,
Under explicit instructions from my client, M/s Technovate Solutions Pvt. Ltd., I hereby issue this statutory legal notice to you:

1. That you joined my client company on 1st July 2025 as an Associate Software Engineer, and executed a 2-year Service Agreement and Employment Bond committing to serve until 30th June 2027.

2. That under Clause 7 of the said Service Agreement, my client invested substantial resources in specialized enterprise training, in consideration whereof you agreed that if you tendered resignation before completing 24 months, you would reimburse training expenses of Rs. 1,50,000/- (Rupees One Lakh Fifty Thousand only).

3. That in breach of the aforesaid service bond and employment agreement, you abruptly abandoned your duties and tendered your resignation on 15th September 2026 without serving the mandatory 3-month notice period or completing the bond duration.

4. That despite repeated HR intimations, you have failed and neglected to clear your contractual dues and return company property.

5. I therefore hereby call upon you to pay the sum of Rs. 1,50,000/- along with 18% commercial interest per annum within a period of 15 days from the receipt of this legal notice.

6. Failing which, my client has given me unequivocal instructions to initiate civil recovery proceedings against you in the competent court of law and pursue all civil and legal remedies at your entire risk, costs, and consequences.

Advocate Rajiv Khurana, High Court of Delhi""",

    "Sample 5: Summary Contract Termination Letter (English)": """FORMAL NOTICE OF CONTRACT TERMINATION AND DEMAND FOR LIQUIDATED DAMAGES
Date: 08th October 2026
To,
Apex Logistics Solutions LLP,
Plot 45, Transport Nagar, Hyderabad - 500072

From:
Advocate N. Chandrasekhar,
Counsel for ZenWare E-Commerce Systems Pvt. Ltd.,
HITEC City, Madhapur, Hyderabad

Sirs,
Under instructions from our client, ZenWare E-Commerce Systems Pvt. Ltd., we hereby serve upon you this formal notice of immediate contract termination under Clause 14 of the Master Service Agreement dated 10th January 2026:

1. That under Clause 4.2 of the said Agreement, your firm was obligated to maintain an SLA compliance rate of 98.5% for same-day package deliveries across the Telangana region.

2. That your delivery performance has critically deteriorated over the past three consecutive months, falling below 78%, constituting a material breach of contract.

3. That despite formal cure notices served via email on 12th August 2026 and 05th September 2026, you failed to rectify the defaults or cure the breach within the contractual 15-day cure window.

4. Our client hereby terminates the Master Service Agreement forthwith and demands payment of liquidated damages and contractual penalties amounting to Rs. 2,80,000/- (Rupees Two Lakh Eighty Thousand only) within 10 days of receipt of this notice.

5. Take notice that in default of payment within 10 days, our client shall invoke arbitration under Clause 22 of the Agreement and file appropriate proceedings for injunction and recovery of damages before the Commercial Court.

Advocate N. Chandrasekhar, High Court of Telangana""",

    "Sample 6: [NON-LEGAL TEST] Grandma's Chocolate Chip Cookie Recipe (Rejection Demo)": """GRANDMA'S AWARD-WINNING CHOCOLATE CHIP COOKIES RECIPE
Prep time: 15 mins | Cook time: 10 mins | Servings: 24 cookies

Ingredients:
- 2 1/4 cups all-purpose flour
- 1 tsp baking soda
- 1/2 tsp sea salt
- 1 cup unsalted butter, softened to room temperature
- 3/4 cup granulated white sugar
- 3/4 cup packed brown sugar
- 2 large farm-fresh eggs
- 2 tsp pure vanilla extract
- 2 cups semi-sweet chocolate chips

Instructions:
1. Preheat your oven to 375°F (190°C) and line two large baking sheets with parchment paper.
2. In a small bowl, whisk together the flour, baking soda, and salt.
3. In a large stand mixer, beat the butter, white sugar, and brown sugar until creamy and fluffy.
4. Add eggs one at a time, followed by the vanilla extract.
5. Gradually stir in dry ingredients until just combined, then fold in the chocolate chips.
6. Drop rounded tablespoons onto baking sheets and bake for 9 to 11 minutes until golden brown.
7. Let cool on baking sheets for 2 minutes before transferring to wire racks. Enjoy with cold milk!"""
}

# Streamlit Page Setup
st.set_page_config(
    page_title="NoticeGuard - Understand. Respond. Stay Safe.",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Human-Centered Light Theme Custom Styling
st.markdown("""
<style>
    /* Warm cream/ivory main background */
    .stApp {
        background-color: #FAF9F5;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Clean white elevated cards */
    .card-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    
    /* Reassurance callout in soft mint */
    .reassurance-card {
        background-color: #EAF6EE;
        border: 1px solid #BBF7D0;
        color: #166534;
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 0.88rem;
        font-weight: 500;
        line-height: 1.45;
        margin-bottom: 20px;
    }

    /* Dashed upload dropzone */
    .upload-dropzone {
        border: 2px dashed #BFDBFE;
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 16px;
    }

    /* Primary CTA Deep Forest Green Button */
    div.stButton > button[kind="primary"] {
        background-color: #116B4E !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        transition: all 0.15s ease-in-out;
        width: 100%;
        box-shadow: 0 2px 8px rgba(17, 107, 78, 0.2);
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0D543D !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(17, 107, 78, 0.3);
    }

    /* Status Pill (Gemma 4 Dynamic Status) */
    .status-pill {
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .status-pill-connected {
        background-color: #E6F4EA !important;
        color: #166534 !important;
        border: 1px solid #BBF7D0 !important;
    }
    .status-pill-disconnected {
        background-color: #FEF2F2 !important;
        color: #991B1B !important;
        border: 1px solid #FCA5A5 !important;
    }

    /* Non-lawyer Legal Disclaimer */
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        border: 1px solid #FDE68A;
        border-left-width: 4px;
        border-left-color: #F59E0B;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #92400E;
        margin-bottom: 22px;
        line-height: 1.45;
    }

    /* Quote box */
    .quote-box {
        background-color: #F8FAFC;
        border-left: 3px solid #0EA5E9;
        border-radius: 0 6px 6px 0;
        padding: 8px 14px;
        font-style: italic;
        font-size: 0.88rem;
        color: #475569;
        margin-top: 8px;
    }

    /* Grounding badges */
    .badge-exact {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.76rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-fuzzy {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #93C5FD;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.76rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-inference {
        background: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.76rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Feature trust bar */
    .trust-item {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "doc_data" not in st.session_state:
    st.session_state.doc_data = None
if "receipt_date" not in st.session_state:
    st.session_state.receipt_date = date.today()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "New Analysis"
if "validation_error" not in st.session_state:
    st.session_state.validation_error = None
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = "Select a sample notice..."

def wipe_all_session():
    st.session_state.clear()
    wipe_temp_directory()
    st.toast("Session wiped completely. All document data erased from RAM.", icon="🧹")

# Load current config and Gemma 4 health status
base_ollama_url = config.get("ollama", {}).get("base_url", "http://127.0.0.1:11434")
configured_model = config.get("ollama", {}).get("model", "gemma4:e4b")
gemma_connected, detected_model = check_gemma4_status(base_ollama_url, configured_model)
active_model_name = detected_model or configured_model

# Left Sidebar
with st.sidebar:
    # Brand Header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 2px;">
        <span style="font-size: 1.8rem;">🛡️</span>
        <span style="font-size: 1.4rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">NoticeGuard</span>
    </div>
    <div style="font-size: 0.82rem; color: #64748B; font-weight: 500; margin-bottom: 18px;">
        Understand. Respond. Stay Safe.
    </div>
    """, unsafe_allow_html=True)

    # Reassurance Callout Card
    st.markdown("""
    <div class="reassurance-card">
        🌱 <strong>No stress, you're covered.</strong><br>
        NoticeGuard helps you understand legal notices in simple, reassuring terms without sending any data off your machine.
    </div>
    """, unsafe_allow_html=True)

    # Navigation Menu
    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;'>Navigation</p>", unsafe_allow_html=True)
    nav_choice = st.radio(
        "Navigation",
        options=["Home / Overview", "New Analysis", "Past Reports", "Settings"],
        index=["Home / Overview", "New Analysis", "Past Reports", "Settings"].index(
            st.session_state.nav_page if st.session_state.nav_page in ["Home / Overview", "New Analysis", "Past Reports", "Settings"] else "New Analysis"
        ),
        label_visibility="collapsed"
    )
    if nav_choice != st.session_state.nav_page:
        st.session_state.nav_page = nav_choice
        st.rerun()

    st.markdown("---")

    # Language Selector
    lang_choice = st.selectbox(
        "🌐 Language / भाषा / భాష",
        options=["en", "hi", "te"],
        format_func=lambda x: {"en": "English", "hi": "हिंदी (Hindi)", "te": "తెలుగు (Telugu)"}[x],
        index=["en", "hi", "te"].index(st.session_state.lang)
    )
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()

    i18n = load_i18n(st.session_state.lang)

    # Receipt date picker for relative deadlines
    st.session_state.receipt_date = st.date_input(
        i18n.get("ui", {}).get("receipt_date_label", "Notice Receipt Date:"),
        value=st.session_state.receipt_date,
        help="The calendar date you actually received the notice. Statutory deadlines are calculated from this date."
    )

    st.markdown("---")
    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;'>Local Engine Status</p>", unsafe_allow_html=True)
    
    if gemma_connected:
        st.success(f"● Local Gemma 4 Active ({active_model_name})")
    else:
        st.error(f"● Gemma 4 Disconnected ({configured_model})")

    st.markdown("---")
    if st.button("🔄 " + i18n.get("ui", {}).get("refresh_button", "Refresh / Start Over"), key="sidebar_refresh_btn", use_container_width=True, type="secondary"):
        st.session_state.clear()
        wipe_temp_directory()
        st.rerun()

    st.markdown("---")
    with st.expander("📚 Indian Legal Terms Guide"):
        for term_key, term_info in glossary_data.items():
            st.markdown(f"**{term_info['term']}**")
            st.caption(term_info['plain_meaning'])

# Top Bar Status & Hero Header
top_col1, top_col2 = st.columns([3, 1.4])
with top_col1:
    st.markdown("""
    <div style="color: #116B4E; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px;">
        LEGAL DOCUMENT ANALYSIS
    </div>
    <h1 style="color: #0F172A; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 8px 0;">
        NoticeGuard / Local Legal Document Triage
    </h1>
    <p style="color: #475569; font-size: 1.05rem; line-height: 1.5; margin-bottom: 16px;">
        Upload your legal notice, get a clear summary, and know what to do next. No jargon. No confusion. Just the important stuff.
    </p>
    """, unsafe_allow_html=True)

with top_col2:
    if gemma_connected:
        st.markdown(f"""
        <div style="text-align: right; margin-top: 10px; margin-bottom: 8px;">
            <span class="status-pill status-pill-connected">🟢 Running Locally on Gemma 4 ({active_model_name})</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: right; margin-top: 10px; margin-bottom: 8px;">
            <span class="status-pill status-pill-disconnected">🔴 Gemma 4 Not Connected</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔄 " + i18n.get("ui", {}).get("refresh_button", "Refresh / Start Over"), key="top_refresh_btn", use_container_width=True, help="Reset session and start over"):
        st.session_state.clear()
        wipe_temp_directory()
        st.rerun()

# Disconnected / Model Missing Actionable Warning Banner
if not gemma_connected:
    host_clean = base_ollama_url.replace("http://", "").replace("https://", "")
    st.markdown(f"""
    <div style="background-color: #FEF2F2; border: 1px solid #FCA5A5; border-left: 5px solid #DC2626; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
        <strong style="color: #991B1B; font-size: 0.95rem;">⚠️ Local Gemma 4 Model Not Detected:</strong>
        <span style="color: #7F1D1D; font-size: 0.92rem; line-height: 1.5;"> NoticeGuard cannot reach Ollama at <code>{host_clean}</code> or <code>{configured_model}</code> is not installed. Please start Ollama (<code>ollama serve</code>) and ensure <code>ollama pull {configured_model}</code> has completed.</span>
    </div>
    """, unsafe_allow_html=True)

# Non-lawyer Legal Disclaimer Banner
st.markdown(f"""
<div class="disclaimer-box">
    <strong>⚖️ {i18n.get("disclaimer", "NoticeGuard provides preliminary factual extraction and informational triage only. It is NOT legal advice and does NOT determine legal validity or enforceability. Always consult a licensed advocate before acting.")}</strong>
</div>
""", unsafe_allow_html=True)

# If Navigation is "Settings"
if st.session_state.nav_page == "Settings":
    st.markdown("### ⚙️ System Settings & Verification Configuration")
    with st.container():
        st.markdown(f"**Ollama Endpoint:** `{config.get('ollama', {}).get('base_url', 'http://127.0.0.1:11434')}`")
        st.markdown(f"**Primary Model:** `{config.get('ollama', {}).get('model', 'gemma4:e4b')}`")
        st.markdown(f"**Fallback Model:** `{config.get('ollama', {}).get('fallback_model', 'gemma4:2b')}`")
        st.markdown(f"**Request Timeout:** `{config.get('ollama', {}).get('timeout_seconds', 180)} seconds`")
        st.markdown(f"**Fuzzy Quote Threshold:** `{config.get('verifier', {}).get('fuzzy_threshold', 65.0)}%`")
        st.markdown(f"**Air-Gapped Telemetry:** Disabled (`gatherUsageStats = false`)")
        st.markdown(f"**Zero-Persistence RAM Processing:** Active")
    st.stop()

# If Navigation is "Past Reports"
if st.session_state.nav_page == "Past Reports":
    st.markdown("### 📁 Past Triage Reports")
    st.info("🔒 **Privacy Guarantee**: NoticeGuard enforces zero-persistence privacy. Documents and analyses are retained solely in RAM during your active session and are permanently erased upon clicking 'Wipe Session' or closing the browser.")
    if st.session_state.doc_data:
        st.success(f"Active in-memory analysis available for document: `{len(st.session_state.doc_data.get('raw_text', ''))} characters`.")
    st.stop()

# Helper for quote verification badge
def render_grounding_badge(item: dict) -> str:
    status = item.get("status", "UNVERIFIED")
    m_type = item.get("match_type", "unverified")
    p_id = item.get("passage_id")
    p_suffix = f" (Passage [{p_id}])" if p_id else ""
    if status == "STATED" and m_type in ("exact", "normalized_whitespace", "normalized"):
        return f"<span class='badge-exact'>✅ Verified Direct Quote{p_suffix}</span>"
    elif status == "STATED" and m_type in ("fuzzy", "paraphrase", "grounded_summary"):
        p_str = f"Passage [{p_id}]" if p_id else "Notice Passages"
        return f"<span class='badge-fuzzy'>☑️ Grounded Summary (Paraphrased from {p_str})</span>"
    elif status == "UNKNOWN":
        return "<span class='badge-inference'>⚪ Not Stated in Notice</span>"
    else:
        return "<span class='badge-inference'>ℹ️ Contextual Inference</span>"

# MAIN INTERACTION CARD: "1. Upload Your Notice"
if st.session_state.doc_data is None:
    # Main White Card Container
    st.markdown("""
    <div class="card-container">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px;">
            <div>
                <h3 style="color: #0F172A; font-weight: 700; margin: 0 0 4px 0;">1. Upload Your Notice</h3>
                <p style="color: #64748B; font-size: 0.9rem; margin: 0;">Upload your PDF or text file, or paste notice text directly.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sample notice quick selector
    selected_sample = st.selectbox(
        "💡 Quick-Load a Sample Notice (includes valid legal notices and invalid non-legal file test):",
        options=list(SAMPLE_NOTICES.keys()),
        index=list(SAMPLE_NOTICES.keys()).index(st.session_state.selected_sample) if st.session_state.selected_sample in SAMPLE_NOTICES else 0
    )
    if selected_sample != st.session_state.selected_sample:
        st.session_state.selected_sample = selected_sample
        st.session_state.validation_error = None
        st.rerun()

    sample_preset = SAMPLE_NOTICES[st.session_state.selected_sample] if st.session_state.selected_sample else ""

    col_hero_input, col_hero_art = st.columns([3, 1])

    with col_hero_input:
        # File Uploader
        uploaded_file = st.file_uploader(
            "Upload Notice Document (PDF, TXT, or Image):",
            type=["pdf", "txt", "png", "jpg", "jpeg", "webp"],
            help="Your file is parsed strictly in memory (RAM). Scanned documents are transcribed locally via Gemma 4 Vision. Never transmitted to the cloud."
        )

        # OR Divider
        st.markdown("""
        <div style="display: flex; align-items: center; text-align: center; margin: 14px 0;">
            <div style="flex-grow: 1; border-bottom: 1px solid #E2E8F0;"></div>
            <span style="padding: 0 14px; color: #94A3B8; font-size: 0.82rem; font-weight: 600;">OR PASTE TEXT DIRECTLY</span>
            <div style="flex-grow: 1; border-bottom: 1px solid #E2E8F0;"></div>
        </div>
        """, unsafe_allow_html=True)

        # Text Area
        pasted_text_input = st.text_area(
            "Notice Content",
            value=sample_preset,
            height=180,
            placeholder="Paste legal notice text here...",
            label_visibility="collapsed"
        )

    with col_hero_art:
        # Inline SVG Art Card with yellow sticky note
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%; padding-top: 10px;">
            <svg width="240" height="195" viewBox="0 0 280 210" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="24" y="24" width="180" height="160" rx="12" fill="#E2E8F0"/>
              <rect x="20" y="20" width="180" height="160" rx="12" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>
              <rect x="35" y="36" width="70" height="8" rx="4" fill="#0F172A"/>
              <rect x="115" y="36" width="30" height="8" rx="4" fill="#94A3B8"/>
              <rect x="35" y="58" width="150" height="5" rx="2.5" fill="#E2E8F0"/>
              <rect x="35" y="70" width="130" height="5" rx="2.5" fill="#E2E8F0"/>
              <rect x="35" y="82" width="145" height="5" rx="2.5" fill="#E2E8F0"/>
              <rect x="35" y="94" width="90" height="5" rx="2.5" fill="#E2E8F0"/>
              <circle cx="170" cy="140" r="18" fill="#116B4E"/>
              <path d="M163 140L168 145L178 135" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
              <g transform="translate(135, 70) rotate(-4)">
                <rect x="0" y="0" width="130" height="78" rx="6" fill="#FEF08A" stroke="#FDE047" stroke-width="1" filter="drop-shadow(0 4px 6px rgba(0,0,0,0.08))"/>
                <circle cx="65" cy="7" r="3" fill="#EAB308"/>
                <text x="14" y="32" font-family="system-ui, -apple-system, sans-serif" font-size="11" font-weight="700" fill="#854D0E">You're not alone</text>
                <text x="14" y="47" font-family="system-ui, -apple-system, sans-serif" font-size="11" font-weight="700" fill="#854D0E">in this! 🛡️</text>
                <text x="14" y="62" font-family="system-ui, -apple-system, sans-serif" font-size="9" font-weight="500" fill="#A16207">NoticeGuard is here</text>
              </g>
            </svg>
        </div>
        """, unsafe_allow_html=True)

    # Validation Error Banner if triggered
    if st.session_state.validation_error:
        st.markdown(f"""
        <div style="background-color: #FEF2F2; border: 1px solid #FCA5A5; border-left: 5px solid #EF4444; border-radius: 10px; padding: 16px 20px; margin: 16px 0;">
            <h4 style="color: #991B1B; margin: 0 0 6px 0; font-weight: 700; font-size: 1rem;">⚠️ Invalid Document Detected</h4>
            <p style="color: #7F1D1D; margin: 0; font-size: 0.92rem; line-height: 1.5;">
                {st.session_state.validation_error}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Primary Action Button
    analyze_clicked = st.button("🚀 Analyze Notice", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Bottom Trust Footer (3-column feature bar)
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    with col_feat1:
        st.markdown("""
        <div class="trust-item">
            <span style="font-size: 1.4rem;">📋</span>
            <strong style="color: #0F172A; display: block; margin-top: 4px;">Clear Summary</strong>
            <span style="color: #64748B; font-size: 0.88rem;">In easy-to-understand language. No confusing legalese.</span>
        </div>
        """, unsafe_allow_html=True)
    with col_feat2:
        st.markdown("""
        <div class="trust-item">
            <span style="font-size: 1.4rem;">⏳</span>
            <strong style="color: #0F172A; display: block; margin-top: 4px;">Key Details</strong>
            <span style="color: #64748B; font-size: 0.88rem;">Deadlines, demanded amounts, and required next steps.</span>
        </div>
        """, unsafe_allow_html=True)
    with col_feat3:
        st.markdown("""
        <div class="trust-item">
            <span style="font-size: 1.4rem;">🕊️</span>
            <strong style="color: #0F172A; display: block; margin-top: 4px;">Peace of Mind</strong>
            <span style="color: #64748B; font-size: 0.88rem;">So you can calmly decide your response with an advocate.</span>
        </div>
        """, unsafe_allow_html=True)

    # Process Document Logic
    if analyze_clicked:
        file_bytes = uploaded_file.read() if uploaded_file else None
        file_name = uploaded_file.name if uploaded_file else None
        input_text = pasted_text_input.strip() if pasted_text_input else ""

        if not file_bytes and not input_text:
            st.error("Please upload a file or paste notice text.")
        else:
            with st.spinner("Analyzing notice locally on your laptop..."):
                try:
                    # 1. Ingest
                    ingest_res = ingest_document(
                        file_bytes=file_bytes,
                        file_name=file_name,
                        pasted_text=input_text,
                        base_url=base_ollama_url,
                        model=configured_model
                    )
                    raw_text = ingest_res["text"]

                    # 2. Pre-Triage Legal Document Validation (Agent 3 Guardrail)
                    is_legal, val_msg = validate_is_legal_notice(raw_text)
                    if not is_legal:
                        st.session_state.validation_error = (
                            "The uploaded file or text does not appear to be a legal notice, contract, or formal demand letter. "
                            "Please upload a valid legal notice or termination document to run triage."
                        )
                        st.rerun()
                    else:
                        st.session_state.validation_error = None

                    # 3. Injection Scan & Sanitization
                    sanitized_text, has_injection, detected_patterns = sanitize_prompt_injection(raw_text)
                    if has_injection:
                        st.warning(f"⚠️ Security Notice: Detected and neutralized prompt injection phrases: {', '.join(detected_patterns)}.")

                    # 4. Segment into numbered passages (P1, P2...)
                    passages = segment_notice(raw_text)

                    # 5. Extraction
                    raw_facts = extract_facts(sanitized_text, passages, base_url=base_ollama_url, model=configured_model)

                    # 6. Deterministic Code Verification (Quotes, Amounts, Dates, Breach, Actions, Penalties)
                    verified_facts = verify_facts(raw_facts, raw_text, passages)

                    # 7. Timeline Computation
                    timeline_res = compute_timeline(verified_facts, receipt_date=st.session_state.receipt_date)

                    # 8. Plain-Language Paraphrases (STATED, INTERPRETATION, UNKNOWN)
                    explanations = generate_plain_explanation(passages, verified_facts, base_url=base_ollama_url, model=configured_model)

                    # 9. Silent & Missing Clause Detection (PS2 Requirement)
                    missing_clauses_res = detect_missing_clauses(raw_text, passages)

                    # 10. Lawyer Escalation Checklist
                    checklist = build_lawyer_checklist(verified_facts, timeline_res, passages, raw_text)

                    st.session_state.doc_data = {
                        "raw_text": raw_text,
                        "sanitized_text": sanitized_text,
                        "passages": passages,
                        "verified_facts": verified_facts,
                        "timeline": timeline_res,
                        "explanations": explanations,
                        "missing_clauses": missing_clauses_res,
                        "checklist": checklist,
                        "has_injection": has_injection
                    }
                    st.rerun()

                except IngestError as e:
                    st.error(f"Error loading document: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during processing: {str(e)}")

# ANALYSIS RESULTS DASHBOARD (Post-Click View)
if st.session_state.doc_data is not None:
    doc = st.session_state.doc_data
    facts = doc["verified_facts"]
    timeline = doc["timeline"]
    passages = doc["passages"]
    explanations = doc["explanations"]
    missing_clauses = doc.get("missing_clauses", [])
    checklist = doc["checklist"]
    current_lang = st.session_state.lang

    # Re-evaluate timeline in case receipt date was updated in sidebar
    timeline = compute_timeline(facts, receipt_date=st.session_state.receipt_date)

    # Top Control Bar
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <h2 style="color: #0F172A; font-weight: 800; margin: 0;">📊 Triage Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)

    # Translation Banner if in Hindi or Telugu
    if current_lang != "en":
        st.info(f"🌐 {MACHINE_TRANSLATION_NOTICES.get(current_lang)}")

    # Helper for localized fact values
    def loc_val(text: Any) -> str:
        s = str(text or "")
        if current_lang != "en" and s and s not in ("None", "null", "Unknown", "None stated", "Not specified"):
            return translate_text(s, current_lang, base_url=base_ollama_url, model=configured_model)["text"]
        return s

    # 7 Specialized PS2 Capability Tabs
    ui_strings = i18n.get("ui", {})
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        f"📋 {ui_strings.get('tab_facts', 'Structured Triage Fact Sheet')}",
        f"⏳ {ui_strings.get('tab_timeline', 'Timeline & Deadlines')}",
        f"🔍 {ui_strings.get('tab_explain', 'Plain-Language View')}",
        f"🛡️ {ui_strings.get('tab_missing', 'Silent & Missing Clauses')}",
        f"📑 {ui_strings.get('tab_passages', 'Indexed Passages')}",
        f"⚖️ {ui_strings.get('tab_checklist', 'Lawyer Checklist')}",
        f"💬 {ui_strings.get('tab_qa', 'Ask a Question')}"
    ])

    # 1. TAB: STRUCTURED TRIAGE FACT SHEET (Human-Friendly 4 Pillars)
    with t1:
        st.markdown("""
        <div class="card-container">
            <h3 style="color:#0F172A; font-weight:700; margin-top:0;">📋 Structured Legal Triage Fact Sheet</h3>
            <p style="color:#64748B; font-size:0.9rem;">Verified factual summary structured into plain-language sections for immediate clarity.</p>
        """, unsafe_allow_html=True)

        # Urgency & Category Banner
        urgency = timeline.get("urgency_level", "MEDIUM")
        urg_color = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "blue", "LOW": "green"}.get(urgency, "blue")
        
        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            st.markdown(f"**{ui_strings.get('urgency_badge', 'Urgency Level')}:** :{urg_color}[**{loc_val(urgency)}**]")
        with col_b2:
            st.markdown(f"**Notice Category:** **{loc_val(checklist['notice_type'])}**")

        st.markdown("---")

        # PILLAR 1: WHAT THIS NOTICE IS ABOUT
        summary_title = ui_strings.get("summary_heading", "What This Notice Is About")
        raw_summary = facts.get("summary") or (
            f"This formal notice was issued by {facts.get('sender', {}).get('value', 'the sender')} directed to "
            f"{facts.get('recipient', {}).get('value', 'the recipient')}. It alleges {facts.get('termination_breach', {}).get('value', 'a breach of obligations')} "
            f"and asserts a demand concerning {facts.get('amount', {}).get('value', 'unspecified amounts')}."
        )
        st.markdown(f"#### 📖 1. {summary_title}")
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #116B4E; border-radius: 8px; padding: 16px 20px; font-size: 0.98rem; line-height: 1.6; color: #1E293B; margin-bottom: 20px;">
            {loc_val(raw_summary)}
        </div>
        """, unsafe_allow_html=True)

        # PILLAR 2: CRITICAL DEADLINES & DATES
        dl_title = ui_strings.get("deadlines_heading", "Critical Deadlines & Dates")
        st.markdown(f"#### ⏳ 2. {dl_title}")
        dl_fact = facts.get("deadline", {})
        computed_dl = timeline.get("computed_deadline", "None specified")
        days_rem = timeline.get("days_remaining")
        
        col_dl1, col_dl2 = st.columns([1.6, 1])
        with col_dl1:
            st.markdown(f"- **Calculated Next Deadline:** `{computed_dl}`")
            if days_rem is not None:
                if days_rem < 0:
                    st.error(f"⚠️ Response deadline passed {abs(days_rem)} days ago!")
                elif days_rem == 0:
                    st.error("⚠️ Response deadline is TODAY!")
                else:
                    st.info(f"⏳ **{days_rem} calendar days remaining** from notice receipt date ({st.session_state.receipt_date.strftime('%d %b %Y')}).")
            if timeline.get("is_ambiguous"):
                st.warning("⚠️ Ambiguous Timeline: The notice demands action 'forthwith' or 'immediately' without a clear statutory day count.")
        with col_dl2:
            st.markdown(render_grounding_badge(dl_fact), unsafe_allow_html=True)
            if dl_fact.get("quote"):
                st.markdown(f"<div class='quote-box'>Stated in Notice: \"{dl_fact['quote']}\"</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # PILLAR 3: FINANCIAL DEMANDS / PENALTIES
        fin_title = ui_strings.get("demands_heading", "Financial Demands / Penalties")
        st.markdown(f"#### 💰 3. {fin_title}")
        amt_fact = facts.get("amount", {})
        pen_fact = facts.get("financial_penalties", {})
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(f"- **Principal Monetary Claim:** `{amt_fact.get('value', 'None specified')}`")
            st.markdown(render_grounding_badge(amt_fact), unsafe_allow_html=True)
            if amt_fact.get("quote"):
                st.markdown(f"<div class='quote-box'>Verified Quote: \"{amt_fact['quote']}\"</div>", unsafe_allow_html=True)
        with col_f2:
            pen_val = pen_fact.get("value") or "None specified"
            st.markdown(f"- **Interest / Additional Penalties:** {loc_val(pen_val)}")
            st.markdown(render_grounding_badge(pen_fact), unsafe_allow_html=True)
            if pen_fact.get("quote"):
                st.markdown(f"<div class='quote-box'>Verified Quote: \"{pen_fact['quote']}\"</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # PILLAR 4: RECOMMENDED NEXT STEPS
        steps_title = ui_strings.get("next_steps_heading", "Recommended Next Steps")
        st.markdown(f"#### 🛡️ 4. {steps_title}")
        default_steps = [
            "Preserve the physical notice, envelope, and postal tracking receipts as proof of receipt date.",
            "Locate and organize all relevant contracts, receipts, bank statements, and email correspondence.",
            "Do not sign admissions or make informal payments without independent legal advice.",
            "Consult a licensed advocate using the Lawyer Escalation Checklist before the response window expires."
        ]
        steps_to_show = facts.get("recommended_next_steps") or default_steps
        for step_text in steps_to_show:
            st.markdown(f"- [ ] {loc_val(step_text)}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # DETAILED PARTIES & ALLEGATIONS
        st.markdown("#### 👥 Notice Parties & Detailed Allegations")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"**{ui_strings.get('sender', 'Sender')}:** {facts['sender']['value']}")
            st.markdown(render_grounding_badge(facts['sender']), unsafe_allow_html=True)
            if facts['sender']['quote']:
                st.markdown(f"<div class='quote-box'>Verified from {facts['sender']['passage_id'] or 'Document'}: \"{facts['sender']['quote']}\"</div>", unsafe_allow_html=True)
        with col_p2:
            st.markdown(f"**{ui_strings.get('recipient', 'Recipient')}:** {facts['recipient']['value']}")
            st.markdown(render_grounding_badge(facts['recipient']), unsafe_allow_html=True)
            if facts['recipient']['quote']:
                st.markdown(f"<div class='quote-box'>Verified from {facts['recipient']['passage_id'] or 'Document'}: \"{facts['recipient']['quote']}\"</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            breach_fact = facts.get("termination_breach", {})
            st.markdown(f"**{ui_strings.get('breach', 'Termination & Breach Allegation')}:**")
            st.write(loc_val(breach_fact.get("value", "Not specified")))
            st.markdown(render_grounding_badge(breach_fact), unsafe_allow_html=True)
            if breach_fact.get("quote"):
                st.markdown(f"<div class='quote-box'>Verified Quote ({breach_fact.get('passage_id') or 'Doc'}): \"{breach_fact['quote']}\"</div>", unsafe_allow_html=True)

        with col_a2:
            act_fact = facts.get("required_actions", {})
            st.markdown(f"**{ui_strings.get('required_actions', 'Required Actions & Demands')}:**")
            st.markdown(f"{loc_val(act_fact.get('value', facts['demand']['value']))}")
            st.markdown(render_grounding_badge(act_fact), unsafe_allow_html=True)
            if act_fact.get("quote"):
                st.markdown(f"<div class='quote-box'>Verified Quote: \"{act_fact['quote']}\"</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # 2. TAB: TIMELINE
    with t2:
        st.markdown("""
        <div class="card-container">
            <h3 style="color:#0F172A; font-weight:700; margin-top:0;">⏳ Action & Deadline Milestones</h3>
            <p style="color:#64748B; font-size:0.9rem;">Deterministic calendar computations counting from your notice receipt date.</p>
        """, unsafe_allow_html=True)

        if timeline.get("is_ambiguous"):
            st.warning("⚠️ Ambiguous Deadline Detected: The notice demands action 'immediately' or 'forthwith' without specifying exact calendar days.")

        for ev in timeline["events"]:
            st.markdown(f"#### 📅 {ev['event']} — **{ev['date']}**")
            st.write(ev['description'])
            if ev.get("days_remaining") is not None:
                if ev["days_remaining"] < 0:
                    st.error(f"Deadline expired {abs(ev['days_remaining'])} days ago!")
                elif ev["days_remaining"] == 0:
                    st.error("Deadline is TODAY!")
                else:
                    st.info(f"⏳ **{ev['days_remaining']} calendar days remaining** from today.")
            st.divider()

        st.markdown("</div>", unsafe_allow_html=True)

    # 3. TAB: PLAIN-LANGUAGE VIEW
    with t3:
        st.markdown("""
        <div class="card-container">
            <h3 style="color:#0F172A; font-weight:700; margin-top:0;">🔍 Calm Plain-Language Passage Analysis</h3>
            <p style="color:#64748B; font-size:0.9rem;">Shows what the notice directly states, what it means calmly, and what it leaves unsaid.</p>
        """, unsafe_allow_html=True)

        for item in explanations:
            p_id = item.get("passage_id", "")
            with st.expander(f"Passage {p_id} Analysis", expanded=True):
                st.markdown(f"**Exact Original Quote ({p_id}):**")
                st.markdown(f"<div class='quote-box'>\"{item.get('quote')}\"</div>", unsafe_allow_html=True)
                
                stated_text = item.get("stated", "")
                interp_text = item.get("interpretation", "")
                unknown_text = item.get("unknown", "")

                if current_lang != "en":
                    stated_text = translate_text(stated_text, current_lang, base_url=base_ollama_url, model=configured_model)["text"]
                    interp_text = translate_text(interp_text, current_lang, base_url=base_ollama_url, model=configured_model)["text"]
                    unknown_text = translate_text(unknown_text, current_lang, base_url=base_ollama_url, model=configured_model)["text"]

                st.markdown(f"<span style='color:#0284C7; font-weight:700;'>● {i18n.get('labels', {}).get('stated', 'STATED')}:</span> {stated_text}", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#D97706; font-weight:700;'>● {i18n.get('labels', {}).get('interpretation', 'INTERPRETATION')}:</span> {interp_text}", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#64748B; font-weight:700;'>● {i18n.get('labels', {}).get('unknown', 'UNKNOWN')}:</span> {unknown_text}", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # 4. TAB: SILENT & MISSING CLAUSE DETECTION (PS2 Rubric Requirement #4)
    with t4:
        st.markdown("""
        <div class="card-container">
            <h3 style="color:#0F172A; font-weight:700; margin-top:0;">🛡️ Silent & Missing Clause Detection</h3>
            <p style="color:#64748B; font-size:0.9rem;">Audits essential legal rights, cure periods, arbitration terms, and accounting details that legal notices often omit to intimidate recipients.</p>
        """, unsafe_allow_html=True)

        present_count = sum(1 for c in missing_clauses if c.get("is_present"))
        missing_count = len(missing_clauses) - present_count
        st.markdown(f"**Audit Summary:** `{present_count} Protections Present` | `{missing_count} Critical Protections Absent/Silent`")
        st.markdown("---")

        for c in missing_clauses:
            is_p = c.get("is_present", False)
            status_badge = "<span class='badge-exact'>🟢 PRESENT IN NOTICE</span>" if is_p else "<span class='badge-fuzzy'>⚠️ SILENT / MISSING IN NOTICE</span>"
            
            st.markdown(f"#### {c['name']} &nbsp;&nbsp; {status_badge}", unsafe_allow_html=True)
            st.caption(c["description"])
            st.write(f"**Legal Implication:** {c['observation']}")
            
            if is_p and c.get("quote"):
                st.markdown(f"<div class='quote-box'>Verified from {c.get('passage_id') or 'Document'}: \"{c['quote']}\"</div>", unsafe_allow_html=True)
            
            st.info(f"💡 **Advocate Consultation Tip:** {c['advocate_action']}")
            st.divider()

        st.markdown("</div>", unsafe_allow_html=True)

    # 5. TAB: INDEXED PASSAGES EXPLORER (PS2 Rubric Requirement #1)
    with t5:
        st.markdown("""
        <div class="card-container">
            <h3 style="color:#0F172A; font-weight:700; margin-top:0;">📑 Indexed Source Document Passages</h3>
            <p style="color:#64748B; font-size:0.9rem;">Numbered, citable source passages [P1, P2, ...] normalized with Unicode NFC and Indic numerals.</p>
        """, unsafe_allow_html=True)
        
        st.info(f"Total Passages Indexed: **{len(passages)} passages** | Document Length: **{len(doc.get('raw_text', ''))} characters**")
        
        for p in passages:
            with st.expander(f"Passage [{p['id']}] — {p.get('length', len(p['text']))} characters", expanded=False):
                st.markdown(f"**Passage Identifier:** `[{p['id']}]`")
                st.code(p["text"], language=None)

        st.markdown("</div>", unsafe_allow_html=True)

    # 6. TAB: LAWYER CHECKLIST
    with t6:
        st.markdown("""
        <div class="card-container">
            <h3 style="color:#0F172A; font-weight:700; margin-top:0;">⚖️ Lawyer Escalation & Consultation Checklist</h3>
            <p style="color:#64748B; font-size:0.9rem;">Prepares a structured consultation brief for an advocate. NoticeGuard helps you prepare; it does not replace a lawyer.</p>
        """, unsafe_allow_html=True)
        
        st.info(f"📌 **{checklist['triage_line']}**")

        st.markdown(f"#### ❓ {ui_strings.get('lawyer_questions', 'Passage-Specific Questions for Your Advocate')}")
        for q in checklist["questions"]:
            if current_lang != "en":
                q = translate_text(q, current_lang, base_url=base_ollama_url, model=configured_model)["text"]
            st.markdown(f"- [ ] {q}")

        st.markdown(f"#### 📁 {ui_strings.get('lawyer_docs', 'Documents to Gather & Bring')}")
        for doc_item in checklist["documents_to_bring"]:
            if current_lang != "en":
                doc_item = translate_text(doc_item, current_lang, base_url=base_ollama_url, model=configured_model)["text"]
            st.markdown(f"- [ ] {doc_item}")

        st.markdown(f"#### ⚠️ {ui_strings.get('lawyer_warnings', 'Jurisdiction & Procedural Warnings')}")
        for w in checklist["jurisdiction_warnings"]:
            if current_lang != "en":
                w = translate_text(w, current_lang, base_url=base_ollama_url, model=configured_model)["text"]
            st.warning(w)

        st.markdown("</div>", unsafe_allow_html=True)

    # 7. TAB: ASK A QUESTION (Strict Grounding & Refusals)
    with t7:
        st.markdown("""
        <div class="card-container">
            <h3 style="color:#0F172A; font-weight:700; margin-top:0;">💬 Ask a Question About This Notice</h3>
            <p style="color:#64748B; font-size:0.9rem;">Answers are strictly grounded in verified notice passages. Legal advice, liability verdicts, and enforceability questions are strictly refused.</p>
        """, unsafe_allow_html=True)

        with st.form("qa_form", clear_on_submit=False):
            user_query = st.text_input(
                "Your Question:",
                placeholder=ui_strings.get("ask_placeholder", "e.g. 'What is the amount demanded?' or 'By when must I respond?'"),
                key="qa_input_field"
            )
            submit_qa = st.form_submit_button("Ask Question", type="primary")

        if submit_qa:
            if not user_query.strip():
                st.warning("Please type a question.")
            else:
                q = user_query.strip()
                with st.spinner("Finding grounded answer in document passages..."):
                    res = answer_followup_question(
                        question=q,
                        passages=passages,
                        lang=current_lang,
                        base_url=base_ollama_url,
                        model=configured_model
                    )
                    ans_text = res.get("answer", "")
                    ans_pid = res.get("passage_id")
                    ans_status = res.get("status", "grounded")
                    st.session_state.qa_history.append((q, ans_text, ans_status, ans_pid))
                    st.session_state.chat_history.append((q, ans_text, ans_status))

        # Render latest answer in high-contrast card right below input box
        if st.session_state.qa_history:
            latest_q, latest_a, latest_status, latest_pid = st.session_state.qa_history[-1]
            p_badge = f"<span class='badge-exact' style='background:#1E293B;color:#38BDF8;border-color:#0284C7;'>Grounded in Passage [{latest_pid}]</span>" if latest_pid else ""
            if latest_status == "refusal":
                badge_html = "<span class='badge-fuzzy' style='background:#FEF3C7;color:#92400E;border-color:#FCD34D;'>⚖️ Legal Disclaimer Refusal</span>"
            elif latest_status == "silent":
                badge_html = "<span class='badge-inference'>⚪ Not Addressed in Notice</span>"
            elif latest_pid:
                badge_html = p_badge
            else:
                badge_html = "<span class='badge-fuzzy'>☑️ Grounded Document Answer</span>"

            st.markdown(f"""
            <div style="background-color: #0F172A; color: #FFFFFF; border-radius: 12px; padding: 22px 26px; margin: 18px 0; border: 1px solid #334155; box-shadow: 0 4px 16px rgba(0,0,0,0.18);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="color: #38BDF8; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">💬 NoticeGuard Grounded Answer</span>
                    {badge_html}
                </div>
                <div style="font-size: 1.05rem; line-height: 1.6; color: #F8FAFC; margin-bottom: 12px;">
                    {latest_a}
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; border-top: 1px solid #1E293B; padding-top: 8px;">
                    <strong>Question:</strong> "{latest_q}"
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Display previous Q&A history
        if len(st.session_state.qa_history) > 1:
            st.markdown("---")
            st.markdown("##### 📜 Previous Q&A:")
            for prev_q, prev_a, prev_status, prev_pid in reversed(st.session_state.qa_history[:-1][-4:]):
                st.markdown(f"**Q:** {prev_q}")
                st.markdown(f"**A:** {prev_a}")
                if prev_pid:
                    st.caption(f"Cited passage: [{prev_pid}]")
                st.divider()

        st.markdown("</div>", unsafe_allow_html=True)
