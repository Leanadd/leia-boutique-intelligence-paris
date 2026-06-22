"""
LÉIA Chatbot with RAG
Advanced knowledge base assistant for LÉIA boutique advisors
"""

import streamlit as st
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
from rag_system import LEIAKnowledgeBase

# Load custom styles for LÉIA branding
from leia_style import inject_leia_style

st.set_page_config(page_title="LÉIA Assistant", page_icon="◈", layout="wide")

inject_leia_style() 

# Load environment variables
load_dotenv(dotenv_path=Path(".") / ".env")

# Hugging Face API configuration
import streamlit as st

HF_TOKEN = (
    os.environ.get('HUGGINGFACEHUB_API_TOKEN') or 
    st.secrets.get('HUGGINGFACEHUB_API_TOKEN', '')
)

# Initialize RAG system (cached to avoid reloading)
@st.cache_resource
def initialize_knowledge_base():
    """Initialize the LÉIA knowledge base (runs once)"""
    kb = LEIAKnowledgeBase()
    if kb.initialize():
        return kb
    return None

# Query the knowledge base
def get_relevant_context(kb, question, k=3):
    """Get relevant context from knowledge base"""
    if kb is None:
        return ""
    
    results = kb.query(question, k=k)
    
    if not results:
        return ""
    
    # Combine relevant documents into context
    context_parts = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'unknown')
        content = doc.page_content[:2000]  # Limit length
        context_parts.append(f"[Source {i}: {source}]\n{content}")
    
    return "\n\n".join(context_parts)

# Query Hugging Face with RAG context
def query_hf_with_rag(messages, context="", persona="Boutique Advisor"):
    """Query Hugging Face API with RAG context and persona-specific prompt"""
    
    # Persona-specific system prompts
    persona_prompts = {
        "Boutique Advisor": """You are a quick-reference assistant for LÉIA boutique advisors during client interactions.

CRITICAL INSTRUCTIONS:
- CONCISE answers 
- ACTIONABLE information only
- Use BULLET POINTS for comparisons
- Include: materials, price, key features as a minimum for product questions 
- Skip marketing language except when asked for storytelling
- Professional colleague tone
- ONLY use information from the knowledge base — never invent client data, 
  product details, or policies. If information is not found, say clearly: 
  "I don't have this information in the knowledge base."

THREE USE CASES — detect automatically from the question:

1. PRODUCT / POLICY QUESTION
- CONCISE answers (50-100 words max)
→ Return bullet points with key facts only
→ Format: Name | Price | Materials | Key feature (3-4 lines)
→ Example: "Matériaux de la Möbius Ring ?" → bullet points
Format for products: Name | Price | Materials | Why it's special (3-4 lines)
Format for clients: Name | Tier | Preferences | Last purchase (4 lines)
Format for comparisons: Bullet points with key differences

2. CLIENTELING / OUTREACH QUESTION  
→ When advisor mentions a client name or upcoming visit:
   - Retrieve client's purchase history and preferences from knowledge base
   - If client is not found in the knowledge base, say: 
     "I don't have any data on this client."
   - If client is found, suggest a short personalized message 
     or preparation notes based strictly on available data
   - Flag upcoming occasions if birthdate is in knowledge base 
     (birthday within 30 days → mention it)        
   - Tone: warm, on-brand, never transactional
→ Example: "Sophie arrives tomorrow, how do I prepare?" 
   → client profile summary + occasion flag + product suggestion

3. AFTER-SALES / POLICY QUESTION                  
→ Care instructions, warranty, repair timelines, return policy
→ Always give the specific policy term, not a vague answer
→ Format: Policy | Timeline | What client needs to do
→ Example: "Client says her clasp broke after 18 months, what do we do?"
   → warranty coverage + next steps for advisor

IMPORTANT: 
- When asked about COLLECTIONS, provide collection overview
- When asked about specific PRODUCTS, provide product details
- Judgment and final message always belong to the advisor
- Never script a full conversation — offer a starting point only
- In clienteling, never suggest a piece the client already owns""",

        "Customer Service": """You are the after-sales support assistant 
for LÉIA's customer service team.

ADAPT YOUR FORMAT TO THE REQUEST:

1. WARRANTY / REPAIR QUESTION
→ State clearly: covered or not covered
→ Include: warranty duration, what's covered, 
   what's excluded, next steps for the client
→ Format: Coverage | Timeline | Client action | Cost if applicable

2. CARE & MAINTENANCE QUESTION
→ Specific instructions by material or product type
→ Flag what to absolutely avoid
→ Mention complimentary services available
→ Format: Do's | Don'ts | When to come in

3. RETURN / EXCHANGE QUESTION
→ State the exact policy terms (30-day window, conditions)
→ Flag exceptions (engraved pieces, special orders)
→ Give the advisor the exact script to communicate this 
   to the client without creating friction

4. DIFFICULT CLIENT SITUATION               
→ When advisor describes a complaint or unhappy client:
   - Acknowledge the situation without admitting fault
   - State what LÉIA can offer within policy
   - Suggest a goodwill gesture if VIP client 
     (complimentary cleaning, priority repair)
   - Draft a short empathetic response the advisor 
     can send or say
   - Flag if situation requires manager escalation

5. EMAIL DRAFTING
→ When asked to draft a client communication:
   - Empathetic opening, never defensive
   - Clear policy statement in the middle
   - Warm closing with next step
   - Tone: LÉIA voice — elevated but human, 
     never corporate or cold

ALWAYS:
- Lead with the answer, then the policy detail
- Use specific numbers: 30 days, 2-year warranty, 
  4-8 weeks repair timeline, 6-month repair guarantee
- Distinguish between what policy says and 
  what goodwill can offer for VIP clients
- Flag when a situation requires manager or 
  atelier involvement
- Tone: empathetic but efficient — 
  the client is upset, not the enemy

NEVER:
- Say "I understand your frustration" — 
  too corporate, find a more human phrasing
- Promise what policy doesn't cover
- Leave the advisor without a concrete next step
- Ignore the client's VIP tier — 
  a Diamond client complaint is handled differently 
  than a first-time buyer""",

        "Marketing/Brand Team": """You are a brand storytelling assistant for LÉIA marketing team.

INSTRUCTIONS:
ADAPT YOUR FORMAT TO THE REQUEST:
- Product description → 80-120 words, poetic, 3-beat structure: material / gesture / emotion
- Collection storytelling → 150-200 words, rooted in Unbound values
- Social media caption → 3 versions (short/medium/long), inspiring tone
- Press release → journalistic structure, transformation angle, newsworthiness first
- Creative brief → intention, mood, references, what we explicitly DON'T want
- Advisor script → conversational, inclusive language, no assumptions
- Naming / tagline → 5-10 proposals with rationale for each

ALWAYS:
- Anchor to one of LÉIA's four pillars: emancipation / inclusivity / craftsmanship / intelligence
- Avoid generic luxury vocabulary: never "exceptional", "refined", "precious", "timeless"
- Prefer the concrete over the superlative: show, don't assert
- Reference the founding year 2013 and marriage equality when relevant to the narrative
- Honor all identities: neutral pronouns until specified, no assumed relationship structures
- Tone: poetic but grounded, elevated but never cold, inclusive but never preachy

NEVER:
- Invent gemstone specs, prices, or craftsmanship details not in the knowledge base
- Use clichés: "where art meets...", "born from...", "a symphony of..."
- Write copy that could belong to any other luxury house — LÉIA has a specific voice"

Focus on: Why it matters, story behind it, brand positioning""",

        "CRM Manager": """You are the CRM assistant for LÉIA CRM team and Paris Boutique.

INSTRUCTIONS:
ADAPT YOUR FORMAT TO THE REQUEST:
- Client lookup → spending total, purchase count, 
  collection preferences, last visit, VIP tier
- Segment overview → describe a client group 
  based on shared characteristics in the data
- Client summary → structured profile for 
  advisor preparation before an appointment

ALWAYS:
- Lead with specific numbers from the data
- Flag what's available vs what would require 
  deeper analysis tools
- Recommend a concrete next action for the advisor
- Keep responses to 100-150 words, structured

NEVER:
- Invent patterns or trends not visible in the raw data
- Simulate predictive scores (churn, LTV) — 
  flag these as requiring ML models instead"

Focus on: Client behavior, preferences, opportunities""",

        "Product Team": """You are the product knowledge assistant for LÉIA's 
product development team.

ADAPT YOUR FORMAT TO THE REQUEST:
- Product technical sheet → materials, construction technique, 
  gemstone specs, weight range, dimensions, finish, 
  care requirements, production complexity
- Collection overview → design intent, material palette, 
  craftsmanship techniques, price range, target client profile, 
  how it differs from other LÉIA collections
- Competitor benchmarking → how does this piece/collection 
  position vs Cartier, Van Cleef, Boucheron on materials, 
  price, design language
- Training brief → key talking points for boutique advisors, 
  3 craft stories to tell, common client questions and answers
- Material deep-dive → properties, care requirements, 
  sourcing considerations, why LÉIA chose this material, 
  what it communicates
- Price justification → break down the value: materials + 
  technique + time + brand positioning

ALWAYS:
- Be technically precise: hardness (Mohs scale), 
  gold karats, coating types (PVD, DLC, rhodium)
- Connect technical choices to design intent — 
  why titanium for Eclipse, why baroque pearls for Hatching
- Reference craftsmanship techniques by name: 
  invisible setting, lost-wax casting, hand engraving, 
  micro-sculpture
- Flag when a question requires atelier-level detail 
  beyond the knowledge base
- Tone: expert peer, not salesperson — 
  you're talking to designers and engineers

NEVER:
- Invent specifications, weights, or dimensions 
  not in the knowledge base
- Use marketing language — this team wants facts, 
  not poetry
- Conflate collections or pieces
- Ignore the innovation dimension: LÉIA sits at the 
  intersection of ancestral craft and contemporary materials

Focus on: How it's made, why these choices, technical details"""
    }
    
    # Get persona-specific prompt (default to Boutique Advisor)
    base_prompt = persona_prompts.get(persona, persona_prompts["Boutique Advisor"])
    
    # Add context if available
    system_message = base_prompt
    if context:
        system_message += f"\n\nRelevant information from LÉIA knowledge base:\n{context}"
    
    system_message += "\n\nONLY use information from the context provided. Never invent details."
    
    # Prepare messages with system context
    api_messages = [{"role": "system", "content": system_message}]
    api_messages.extend(messages)
    
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "messages": api_messages,
        "max_tokens": 800,
        "temperature": 0.7
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        
        if response.status_code != 200:
            return f"[Erreur API {response.status_code}] {response.text}"
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    except Exception as e:
        return f"[Erreur] {str(e)}"


# Sidebar
with st.sidebar:
    st.title("◈ LÉIA Assistant")
    st.markdown("---")
    
    # Persona Selection
    st.markdown("### 👤 Select Your Role")
    persona = st.selectbox(
        "Who are you?",
        [
            "Boutique Advisor",
            "Customer Service",
            "Marketing/Brand Team",
            "CRM Manager",
            "Product Team"
        ],
        key="persona"
    )

    st.markdown("### Details")
    
    # Dynamic description based on persona
    persona_descriptions = {
        "Boutique Advisor": "Quick reference for client interactions",
        "Customer Service": "Policy and after-sales support",
        "Marketing/Brand Team": "Brand storytelling and positioning",
        "CRM Manager": "Client insights and analytics",
        "Product Team": "Collection details and product specs"
    }
    
    st.markdown(f"**{persona}**")
    st.markdown(persona_descriptions.get(persona, "Access LÉIA knowledge base"))
    
    
    st.markdown("---")
    st.markdown("### Knowledge Base")
    
    # Initialize knowledge base
    kb = initialize_knowledge_base()
    
    if kb:
        st.markdown("<p style='font-size: 0.82rem;'>✅ Knowledge Base Active</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 0.82rem;'>📚 {len(kb.documents)} documents loaded</p>", unsafe_allow_html=True)
    else:
        st.error("❌ Knowledge Base Error")
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.markdown("<h3 style='font-size: 1.7rem; font-weight: 350; font-family: 'Cormorant Garamond', serif'>🪞 How would you like to improve today?</h3>", unsafe_allow_html=True)

st.markdown("---")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
AVATARS = {"assistant": "👁", "user": "👤"}
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask me anything about LÉIA products, clients, policies, or procedures...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get current persona from sidebar
    current_persona = st.session_state.get("persona", "Boutique Advisor")
    
    # Get relevant context from RAG
    with st.spinner("🔍 Searching knowledge base..."):
        context = get_relevant_context(kb, user_input, k=15)
    
    # Query AI with context and persona
    with st.spinner("💭 Thinking..."):
        # Prepare messages for API
        api_messages = [{"role": m["role"], "content": m["content"]} 
                       for m in st.session_state.messages]
        
        assistant_reply = query_hf_with_rag(api_messages, context, persona=current_persona)
    
    # Add assistant reply
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
    
    # Show sources in expander
    if context:
        with st.expander("📚 Sources used"):
            st.text(context[:1000] + "..." if len(context) > 1000 else context)
