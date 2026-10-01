import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables from .env file
load_dotenv()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Enterprise Support Bot", page_icon="🏢", layout="wide")
st.title("🏢 Multi-Department AI Customer Support")
st.markdown("Powered by **AutoGen Multi-Agent Framework** & **Streamlit**")

# Validation check for OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OPENAI_API_KEY is missing from your .env file. Please add it and restart the app.")
    st.stop()

# --- Initialize Model Client ---
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=api_key,
)

# --- Define Department Agents with Elaborate Knowledge Bases ---
it_agent = AssistantAgent(
    name="IT_Agent",
    model_client=model_client,
    system_message=(
        "You are the IT Support Specialist for Enterprise Corp. Provide helpful, structured, multi-sentence answers.\n"
        "- Password Reset: Go to https://login.enterprise.internal/reset, enter your corporate employee ID, and follow the 6-digit OTP verification sent to your mobile device. If your account is locked out after 3 failed attempts, call the IT Helpdesk at extension #4357.\n"
        "- VPN Connectivity: Download the GlobalProtect client from the internal software center, connect to the gateway at vpn.enterprise.internal, and authenticate using your primary Single Sign-On (SSO) credentials.\n"
        "- Software & Hardware: To request new software licenses (like IDEs or design suites) or hardware peripherals (monitors, keyboards), submit a ticket via the IT ServiceNow portal under 'Asset Provisioning'. Requests take 2–3 business days for manager approval."
    )
)

hr_agent = AssistantAgent(
    name="HR_Agent",
    model_client=model_client,
    system_message=(
        "You are the Human Resources Specialist for Enterprise Corp. Provide helpful, structured, multi-sentence answers.\n"
        "- Leave Policy: Full-time employees accrue 24 annual paid leave days (2 per month) and 12 sick leave days. Casual or planned leave must be requested at least 3 days in advance through the HR Workday portal.\n"
        "- Payroll & Salary: Monthly salary statements and pay slips are processed by the last working day of every month and can be downloaded from the employee self-service portal under 'Payroll History'.\n"
        "- Benefits & Onboarding: Health insurance enrollment, dental coverage options, and tax declaration forms must be updated on the Darwinbox platform during your first week or during the annual open enrollment window in November."
    )
)

finance_agent = AssistantAgent(
    name="Finance_Agent",
    model_client=model_client,
    system_message=(
        "You are the Finance Department Specialist. Provide helpful, structured, multi-sentence answers.\n"
        "- Expense Claims: Corporate travel, client entertainment, and official purchases must be submitted via the Concur expense portal by the 25th of every month. Itemized digital PDF receipts are mandatory for any expense exceeding $25.\n"
        "- Vendor Invoices: Third-party vendors and external contractors should submit official invoices directly to billing@enterprise.internal with an attached purchase order (PO) number for verification.\n"
        "- Travel Per Diem: Standard daily food and lodging allowances during domestic and international business trips follow the tiered schedule outlined in Section 8 of the Concur corporate travel handbook."
    )
)

operations_agent = AssistantAgent(
    name="Operations_Agent",
    model_client=model_client,
    system_message=(
        "You are the Operations Specialist. Provide helpful, structured, multi-sentence answers.\n"
        "- Facility Management: Meeting rooms, conference spaces, and desk bookings can be scheduled using the Workplace Services mobile app. Report any office maintenance issues (lighting, HVAC) directly to facility helpdesk.\n"
        "- Supply Chain & Shipping: Internal equipment transfers, hardware shipping requests, and inventory tracking can be managed through the internal Ops portal dashboard.\n"
        "- Access Badges: Physical security identification badges must be worn at all times in the building. Report lost cards immediately to building security and request a replacement pass through the facilities portal."
    )
)

# --- Dedicated Intent Classifier Router ---
router_agent = AssistantAgent(
    name="Triage_Router",
    model_client=model_client,
    system_message=(
        "You are an expert intent classifier for enterprise support. "
        "Analyze the user query and output ONLY ONE exact agent name based on these precise rules:\n"
        "- IT_Agent: passwords, VPN, software licenses, hardware peripherals, computer access.\n"
        "- HR_Agent: leave policies, paid time off, salary slips, payroll statements, health insurance, employee onboarding.\n"
        "- Finance_Agent: expense claims, travel per diems, vendor invoices, financial reimbursements.\n"
        "- Operations_Agent: meeting room bookings, facility management, office access badges, equipment shipping/logistics.\n"
        "Output ONLY the agent name (e.g., HR_Agent) with no extra markdown, punctuation, or explanation."
    )
)

# Map names to actual agent objects
agent_map = {
    "IT_Agent": it_agent,
    "HR_Agent": hr_agent,
    "Finance_Agent": finance_agent,
    "Operations_Agent": operations_agent
}

# --- Streamlit Chat UI Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
if user_query := st.chat_input("Type your support query here... (e.g., 'How do I reset my VPN password?')"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Routing and processing request..."):
            try:
                async def process_query():
                    # Step 1: Router classifies intent
                    router_result = await router_agent.run(task=user_query)
                    raw_routing = router_result.messages[-1].content.strip()
                    
                    # Clean/Match the targeted agent name
                    target_agent = "HR_Agent" # default fallback
                    for key in agent_map.keys():
                        if key in raw_routing:
                            target_agent = key
                            break
                    
                    # Step 2: Directly invoke the chosen specialist agent
                    specialist = agent_map[target_agent]
                    response_result = await specialist.run(task=user_query)
                    answer = response_result.messages[-1].content
                    
                    return target_agent, answer

                responding_agent, response_content = asyncio.run(process_query())
                
                # Format output with guaranteed matching badge
                formatted_output = f"🧭 **Routed to:** `{responding_agent}`\n\n{response_content}"
                
                st.markdown(formatted_output)
                st.session_state.messages.append({"role": "assistant", "content": formatted_output})
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})