import streamlit as st
import pickle
import re

# Load model and vectorizer
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Recommendation engine
def recommend_action(statement):
    cleaned = clean_text(statement)
    vectorized = tfidf.transform([cleaned])
    prediction = rf_model.predict(vectorized)[0]
    statement_lower = statement.lower()

    if prediction == 'Normal':
        action = "No action required. Log daily check as completed."
        priority = "🟢 NORMAL"
        color = "green"

    elif prediction == 'Intervention Later':
        if 'oil' in statement_lower:
            action = "Schedule oil filter check and top up within 48 hours."
        elif 'coolant' in statement_lower or 'water' in statement_lower:
            action = "Schedule coolant level check and refill within 48 hours."
        elif 'belt' in statement_lower:
            action = "Schedule belt inspection and tension adjustment soon."
        elif 'battery' in statement_lower:
            action = "Schedule battery test and terminal cleaning soon."
        elif 'filter' in statement_lower:
            action = "Schedule air filter cleaning or replacement soon."
        else:
            action = "Schedule general inspection within 48 hours."
        priority = "🟡 INTERVENTION LATER"
        color = "orange"

    else:
        if 'oil' in statement_lower:
            action = "STOP ENGINE! Urgent oil system inspection required."
        elif 'coolant' in statement_lower or 'water' in statement_lower:
            action = "STOP ENGINE! Urgent cooling system inspection required."
        elif 'belt' in statement_lower:
            action = "STOP ENGINE! Belt replacement required immediately."
        elif 'battery' in statement_lower:
            action = "DO NOT START! Urgent battery replacement required."
        elif 'smoke' in statement_lower:
            action = "STOP ENGINE! Urgent inspection for fire risk."
        elif 'noise' in statement_lower or 'grinding' in statement_lower:
            action = "STOP ENGINE! Internal engine damage suspected."
        else:
            action = "STOP ENGINE! Urgent full inspection required."
        priority = "🔴 URGENT INTERVENTION"
        color = "red"

    return prediction, priority, action, color

# ============================================
# STREAMLIT APP
# ============================================

st.set_page_config(
    page_title="Smart Maintenance Assistant",
    page_icon="🔧",
    layout="centered"
)

st.title("🔧 Smart Maintenance Assistant")
st.markdown("### Condition-Based Maintenance System using NLP")
st.markdown("---")

st.markdown("**Enter the technician's morning observation below:**")

engine_id = st.text_input("🔩 Engine ID", placeholder="e.g. ENG007")
statement = st.text_area("📝 Technician Statement", 
                          placeholder="e.g. oil leak detected under the engine...",
                          height=100)

if st.button("🔍 Analyze Statement"):
    if not statement:
        st.warning("⚠️ Please enter a statement!")
    else:
        if not engine_id:
            engine_id = "UNKNOWN"
        
        prediction, priority, action, color = recommend_action(statement)
        
        st.markdown("---")
        st.markdown(f"### Result:")
        
        if color == "green":
            st.success(f"**{priority}**")
        elif color == "orange":
            st.warning(f"**{priority}**")
        else:
            st.error(f"**{priority}**")
        
        st.markdown(f"**🔩 Engine:** {engine_id}")
        st.markdown(f"**📝 Statement:** {statement}")
        st.markdown(f"**🛠️ Recommended Action:** {action}")

st.markdown("---")
st.markdown("*ALX Data Science Portfolio — Smart CBMS Project*")