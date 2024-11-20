import streamlit as st
import requests

# Streamlit Page Configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for improved styling
st.markdown("""
    <style>
    .stTextInput > label {
        font-size: 18px;
        font-weight: bold;
        color: #2C3E50;
    }
    .stTextArea {
        border: 1px solid #ddd;
        padding: 10px;
    }
    .stButton > button {
        background-color: #4CAF50; 
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .result-box {
        background-color: #F0F8FF;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
    This tool uses Machine Learning to predict the sentiment of your text. Enter your text in the input box, and click the "Analyze Sentiment" button to see if the sentiment is positive or negative.
    
""")

# Main Content
st.title("🔍 Sentiment Analysis Tool")
st.markdown("""
    Enter a sentence below to analyze its sentiment. This tool will predict whether the sentiment is **Positive** or **Negative** and provide a confidence score.
""")

# User Input Section
input_text = st.text_area(
    label="Enter Your Text Below:",
    placeholder="e.g., I'm feeling great about this new project!",
    height=150
)

# Prediction Button
if st.button("Analyze Sentiment"):
    if input_text:
        with st.spinner("Analyzing..."):
            url = "http://192.168.1.3:8000/predict"  # Local FastAPI server URL
            response = requests.post(url, json={"text": input_text})

            if response.status_code == 200:
                result = response.json()
                sentiment = result["sentiment"]
                confidence = result["confidence"]

                # Display the results in a styled box
                if sentiment == "Positive":
                    result_message = f"<span style='color:black'>😊 <strong>Sentiment: {sentiment}</strong></span>"
                else:
                    result_message = f"<span style='color:black'>😞 <strong>Sentiment: {sentiment}</strong></span>"
                
                confidence_message = f"<span style='color:black'>Confidence Score: <strong>{confidence:.2%}</strong></span>"

                # Display the sentiment and confidence score
                st.markdown(f'<div class="result-box">{result_message}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-box">{confidence_message}</div>', unsafe_allow_html=True)
            else:
                st.error("Error: Could not get a response from the FastAPI backend. Please try again later.")
    else:
        st.warning("Please enter some text for analysis.")


# Footer Section
st.markdown("---")
st.markdown("""

    </div>
""", unsafe_allow_html=True)
