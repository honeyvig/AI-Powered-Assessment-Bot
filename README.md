# AI-Powered-Assessment-Bot
 build two AI systems: a Compliance Assessment AI and a Chatbot AI. The goal of this project is to automate compliance assessments and enhance customer interactions using AI technologies.
----------------------------------------------
Building two AI systems—a Compliance Assessment AI and a Chatbot AI—involves different components and technologies. Below is a Python-based approach to develop these systems. We'll break them down into two parts:
1. Compliance Assessment AI:

This system will automatically assess whether documents, actions, or data comply with predefined rules and regulations. It can analyze text-based documents, reports, and other data to determine compliance based on specific rules.
2. Chatbot AI:

This system will be used for interacting with customers in a conversational manner, answering queries, providing information, and offering guidance.
Let's break this down step-by-step.
Prerequisites:

    Libraries: We'll use libraries like spaCy, Transformers, OpenAI, Flask (for API development), and TensorFlow or PyTorch for the AI models.
    Cloud Services: Optionally, if you want to deploy these systems, you can use cloud services (AWS, Azure, GCP).

1. Compliance Assessment AI:

We'll start by designing an AI that reads and analyzes text-based documents to check if they comply with regulations.
a. Using NLP for Compliance Assessment:

This involves processing text, identifying key terms related to compliance, and comparing them to predefined compliance rules.

import spacy
from transformers import pipeline

# Load NLP model (spaCy for basic text processing)
nlp = spacy.load("en_core_web_sm")

# Predefined rules for compliance (example for GDPR)
compliance_rules = [
    {"term": "personal data", "compliant": True},
    {"term": "data breach", "compliant": False},
    {"term": "data protection", "compliant": True},
]

# Function to assess compliance based on a text
def assess_compliance(text):
    # Process the text using spaCy
    doc = nlp(text)
    
    # Check for compliance with rules
    assessment_results = []
    for rule in compliance_rules:
        found_terms = [token.text for token in doc if token.text.lower() == rule["term"].lower()]
        if found_terms:
            status = "Compliant" if rule["compliant"] else "Non-Compliant"
            assessment_results.append(f"Found '{rule['term']}': {status}")
    
    return assessment_results

# Example text to assess
sample_text = "This document includes personal data and complies with data protection laws but also mentions a data breach."
results = assess_compliance(sample_text)
print("Compliance Assessment Results:")
for result in results:
    print(result)

b. Using Transformer Models for Enhanced Compliance Checking:

For a more advanced approach, we can use a pre-trained transformer model (e.g., BERT, GPT) to assess compliance in a more nuanced way, by training a custom classifier.

from transformers import pipeline

# Using a transformer-based text classification model for compliance
# This could be a fine-tuned model for compliance classification tasks

compliance_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def check_compliance_with_model(text):
    labels = ["Compliant", "Non-Compliant"]
    result = compliance_classifier(text, candidate_labels=labels)
    return result['labels'][0], result['scores'][0]  # Return the label and score

# Example compliance assessment using a model
sample_text = "The company has implemented the necessary steps to protect personal data."
label, score = check_compliance_with_model(sample_text)
print(f"Compliance Label: {label}, Confidence: {score}")

2. Chatbot AI:

The Chatbot AI will engage in conversational interactions. We can use OpenAI GPT-3 (or similar) for conversational AI.
a. Using OpenAI GPT for a Chatbot:

First, make sure you have your OpenAI API key.

pip install openai

Now, we can set up the Chatbot with OpenAI's GPT models.

import openai

# Set your OpenAI API key
openai.api_key = 'your-openai-api-key'

# Function to call the OpenAI GPT-3 API for chatbot interactions
def chatbot_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # GPT-3 engine
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Example interaction
user_input = "What are your office hours?"
response = chatbot_response(user_input)
print(f"Chatbot: {response}")

b. Adding More Context and Interactivity:

You can customize the chatbot's behavior further by fine-tuning the responses and creating a more robust conversation flow. For instance, using a Flask app for an interactive web-based interface.

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

This will start a web service where you can interact with the chatbot via API calls.
c. Adding Compliance Check in Chatbot:

You can also integrate the Compliance Assessment AI into the chatbot to check if the conversations are adhering to certain standards.

@app.route("/compliance_check", methods=["POST"])
def compliance_check():
    user_input = request.json.get("user_input")
    results = assess_compliance(user_input)  # Use the function from the compliance AI
    return jsonify({"compliance_results": results})

# Example of how the chatbot and compliance check can be combined:
@app.route("/chat_with_compliance", methods=["POST"])
def chat_with_compliance():
    user_input = request.json.get("user_input")
    response = chatbot_response(user_input)
    compliance_results = assess_compliance(user_input)
    return jsonify({
        "chatbot_response": response,
        "compliance_results": compliance_results
    })

Full Project Structure

You can now deploy both systems in a combined framework:

    Compliance Assessment AI: Analyze and assess compliance using NLP and transformer models.
    Chatbot AI: Use GPT-based models to engage in natural language conversation and optionally assess compliance in the conversation.

Both systems can be accessed via a web interface (Flask) or as separate REST APIs.
Summary of Technologies Used:

    NLP (Natural Language Processing): SpaCy, Hugging Face Transformers (BERT, GPT, etc.)
    Cloud API Integration: OpenAI GPT-3 for conversational AI
    Compliance Checking: Predefined rules and machine learning models for automated compliance assessments
    Flask: Web framework to expose both AI systems as REST APIs

This setup gives you an automated system that can both assess compliance and handle user interactions via a chatbot. You can extend this further by integrating more advanced features like sentiment analysis, multi-agent workflows, or real-time monitoring.
