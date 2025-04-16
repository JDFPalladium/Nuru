# Import libraries
import os
os.environ["OPENAI_API_KEY"]

from dotenv import load_dotenv
import logging
import sys
import re
import json
from datetime import datetime
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from lingua import Language, LanguageDetectorBuilder
import gradio as gr
from openai import OpenAI as OpenAIOG
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from deep_translator import GoogleTranslator

# Initialize OpenAI clients
client = OpenAIOG()

# Set seed for language detection consistency
DetectorFactory.seed = 0

# Load index for retrieval
storage_context = StorageContext.from_defaults(persist_dir="lamis_lp_metadata")
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=3)

# Define keyword lists
acknowledgment_keywords_yo = ["Ẹ ṣé", "Ẹ ṣé gan", "Ẹ ṣéun", "Ọ ṣeun", "Ọ dára", "Ọ tọ́", "Mo ti gbọ́",
                              "Ẹ ṣeun fún ifọ̀rọ̀wánilẹ́nuwò", "Ó yé mi", "Kò burú"]
acknowledgment_keywords_en = ["thanks", "thank you", "thx", "ok", "okay", "great", "got it", "appreciate", "good", "makes sense"]
follow_up_keywords = ["Ṣùgbọ́n", "Pẹ̀lú", "Tun", "Ati", "Kí ni", "Báwo", "Kí ló dé", "Èéṣé", "Nigbà wo", "Ni", "?",
                     "but", "also", "and", "what", "how", "why", "when", "is"]
greeting_keywords_yo = ["Báwo ni", "Ẹ káàárọ̀", "Ẹ káàsán", "Ẹ kúùrọ̀lẹ́", "Ẹ káàbọ̀", "Ẹ kúulé", "Ẹ kuùjọ̀kòó"]
greeting_keywords_en = ["hi", "hello", "hey", "how's it", "what's up", "yo", "howdy"]

# Define helper functions

def contains_exact_word_or_phrase(text, keywords):
    """Check if the given text contains any exact keyword from the list."""
    text = text.lower()
    return any(re.search(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords)

def contains_greeting_yo(text):
    return contains_exact_word_or_phrase(text, greeting_keywords_yo)

def contains_greeting_en(text):
    return contains_exact_word_or_phrase(text, greeting_keywords_en)

def contains_acknowledgment_yo(text):
    return contains_exact_word_or_phrase(text, acknowledgment_keywords_yo)

def contains_acknowledgment_en(text):
    return contains_exact_word_or_phrase(text, acknowledgment_keywords_en)

def contains_follow_up(text):
    return contains_exact_word_or_phrase(text, follow_up_keywords)

def detect_language(text):
    """Detect language of a given text using Lingua, restricted to Yoruba and English."""
    languages = [Language.ENGLISH, Language.YORUBA]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    detected_language = detector.detect_language_of(text)
    if detected_language is None:
        return "unknown"
    return "yo" if detected_language == Language.YORUBA else "en"


# Define Gradio function
def nishauri(question, conversation_history: list[str]):

    """Process user query, detect language, handle greetings, acknowledgments, and retrieve relevant information."""
    context = " ".join([item["user"] + " " + item["chatbot"] for item in conversation_history])
    
    # Process greetings and acknowledgments
    for lang, contains_greeting, contains_acknowledgment in [("english", contains_greeting_en, contains_acknowledgment_en), ("yoruba", contains_greeting_yo, contains_acknowledgment_yo)]:
        if contains_greeting(question) and not contains_follow_up(question):
            prompt = f"The user said: {question}. Respond accordingly in {lang}."
        elif contains_acknowledgment(question) and not contains_follow_up(question):
            prompt = f"The user acknowledged: {question}. Respond accordingly in {lang}."
        else:
            continue
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        reply_to_user = completion.choices[0].message.content
        conversation_history.append({"user": question, "chatbot": reply_to_user})
        return reply_to_user, conversation_history

    # Detect language and translate if needed
    lang_question = detect_language(question)
    if lang_question == "yo":
        question = GoogleTranslator(source='yo', target='en').translate(question)
    
    # Retrieve relevant sources
    sources = retriever.retrieve(question)
    retrieved_text = "\n\n".join([f"Source {i+1}: {source.text}" for i, source in enumerate(sources[:3])])

    # Combine into new user question - conversation history, new question, retrieved sources
    question_final = (
        f"The user asked the following question: \"{question}\"\n\n"
        f"Use only the content below to answer the question:\n\n{retrieved_text}\n\n"
        "Guidelines:\n"
        "- Only answer the question that was asked.\n"
        "- Do not change the subject or include unrelated information.\n"
        "- Only discuss topics related to HIV and associated infections. If the question is not relevant, say that you can only answer relevant questions.\n"
    )

    # Set LLM instructions. If user consented, add user parameters, otherwise proceed without
    system_prompt = (
        "You are a helpful assistant who only answers questions about Nigeria's HIV guidelines and about using the LAMIS Plus EMR.\n"
        "- Do not answer questions about other topics.\n"
        "- If a question is unrelated to HIV or LAMIS Plus, politely respond that you can only answer HIV- or LAMIS Plus-related questions.\n\n"
    )
 
    # Start with context
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for turn in conversation_history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["chatbot"]})
    
    # Finally, add the current question
    messages.append({"role": "user", "content": question_final})

    # Generate response
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    # Collect response
    reply_to_user = completion.choices[0].message.content

    # add question and reply to conversation history
    conversation_history.append({"user": question, "chatbot": reply_to_user})  

    # If initial question was in yoruba, translate response to yoruba
    if lang_question=="yo":
        reply_to_user = GoogleTranslator(source='auto', target='yo').translate(reply_to_user) 

    # return system_prompt, conversation_history 
    return reply_to_user, sources[0].text, sources[1].text, sources[2].text, conversation_history    


demo = gr.Interface(
    title = "Idahun Chatbot Demo",
    fn=nishauri,
    inputs=["text", gr.State(value=[])],
    outputs=[
        gr.Textbox(label = "Idahun Response", type = "text"),
        gr.Textbox(label = "Source 1", max_lines = 10, autoscroll = False, type = "text"),
        gr.Textbox(label = "Source 2", max_lines = 10, autoscroll = False, type = "text"),
        gr.Textbox(label = "Source 3", max_lines = 10, autoscroll = False, type = "text"),
        gr.State()
            ],
)

demo.launch()

