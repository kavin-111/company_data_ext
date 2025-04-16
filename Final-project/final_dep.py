import streamlit as st
import os
import re
import time
import requests
import numpy as np
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from googlesearch import search
from serpapi import GoogleSearch
import google.generativeai as genai
from supabase import create_client
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv

load_dotenv()
# === Setup ===

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),os.getenv("SUPABASE_KEY")
)

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# === Helper Functions ===

def get_domain(company_name):
    query = company_name + " official site"
    try:
        url = next(search(query, num_results=1))
        domain = urlparse(url).netloc
        return domain
    except Exception as e:
        st.error(f"Error finding domain for {company_name}: {e}")
        return None

def extract_links(domain):
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
    if not domain.startswith("http"):
        domain = "https://" + domain
    base_url = domain.rstrip("/")
    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http"):
                links.add(href)
            elif href.startswith("/"):
                links.add(base_url + href)
        return list(links)
    except Exception as e:
        st.error(f"Error extracting links from {domain}: {e}")
        return []

def scrape_text_with_selenium(url):
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        driver.quit()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        return ""

def scrape_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return scrape_text_with_selenium(url)

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def store_embeddings_in_supabase(chunks, embeddings):
    try:
        supabase.table("embeddings").delete().gte('id', 0).execute()
    except Exception:
        pass
    data = [{"chunk_text": chunk, "embedding": embedding.tolist()} for chunk, embedding in zip(chunks, embeddings)]
    for attempt in range(3):
        try:
            supabase.table("embeddings").insert(data).execute()
            return
        except Exception:
            time.sleep(2)

def retrieve_embeddings_from_supabase(query, top_k=10):
    query_embedding = embedding_model.encode(query).tolist()
    response = supabase.table("embeddings").select("chunk_text, embedding").execute()
    if not response or hasattr(response, 'error') and response.error:
        return []
    data = response.data
    similarities = []
    for row in data:
        chunk_vector = np.array(row["embedding"])
        similarity = np.dot(query_embedding, chunk_vector) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_vector))
        similarities.append((row["chunk_text"], similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in similarities[:top_k]]

def fill_missing_with_serpapi(company_name, fields):
    filled = {}
    for field in fields:
        query = f"{company_name} {field}"
        try:
            search = GoogleSearch({"q": query, "api_key": SERPAPI_KEY})
            results = search.get_dict()
            answer_box = results.get("answer_box", {})
            snippet = results.get("organic_results", [{}])[0].get("snippet", "Not available")
            filled[field] = answer_box.get("answer") or snippet or "Not available"
            time.sleep(1)
        except Exception:
            filled[field] = "Not available"
    return filled

def ask_gemini(text, company_name=""):
    prompt = f"""
I collected this content from the company's website:

\"\"\"{text}\"\"\" 

From this, extract the following:
Company Description  
Software Classification  
Enterprise-grade classification  
Industry  
Customer names  
Employee head count  
Investors  
Geography  
Parent Company  
Country/Region  
City  
ZIP/Postal Code  
Street 1  
Finance  
Email  
Phone  

Return only key-value pairs, no explanations. Do not include any asterisks '*', dashes '-', or markdown.
"""
    response = gemini_model.generate_content(prompt)
    output = response.text.strip()

    extracted = {}
    for line in output.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            extracted[key.strip()] = value.strip()

    missing_fields = [k for k, v in extracted.items() if v.lower() in ("** not available", "not available")]
    if missing_fields:
        filled = fill_missing_with_serpapi(company_name, missing_fields)
        extracted.update(filled)

    return extracted

def run_pipeline(company_name):
    domain = get_domain(company_name)
    if not domain:
        return None, None
    links = extract_links(domain)
    all_chunks = []
    for link in links[:10]:
        text = scrape_text(link)
        if text:
            all_chunks.extend(chunk_text(text))
        time.sleep(1)
    if not all_chunks:
        return domain, {}
    embeddings = np.array([embedding_model.encode(chunk) for chunk in all_chunks])
    store_embeddings_in_supabase(all_chunks, embeddings)
    relevant_chunks = retrieve_embeddings_from_supabase("""Extract company info required to answer these questions 
Company Description  
Software Classification  
Enterprise-grade classification  
Industry  
Customer names  
Employee head count  
Investors  
Geography  
Parent Company  
Country/Region  
City  
ZIP/Postal Code  
Street 1  
Finance  
Email  
Phone""", top_k=10)
    combined_text = "\n".join(relevant_chunks)
    result = ask_gemini(combined_text, company_name)
    return domain, result

# === Streamlit App ===

st.set_page_config(page_title="Company Info Extractor", layout="wide")

st.title("Company Information Extractor")

company_name = st.text_input("Enter a company name")

if st.button("Run Pipeline"):
    if not company_name:
        st.warning("Please enter a company name.")
    else:
        with st.spinner("Running pipeline..."):
            domain, data = run_pipeline(company_name)
        if data:
            st.success(f"Extracted info for {company_name} ({domain})")
            st.write(data)
        else:
            st.error("No information could be extracted.")
