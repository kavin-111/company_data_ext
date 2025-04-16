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
# === Setup ===

# Gemini API
genai.configure(api_key="AIzaSyA9K0QR3BF957udnIbLMqhaKONlJLbJUYs")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Supabase
supabase = create_client(
    "https://lkzunbwgwgzfhemrrvxt.supabase.co",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxrenVuYndnd2d6ZmhlbXJydnh0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ3MzU1NTgsImV4cCI6MjA2MDMxMTU1OH0.2_lTlFjEaWIewQdi5QBbFW5e0dEbzVr-5NFi9bpZQzo"
)

# SerpAPI
SERPAPI_KEY = "your_serpapi_key_here"  # ⬅️ Replace with your actual SerpAPI key

# === Helper Functions ===

def get_domains(company_names):
    domains = {}
    for name in company_names:
        query = name + " official site"
        try:
            url = next(search(query, num_results=1))
            domain = urlparse(url).netloc
            print(f"🌐 Found domain for {name}: {domain}")
            domains[name] = domain
        except Exception as e:
            print(f"❌ Error finding domain for {name}: {e}")
            domains[name] = None
        time.sleep(1)
    return domains

def extract_links(domain):
    headers = {"User-Agent": "Mozilla/5.0"}
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
        print(f"❌ Error extracting links from {domain}: {e}")
        return []

def scrape_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return ""

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def store_embeddings_in_supabase(chunks, embeddings):
    try:
        supabase.table("embeddings").delete().gte('id', 0).execute()
        print("✅ Cleared previous embeddings.")
    except Exception as e:
        print(f"❌ Error clearing embeddings: {e}")

    data = [{"chunk_text": chunk, "embedding": embedding.tolist()} for chunk, embedding in zip(chunks, embeddings)]
    for attempt in range(3):
        try:
            supabase.table("embeddings").insert(data).execute()
            print("✅ Embeddings stored.")
            return
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(2)

def retrieve_embeddings_from_supabase(query, top_k=10):
    query_embedding = embedding_model.encode(query).tolist()
    response = supabase.table("embeddings").select("chunk_text, embedding").execute()
    if not response or hasattr(response, 'error') and response.error:
        print("❌ Error retrieving from Supabase.")
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
        except Exception as e:
            print(f"❌ SerpAPI error for {field}: {e}")
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

    missing_fields = [k for k, v in extracted.items() if v.lower() == "** not available"]
    if missing_fields:
        print(f"🔍 Using SerpAPI to fill: {missing_fields}")
        filled = fill_missing_with_serpapi(company_name, missing_fields)
        extracted.update(filled)

    return "\n".join([f"{k}: {v}" for k, v in extracted.items()])

def save_to_excel(data_list, filename="company_data.xlsx"):
    df = pd.DataFrame(data_list)
    df.to_excel(filename, index=False)
    print(f"📁 Data saved to {filename}")

# === Pipeline ===

def run_pipeline(domain, company_name=""):
    print(f"\n🚀 Running pipeline for: {domain}")
    links = extract_links(domain)
    print(f"🔗 {len(links)} links found. Scraping...")
    all_chunks = []
    for link in links[:10]:
        text = scrape_text(link)
        if text:
            all_chunks.extend(chunk_text(text))
        time.sleep(1)
    if not all_chunks:
        return "No content found."
    embeddings = np.array([embedding_model.encode(chunk) for chunk in all_chunks])
    store_embeddings_in_supabase(all_chunks, embeddings)
    relevant_chunks = retrieve_embeddings_from_supabase("""Extract company info that are required to answer these questions 
Company Description  
Software Classification  
Enterprise-grade classification (classify based on company size or any other criteria if data not available) 
Industry  
Customer names  
Employee head count  
Investors  
Geography  
Parent Company  
Address 1: Country/Region  
Address 1: City  
Address 1: ZIP/Postal Code  
Address 1: Street 1  
Finance  
Email  
Phone  """, top_k=10)
    combined_text = "\n".join(relevant_chunks)
    print("🤖 Sending to Gemini...")
    return ask_gemini(combined_text, company_name)

# === Main ===

if __name__ == "__main__":
    company_names = [
    "ERGO Versicherung Aktiengesellschaft",
    "ERGO Versicherung Aktiengesellschaft (Österreich)",
    "ERGO Versicherungsgruppe AG (Deutschland)",
    "ERGO Vorsorge Lebensversicherung Aktiengesellschaft",
    "Eric Sturdza Management Company S.A.",
    "ESPRIT Netzwerk AG",
    "Essedi Partners S.à r.l.",
    "Etihad Credit Insurance",
    "Etops 1",
    "Etops AG",
    "Etude Antinori",
    "Etude Kronshagen & Associés",
    "Etude Notaire Thierry Becker",
    "EU Care Insurance PCC Ltd",
    "Euro Arab Insurance",
    "Europe Fiduciaire Luxembourg S.A.",
    "European Broker S.A.",
    "Evolute 2",
    "Evolute 3",
    "Evolute 4",
    "Evolute 6",
    "Evolute 8",
    "Evolute 9",
    "EWA Group SA",
    "Exim Banca Romaneasca S.A.",
    "Eynav Gestion",
    "F3 Advisory Ltd.",
    "Fabriek.NL B.V.",
    "Fair Financial Consulting GmbH",
    "FCG Risk & Compliance AB",
    "FCS Services Sàrl",
    "Felten & Associés",
    "Ferrer and Partners Corporate Services",
    "Ferris Accounting & Management Consultancy S.à r.l.",
    "FI Health Insurance Company AD",
    "FI&FO SA",
    "FIACCOM S.A.",
    "Fibetrust S.à r.l.",
    "Ficel Fiduciaire S.A.",
    "Fidassur S.à r.l.",
    "FIDCOMA S.à.r.l. (Ex:Fiduciaire Glacis S.à r.l.)",
    "Fideuro S.A.",
    "Fiducenter S.A.",
    "Fiducia SA",
    "Fiduciaire Comptable Becker Gales Brunetti",
    "Fiduciaire Continentale S.A.",
    "Fiduciaire Denis Soumann Eurl",
    "Fiduciaire des Classes Moyennes SA - FCM",
    "Fiduciaire des P.M.E.",
    "Fiduciaire Fernand Faber SA",
    "Fiduciaire Fernand Sassel & Cie",
    "Fiduciaire Intercommunautaire Sàrl",
    "Fiduciaire Jean-Marc Faber",
    "Fiduciaire Luxembourgeoise Sàrl",
    "Fiduciaire Luxor Sarl",
    "Fiduciaire Marcel Stephany",
    "Fiduciaire Nickels Pütz S.à r.l.",
    "Fiduciaire Pletschette Meisch & Associés SA",
    "FIDUCIAIRE TAX & AUDIT S.à r.l.",
    "Fiduciaire WOTAN S.A",
    "Fidupar S.A.",
    "FIDUSAL S.à r.l.",
    "FIMEXCO SAM",
    "Finamore SA",
    "Finanzplanung, Vermögensberatung, Family Office (Inh. Dr. Dorothee Lotz)",
    "Finare",
    "FINARE ASSET MANAGEMENT S.A.",
    "FinDeal Advisers S.A.",
    "finnova AG Bankware",
    "First International Broker S.A. (ex EP Group S.A.)",
    "Fiscogest S.à r.l.",
    "FKP Management SARL",
    "FKP SERVICES S.A.",
    "Flossbach von Storch Invest S.A.",
    "Flossbach von Storch SE",
    "FLUX-Fiduciaire du Grand-Duché de Luxembourg Sàrl",
    "FONDSNET Vermögensberatung & -verwaltungs GmbH",
    "foo AG & Co. KG",
    "Forca A/S",
    "FormInvest AG",
    "Försäkringsaktiebolaget Alandia",
    "Försäkringsbranschens Pensionskassa (FPK)",
    "Fortius S.A",
    "Fortune Investment Management SA",
    "Frankfurter Bankgesellschaft (Schweiz) AG",
    "Fremtind Forsikring AS",
    "Fuchs & Associés Finance",
    "Fuchs Asset Management SA",
    "Fuchs et Associés Finance S.A.",
    "Fund Solutions SCA",
    "Fürst Fugger Privatbank Aktiengesellschaft",
    "Fürstlich Castell'sche Bank, Credit-Casse AG",
    "FWU Invest S.A.",
    "FWU Takaful GmbH",
    "G&G Associates S.à r.l.",
    "Gabler",
    "Gadd & Cie Luxembourg SA",
    "Garanta Asigurari",
    "GARD MARINE & ENERGY INSURANCE (EUROPE) AS",
    "GATSBY AND WHITE S.A.",
    "Gen II Management Company (Luxembourg) SARL - (ex. Crestbridge Management Company SA)",
    "Generali Deutschland AG",
    "Generali Group",
    "Geneva Call",
    "GETAD SAM",
    "G-Force Corporate Services B.V.",
    "Gjensidige",
    "GJENSIDIGE FORSIKRING ASA",
    "Glarner Kantonalbank",
    "Global Asset Advisors & Management S.A.",
    "Global Finance Consult - GFC",
    "Global Gestion Sàrl",
    "Global United Insurance",
    "GlobalNetint UAB - Lexis RU",
    "Globalrise Capital",
    "Glovis Europe GmbH",
    "GM Corporate and Fiduciary Services Limited",
    "Gothaer Lebensversicherung AG",
    "Gothaer Lebensversicherung AG",
    "Gothaer Solutions GmbH",
    "Gránit Biztosító Zrt. (Previously Wáberer Hungária Biztosító Zrt)",
    "Grant Thornton Ltd",
    "Grant Thornton Malta",
    "Grant Thornton Monaco",
    "Graubündner Kantonalbank",
    "Groupe Advensys Luxembourg SA",
    "Groupe Audit Luxembourg (ex. BJNP Audit)",
    "Groupe C.K. Charles Kieffer",
    "Groupe Open-LN FR",
    "Groupe SMIR SAM",
    "Gryon House",
    "GS&P Kapitalanlagegesellschaft S.A.",
    "GSL Fiduciaire S.à r.l.",
    "GSLP International S.à r.l.",
    "GT Fiduciaires S.A.",
    "Guardian Management SARL",
    "Hamburger Sparkasse AG",
    "HANCE LAW",
    "Hauck & Aufhäuser Fund Services S.A.",
    "Hauck Aufhäuser Lampe Privatbank AG",
    "HBLaw Sàrl",
    "HEAG Pensionszuschusskasse VVaG",
    "Heckler & Koch GmbH",
    "Hedvig Försäkring AB (previous name: Hedvig AB)",
    "Heineken Pensioenfonds",
    "HELLERICH GmbH",
    "Helvetia Leben Maklerservice GmbH",
    "Helvetische Bank AG",
    "HENRI J. VASSALLO LL.D.",
    "Hercules Manager Sarl",
    "Heroal",
    "HESPER FUND, SICAV",
    "Het Nederlandse Rode Kruis",
    "Hi Inov",
    "Hoche Partners Corporate Services S.A.S.",
    "Hoechster Pensionskasse VVaG",
    "HOEGEN DIJKHOF LEGAL SERVICES B.V.",
    "Holland Casino",
    "Hoogewerf & Cie",
    "HRK LUNIS AG",
    "Hrvatsko kreditno osiguranje d.d.",
    "Hrvatsko mirovinsko osiguravajuće društvo",
    "HSBC Bank plc (Guernsey Branch)",
    "HSBC Continental Europe S.A., Germany",
    "HSBC Continental Europe, S.A.",
    "HSBC Global Services (UK) Limited",
    "HSBC Private Bank (Luxembourg) S.A.",
    "HSBC Private Bank (Luxembourg) S.A., French Branch",
    "HSBC Private Bank (Suisse) SA",
    "HSBC UK Bank plc",
    "Hypothekarbank Lenzburg AG",
    "IAC",
    "IBA Partners SA",
    "ICAMAP",
    "ICARE Expertise SA",
    "Identifikaciniai Projektai",
    "IFM Independent Fund Management AG",
    "IFS Independent Financial Services AG",
    "Igri Avocats",
    "IK Investment Partners AIFM",
    "IMC Intl. Management & Trust Aruba",
    "IME Pension Fund (ASSEP)",
    "Immobel Luxembourg SA",
    "IMMOFLEX & BATIFLEX"
]
domains = get_domains(company_names)
results = []

for company, domain in domains.items():
        if not domain:
            print(f"⚠️ Skipping {company}, no domain found.")
            continue
        
        result_text = run_pipeline(domain, company)
        print(f"\n📄 Extracted Info for {company}:\n{result_text}\n{'='*80}")

        result_dict = {
            "Company Name": company,
            "Domain": domain
        }

        # Parse result_text into key-value pairs and add to dict
        for line in result_text.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                result_dict[key.strip()] = value.strip()

        results.append(result_dict)

save_to_excel(results)