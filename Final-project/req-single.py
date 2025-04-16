import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import re
import os
from dotenv import load_dotenv

load_dotenv()
# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Step 1: Extract all anchor tags with English fallback
def extract_anchor_tags(url):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }

    if not url.startswith("http"):
        url = "https://" + url
    base_url = url.rstrip("/")

    response = requests.get(base_url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")

    # Look for a link to the English version
    english_link = None
    for a in soup.find_all("a", href=True):
        link_text = a.get_text(strip=True).lower()
        if "english" in link_text or "/en" in a['href'].lower():
            english_link = a['href']
            break

    if english_link:
        if english_link.startswith("/"):
            base_url = base_url + english_link
        elif english_link.startswith("http"):
            base_url = english_link
        else:
            base_url = url + "/" + english_link

        print(f"🌐 Switching to English page: {base_url}")
        response = requests.get(base_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

    anchors = soup.find_all("a", href=True)
    links = []
    for a in anchors:
        text = a.get_text(strip=True)
        href = a['href']
        if href.startswith("/"):
            href = base_url.rstrip("/") + href
        elif href.startswith("#") or "javascript:" in href:
            continue
        links.append((text, href))
    return links

# Step 2: Ask Gemini which links are relevant
def ask_gemini_for_relevant_links(anchor_data):
    model = genai.GenerativeModel("gemini-1.5-flash")
    formatted = [f"Text: {text} | Href: {href}" for text, href in anchor_data]
    prompt = f"""
Here are anchor tags from a company website:

{chr(10).join(formatted)}

Which of these links are likely to help extract the following business fields?
Company Description  
Software Classification  
Enterprise-grade classification  
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
Phone  

Return only the full HREFs that I should visit.
"""
    response = model.generate_content(prompt)
    return response.text

# Step 3: Extract structured visible text from tags
def extract_structured_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")

    text_data = []

    for p in soup.find_all('p'):
        text_data.append(p.get_text(strip=True))
    for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text_data.append(h.get_text(strip=True))
    for a in soup.find_all('a', href=True):
        text_data.append(a.get_text(strip=True))

    return " ".join(text_data)

# Step 4: Extract structured info using Gemini
def extract_info_from_combined_text(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
I collected this content from the company's website pages:

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
Address 1: Country/Region  
Address 1: City  
Address 1: ZIP/Postal Code  
Address 1: Street 1  
Finance  
Email  
Phone  

Return concise answers. If anything is missing, say "Not available".
"""
    response = model.generate_content(prompt)
    return response.text

# Final Orchestration
def full_extraction_pipeline(domain):
    print(f"🔎 Extracting anchor tags from {domain}")
    anchors = extract_anchor_tags(domain)
    
    print("💬 Asking Gemini for relevant links...")
    gemini_output = ask_gemini_for_relevant_links(anchors)
    print("\n🤖 Gemini suggested URLs:\n", gemini_output)

    relevant_urls = re.findall(r"https?://[^\s\)\]\}\>\n]+", gemini_output)
    print("\n✅ Extracted URLs:", relevant_urls)

    all_text = ""
    for url in relevant_urls:
        try:
            print(f"\n🔍 Scraping structured text from: {url}")
            page_text = extract_structured_text(url)
            all_text += "\n" + page_text
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")

    print("\n📦 Extracting structured business info using Gemini...")
    return extract_info_from_combined_text(all_text)

# Replace with any company domain
domain = "https://www.ferrerandpartners.com"

# Run the pipeline
results = full_extraction_pipeline(domain)
print("\n📄 Final Extracted Info:\n", results)
