import os
import requests

# Ngrok üzerinden yayınladığın BLIP ve GEMMA API URL’lerini buraya tanımlayabilirsin:
BLIP_API_URL = os.environ.get("BLIP_API_URL", "http://localhost:8600/blip")
GEMMA_API_URL = os.environ.get("GEMMA_API_URL", "http://localhost:11434/api/generate")

def blip_generate(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        try:
            r = requests.post(BLIP_API_URL, files=files)
            return r.json().get('description', 'Açıklama alınamadı')
        except Exception as e:
            return f"Hata: {str(e)}"

def gemma_generate(prompt, predict=200):
    try:
        response = requests.post(
            GEMMA_API_URL,
            json={
                "model": "gemma2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": predict
                }
            }
        )
        return response.json().get("response", "Gemma yanıt vermedi")
    except Exception as e:
        return f"Hata: {str(e)}"

def parse_gemma_sections(text):
    text = text.replace("*", "")
    sections = {
        "Kıyafet Analizi": "", "Renk Uyumu": "",
        "Kombin Önerisi": "", "Kombin Yorumu": "", "Alternatif Kombin": ""
    }
    import re
    pattern = r"(Kıyafet Analizi|Renk Uyumu|Kombin Önerisi|Kombin Yorumu|Alternatif Kombin)\s*:\s*(.*?)\s*(?=(Kıyafet Analizi|Renk Uyumu|Kombin Önerisi|Kombin Yorumu|Alternatif Kombin|$))"
    matches = re.findall(pattern, text, re.DOTALL)
    for (title, content, _) in matches:
        sections[title] = content.strip()
    return sections

def generate_kombin(image_paths, tarz, mevsim):
    descriptions = [blip_generate(path) for path in image_paths]
    prompt = f"""Sen moda uzmanısın. Şu kıyafetleri "{tarz}" tarzda ve "{mevsim}" mevsimine uygun analiz et:
{', '.join(descriptions)}.

Lütfen cevaplarını aşağıdaki formatta ver. Her başlık için yalnızca 1-2 cümle yaz:
Kıyafet Analizi:
Renk Uyumu:
Kombin Önerisi:
Kombin Yorumu:
Alternatif Kombin:
"""
    gemma_text = gemma_generate(prompt)
    parsed = parse_gemma_sections(gemma_text)
    return {
        "blip_aciklamalari": descriptions,
        "gemma2_cevabi": parsed
    }

def generate_yorum(text, image_path=None):
    img_desc = blip_generate(image_path) if image_path else ""
    full = f"{text} {img_desc}".strip()

    prompt = f"""Sen moda uzmanısın. Aşağıdaki kıyafeti yorumla:
{full}

Kıyafet Yorumu:"""
    yanit = gemma_generate(prompt, predict=150)
    try:
        return {"yorum": yanit.split("Kıyafet Yorumu:")[1].strip()}
    except:
        return {"yorum": yanit.strip()}
