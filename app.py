from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from werkzeug.utils import secure_filename
import requests
import re
import uuid

app = Flask(__name__)

# Klasör ayarı
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# BLIP Modeli
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)

# Yardımcı fonksiyonlar
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def blip_generate(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors='pt').to(device)
    out = blip_model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

def parse_gemma_sections(text):
    text = text.replace("*", "").replace("**", "").strip()
    sections = {
        "Kıyafet Analizi": "",
        "Renk Uyumu": "",
        "Kombin Önerisi": "",
        "Kombin Yorumu": "",
        "Alternatif Kombin": "",
    }
    pattern = r"(Kıyafet Analizi|Renk Uyumu|Kombin Önerisi|Kombin Yorumu|Alternatif Kombin)\s*:\s*(.*?)\s*(?=(Kıyafet Analizi|Renk Uyumu|Kombin Önerisi|Kombin Yorumu|Alternatif Kombin|$))"
    matches = re.findall(pattern, text, re.DOTALL)
    for (heading, content, _) in matches:
        sections[heading] = content.strip()
    return sections

def extract_yorum(answer, title="Kıyafet Yorumu"):
    answer = answer.replace("*", "").replace("**", "").strip()
    try:
        yorum_text = answer.split(title + ":")[1].strip()
    except:
        yorum_text = answer.strip()
    lines = [line.strip("•-– ") for line in yorum_text.splitlines() if line.strip()]
    return "\n".join(f"- {line}" for line in lines)

# Anasayfa
@app.route('/')
def index():
    return render_template('index.html')

# Kombin oluşturma endpointi
@app.route('/kombin-olustur', methods=['POST'])
def kombin_olustur():
    if 'files[]' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400

    files = request.files.getlist('files[]')
    if len(files) < 2:
        return jsonify({"error": "En az 2 kıyafet resmi yüklemelisiniz."}), 400

    tarz = request.form.get('tarz', 'Casual')
    mevsim = request.form.get('mevsim', 'İlkbahar')
    image_descriptions = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            description = blip_generate(filepath)
            image_descriptions.append(description)

    # PROMPT
    gemma_prompt = f"""
Sen moda uzmanısın. Şu kıyafetleri "{tarz}" tarzda ve "{mevsim}" mevsimine uygun analiz et:
{', '.join(image_descriptions)}

Lütfen aşağıdaki başlıklara 1-2 cümlelik profesyonel ve sade cevaplar ver. Başlıkları belirt:
Kıyafet Analizi:
Renk Uyumu:
Kombin Önerisi:
Kombin Yorumu:
Alternatif Kombin:
"""

    GEMMA_API_URL = os.environ.get("GEMMA_API_URL", "http://localhost:11434/api/generate")
    response = requests.post(
        GEMMA_API_URL,
        json={
            "model": "gemma2",
            "prompt": gemma_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 200
            }
        }
    )

    if response.status_code == 200:
        gemma_answer = response.json().get('response', '')
        parsed_sections = parse_gemma_sections(gemma_answer)
        gemma2_response = {
            "kombin_analizi": parsed_sections["Kıyafet Analizi"],
            "renk_uyumu": parsed_sections["Renk Uyumu"],
            "kombin_onerisi": parsed_sections["Kombin Önerisi"],
            "kombin_yorumu": parsed_sections["Kombin Yorumu"],
            "alternatif_kombin": parsed_sections["Alternatif Kombin"]
        }
    else:
        gemma2_response = {key: "Model yanıt vermedi" for key in [
            "kombin_analizi", "renk_uyumu", "kombin_onerisi", "kombin_yorumu", "alternatif_kombin"
        ]}

    return jsonify({
        "blip_aciklamalari": image_descriptions,
        "gemma2_cevabi": gemma2_response
    })

# Kıyafet Yorumla
@app.route('/yorum-olustur', methods=['POST'])
def yorum_olustur():
    yorum_text = request.form.get('yorum', '')
    file = request.files.get('yorumFile')
    description = ""
    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        description = blip_generate(filepath)

    full_description = " ".join([yorum_text, description]).strip()
    if not full_description:
        return jsonify({"error": "Yorum veya görsel gerekli"}), 400

    gemma_yorum_prompt = f"""
Sen deneyimli bir moda stil danışmanısın. Kullanıcının tanımladığı veya görsel olarak paylaştığı kıyafeti analiz et. 
Yorumun hem kıyafetin genel şıklığını hem de giyilebilecek ortamı ve küçük önerileri içersin.

Yalnızca şu formatta cevap ver:
Kıyafet Yorumu:
- Kıyafeti kısa ve profesyonel bir dille değerlendir.
- Uygun ortam/tören için görüş belirt.
- Dilersen 1 küçük dokunuş öner.

Maddeleri açık ve anlaşılır yaz. Süs karakterleri kullanma.
"""

    GEMMA_API_URL = os.environ.get("GEMMA_API_URL", "http://localhost:11434/api/generate")
    response = requests.post(
        GEMMA_API_URL,
        json={
            "model": "gemma2",
            "prompt": gemma_yorum_prompt + "\n" + full_description,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 150
            }
        }
    )

    if response.status_code == 200:
        gemma_yorum_answer = response.json().get('response', '')
        yorum_result = extract_yorum(gemma_yorum_answer)
    else:
        yorum_result = "Model yanıt vermedi."

    return jsonify({
        "yorum": yorum_result
    })

if __name__ == '__main__':
    app.run(debug=True, port=8501)



# Flask uygulamasını başlat













'''from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from werkzeug.utils import secure_filename
import requests
import re

app = Flask(__name__)

# Upload folder'ı oluştur
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# BLIP Modelini yükleyelim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)

# İzin verilen uzantılar
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Dosya uzantı kontrolü
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

def parse_gemma_answer(answer_text):
    """
    Gemma2'den gelen cevabı,
    "Kıyafet Analizi: ...",
    "Renk Uyumu: ...",
    "Kombin Önerisi: ...",
    "Kombin Yorumu: ...",
    "Alternatif Kombin: ..."
    başlıklarına göre parçalara ayırır.
    Gelen yanıttan "*" karakterlerini temizler.
    """
    answer_text = answer_text.replace("*", "")
    
    sections = {
        "Kıyafet Analizi": "",
        "Renk Uyumu": "",
        "Kombin Önerisi": "",
        "Kombin Yorumu": "",
        "Alternatif Kombin": "",
    }
    pattern = r"(Kıyafet Analizi|Renk Uyumu|Kombin Önerisi|Kombin Yorumu|Alternatif Kombin)\s*:\s*(.*?)\s*(?=(Kıyafet Analizi|Renk Uyumu|Kombin Önerisi|Kombin Yorumu|Alternatif Kombin|$))"
    matches = re.findall(pattern, answer_text, re.DOTALL)
    
    for (heading, content, _) in matches:
        sections[heading] = content.strip()

    return sections

@app.route('/kombin-olustur', methods=['POST'])
def kombin_olustur():
    if 'files[]' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400

    files = request.files.getlist('files[]')
    
    if len(files) < 2:
        return jsonify({"error": "En az 2 kıyafet resmi yüklemelisiniz."}), 400

    tarz = request.form.get('tarz', 'Casual')
    mevsim = request.form.get('mevsim', 'İlkbahar')

    image_descriptions = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # BLIP ile açıklama üretme
            image = Image.open(filepath).convert('RGB')
            inputs = processor(image, return_tensors='pt').to(device)
            out = blip_model.generate(**inputs)
            description = processor.decode(out[0], skip_special_tokens=True)
            image_descriptions.append(description)

    # Gemma2 prompt hazırla (daha net yanıt alması için düzenlendi)
    gemma_prompt = f"""
Sen moda uzmanısın. Şu kıyafetleri "{tarz}" tarzda ve "{mevsim}" mevsimine uygun analiz et:
{', '.join(image_descriptions)}.

Lütfen cevaplarını aşağıdaki formatta ver. Her başlık için yalnızca 1-2 cümle yaz ve başlıkları tekrarlama:
Kıyafet Analizi:
Renk Uyumu:
Kombin Önerisi:
Kombin Yorumu:
Alternatif Kombin:
"""

    # Ollama ile Gemma2 çağrısı
    GEMMA_API_URL = os.environ.get("GEMMA_API_URL", "http://localhost:11434/api/generate")

    response = requests.post(
        GEMMA_API_URL,
        json={
        "model": "gemma2",
        "prompt": gemma_prompt,  # veya gemma_yorum_prompt
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 200  # yorum kısmında 150 olabilir
            }
        }
    )

    if response.status_code == 200:
        gemma_answer = response.json().get('response', '')
        parsed_sections = parse_gemma_answer(gemma_answer)
        gemma2_response = {
            "kombin_analizi": parsed_sections["Kıyafet Analizi"],
            "renk_uyumu": parsed_sections["Renk Uyumu"],
            "kombin_onerisi": parsed_sections["Kombin Önerisi"],
            "kombin_yorumu": parsed_sections["Kombin Yorumu"],
            "alternatif_kombin": parsed_sections["Alternatif Kombin"]
        }
    else:
        gemma2_response = {
            "kombin_analizi": "Model yanıt vermedi",
            "renk_uyumu": "Model yanıt vermedi",
            "kombin_onerisi": "Model yanıt vermedi",
            "kombin_yorumu": "Model yanıt vermedi",
            "alternatif_kombin": "Model yanıt vermedi"
        }

    return jsonify({
        "blip_aciklamalari": image_descriptions,
        "gemma2_cevabi": gemma2_response
    })

@app.route('/yorum-olustur', methods=['POST'])
def yorum_olustur():
    # Kullanıcının girdiği metni al
    yorum_text = request.form.get('yorum', '')
    file = None
    if 'yorumFile' in request.files:
        file = request.files['yorumFile']

    description = ""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image = Image.open(filepath).convert('RGB')
        inputs = processor(image, return_tensors='pt').to(device)
        out = blip_model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)

    # Kullanıcının girdiği metin ve/veya fotoğraftan elde edilen açıklamayı birleştir
    if yorum_text and description:
        full_description = yorum_text + " " + description
    elif yorum_text:
        full_description = yorum_text
    elif description:
        full_description = description
    else:
        return jsonify({"error": "Lütfen yorum yapmak için metin giriniz veya fotoğraf yükleyiniz."}), 400

    # Gemma2'ye gönderilecek prompt: yalnızca kıyafet yorumu istenecek
    gemma_yorum_prompt = f"""
Sen moda uzmanısın. Aşağıdaki kıyafeti yorumla:
{full_description}

Lütfen kısa ve öz bir kıyafet yorumu yap ve sadece "Kıyafet Yorumu:" kısmını oluştur.
Kıyafet Yorumu:
"""

    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": "gemma2",
            "prompt": gemma_yorum_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 150
            }
        }
    )
    
    if response.status_code == 200:
        gemma_yorum_answer = response.json().get('response', '')
        # Cevabı ayrıştırmak için "Kıyafet Yorumu:" ifadesinden sonrasını alıyoruz
        def extract_yorum(answer, title):
            try:
                return answer.split(title + ":")[1].strip()
            except:
                return answer.strip()
        yorum_result = extract_yorum(gemma_yorum_answer, "Kıyafet Yorumu")
    else:
        yorum_result = "Model yanıt vermedi"

    return jsonify({
        "yorum": yorum_result
    })

if __name__ == '__main__':
    app.run(debug=True, port=8501)
# Flask uygulamasını başlat
'''



'''from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from werkzeug.utils import secure_filename
import requests
import json

app = Flask(__name__)

# Upload folder'ı oluştur
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# BLIP Modelini yükleyelim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)

# İzin verilen uzantılar
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Dosya uzantı kontrolü
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

import requests  # en üste ekle

@app.route('/kombin-olustur', methods=['POST'])
def kombin_olustur():
    if 'files[]' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    
    files = request.files.getlist('files[]')
    
    if len(files) < 2:
        return jsonify({"error": "En az 2 kıyafet resmi yüklemelisiniz."}), 400

    tarz = request.form.get('tarz', 'Casual')
    mevsim = request.form.get('mevsim', 'İlkbahar')

    image_descriptions = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # BLIP ile açıklama üretme
            image = Image.open(filepath).convert('RGB')
            inputs = processor(image, return_tensors='pt').to(device)
            out = blip_model.generate(**inputs)
            description = processor.decode(out[0], skip_special_tokens=True)
            image_descriptions.append(description)

    # Gemma2 prompt hazırla
    gemma_prompt = f"""
Sen moda uzmanısın. Şu kıyafetleri "{tarz}" tarzda ve "{mevsim}" mevsimine uygun analiz et:
{', '.join(image_descriptions)}.

Maksimum 1-2 cümle ile şu başlıklarda yanıtla:
- Kıyafet Analizi:
- Renk Uyumu:
- Kombin Önerisi:
- Kombin Yorumu:
- Alternatif Kombin:
"""


    # Ollama ile Gemma2 çağrısı
    response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        "model": "gemma2",
        "prompt": gemma_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,  # daha stabil cevaplar
            "num_predict": 200   # daha kısa cevaplar
        }
    }
)


    if response.status_code == 200:
        gemma_answer = response.json()['response']

        # Gemma cevabını başlıklara göre parçala
        def extract_section(answer, title):
            try:
                return answer.split(title+":")[1].split("-")[0].strip()
            except:
                return "Analiz yapılamadı."

        gemma2_response = {
            "kombin_analizi": extract_section(gemma_answer, "Kıyafet Analizi"),
            "renk_uyumu": extract_section(gemma_answer, "Renk Uyumu"),
            "kombin_onerisi": extract_section(gemma_answer, "Kombin Önerisi"),
            "kombin_yorumu": extract_section(gemma_answer, "Kombin Yorumu"),
            "alternatif_kombin": extract_section(gemma_answer, "Alternatif Kombin")
        }
    else:
        gemma2_response = {
            "kombin_analizi": "Model yanıt vermedi",
            "renk_uyumu": "Model yanıt vermedi",
            "kombin_onerisi": "Model yanıt vermedi",
            "kombin_yorumu": "Model yanıt vermedi",
            "alternatif_kombin": "Model yanıt vermedi"
        }

    return jsonify({
        "blip_aciklamalari": image_descriptions,
        "gemma2_cevabi": gemma2_response
    })


if __name__ == '__main__':
    app.run(debug=True,port=8501)
    # Flask uygulamasını başlat
'''