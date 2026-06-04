# 6 VLM Model ile Few-Shot Deney Planı — RTX 4060 Uyumlu

Bu dosya, mevcut klasör yapındaki veri setini kullanarak 6 farklı Vision-Language Model (VLM) ile **few-shot inference/evaluation** yapman için hazırlanmıştır.

> Hedef: Eğitim/fine-tuning yapmak değil; her sınıftan az sayıda örneği prompt içine koyarak VLM modellerinden tahmin almak, sonuçları karşılaştırmak ve raporlamak.

---

## 1. Mevcut Proje Klasör Yapısı

Ekran görüntüsüne göre proje klasörün yaklaşık şu yapıda:

```text
project/
├── .venv/
├── .vscode/
├── archive/
├── ConvNeXt/
├── CVT/
├── data/
├── EfficientNetV2/
├── MobileNetV3/
├── models/
├── notebooks/
├── outputs/
├── ResNeSt/
├── ResNeXt/
├── data_prep_gui.py
├── evaluation.ipynb
├── README.md
└── requirements.txt
```

Bu çalışma için ana klasörler:

```text
data/       → veri seti
outputs/    → model sonuçları
models/     → indirilen/cache model dosyaları veya konfigürasyonlar
notebooks/  → deney not defterleri
```

Önerilen yeni yapı:

```text
project/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── outputs/
│   └── vlm_fewshot/
│       ├── predictions/
│       ├── metrics/
│       └── logs/
├── scripts/
│   ├── vlm_fewshot_config.py
│   ├── build_fewshot_examples.py
│   ├── run_vlm_fewshot.py
│   ├── evaluate_vlm_results.py
│   └── compare_models.py
└── reports/
    └── vlm_fewshot_report.md
```

---

## 2. RTX 4060 için Gerçekçi Strateji

RTX 4060 genellikle 8 GB VRAM ile kullanılıyor. Bu yüzden:

- Büyük VLM modellerini tam precision çalıştırma.
- 7B modellerde mutlaka 4-bit quantization kullan.
- Aynı anda tek model yükle.
- Görsel çözünürlüğünü sınırlı tut.
- Batch size değerini 1 yap.
- Her modelin sonucunu ayrı ayrı kaydet.
- Model karşılaştırmasını sonradan CPU üzerinde yap.

Önerilen ayarlar:

```python
IMAGE_SIZE = 224 veya 336
BATCH_SIZE = 1
NUM_SHOTS = 1, 3, 5
MAX_NEW_TOKENS = 32
LOAD_IN_4BIT = True
DEVICE = "cuda"
DTYPE = "float16"
```

---

## 3. Seçilecek 6 VLM Model

Aşağıdaki modeller few-shot denemeleri için uygundur. Amaç hepsini aynı prompt yapısı ile denemek ve sonuçları karşılaştırmaktır.

| No | Model | Yaklaşık Boyut | RTX 4060 Uygunluğu | Not |
|---:|---|---:|---|---|
| 1 | `HuggingFaceTB/SmolVLM-Instruct` | 2B | Çok uygun | Hafif ve hızlı başlangıç modeli |
| 2 | `HuggingFaceTB/SmolVLM-500M-Instruct` | 500M | Çok uygun | En hafif karşılaştırma modeli |
| 3 | `Qwen/Qwen2.5-VL-3B-Instruct` | 3B | Uygun | Güçlü ve güncel VLM |
| 4 | `OpenGVLab/InternVL2_5-2B` | 2B | Uygun | Küçük ama güçlü alternatif |
| 5 | `google/paligemma-3b-mix-224` veya `google/paligemma-3b-pt-224` | 3B | Uygun | 224px ile daha rahat çalışır |
| 6 | `llava-hf/llava-1.5-7b-hf` | 7B | 4-bit ile denenebilir | Daha ağır referans model |

Alternatif model:

```text
microsoft/Phi-3.5-vision-instruct
```

Bu model güçlüdür fakat bazı kurulumlarda daha fazla VRAM/özel kod gerektirebilir. İlk 6 model sorunsuz çalıştıktan sonra ek deney olarak denenebilir.

---

## 4. Few-Shot Ne Demek?

Few-shot yaklaşımında model eğitilmez. Modele şu mantıkta örnekler verilir:

```text
Aşağıda sınıflandırma örnekleri var.

Örnek 1:
Görsel: image_a.jpg
Cevap: sınıf_1

Örnek 2:
Görsel: image_b.jpg
Cevap: sınıf_2

Şimdi bu yeni görselin sınıfını tahmin et:
Görsel: test_image.jpg

Sadece sınıf adını yaz.
```

VLM tarafında her modelin çoklu görsel desteği farklı olabilir. Bu yüzden iki yöntem hazırlanmalı:

### Yöntem A — Gerçek Multi-Image Few-Shot

Model birden fazla görseli aynı prompt içinde alabiliyorsa kullanılır.

Uygun olabilecek modeller:

```text
Qwen2.5-VL
InternVL
SmolVLM
LLaVA
```

### Yöntem B — Textual Few-Shot

Eğer model çoklu görsel desteğinde problem çıkarırsa, örnek görsellerin sadece sınıf adları ve kısa açıklamaları prompta yazılır.

```text
Örnekler:
- Bu veri setinde "class_1" genellikle şuna benzer: ...
- Bu veri setinde "class_2" genellikle şuna benzer: ...

Yeni görseli incele ve sınıfı tahmin et.
```

Bu yöntem daha zayıftır ama tüm modellerde daha stabil çalışır.

---

## 5. Veri Seti Formatı

En temiz yapı:

```text
data/
├── train/
│   ├── class_1/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   ├── class_2/
│   └── class_3/
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── class_3/
└── test/
    ├── class_1/
    ├── class_2/
    └── class_3/
```

Eğer şu an `data/` klasörün bu formatta değilse önce dönüştürülmeli.

Kontrol scripti:

```python
from pathlib import Path

DATA_DIR = Path("data")

for split in ["train", "val", "test"]:
    split_dir = DATA_DIR / split
    print(f"\n[{split}]")
    if not split_dir.exists():
        print("Yok:", split_dir)
        continue

    for class_dir in split_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.*")))
            print(class_dir.name, count)
```

---

## 6. Few-Shot Örnek Seçimi

Her sınıftan eşit sayıda örnek seç.

Denenecek shot sayıları:

```text
1-shot
3-shot
5-shot
```

Örnek seçim stratejisi:

```python
import random
from pathlib import Path

def build_fewshot_examples(train_dir, shots_per_class=3, seed=42):
    random.seed(seed)
    train_dir = Path(train_dir)

    examples = []

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            images.extend(class_dir.glob(ext))

        selected = random.sample(images, min(shots_per_class, len(images)))

        for img_path in selected:
            examples.append({
                "image_path": str(img_path),
                "label": class_dir.name
            })

    random.shuffle(examples)
    return examples
```

---

## 7. Prompt Şablonu

Ana sınıflandırma promptu:

```text
You are a visual classification assistant.

Your task is to classify the given image into exactly one of the following classes:

{class_names}

Use the few-shot examples as guidance.
Return only the class name.
Do not explain.
Do not add extra words.
```

Türkçe veri seti için alternatif:

```text
Sen bir görsel sınıflandırma asistanısın.

Görevin, verilen görseli aşağıdaki sınıflardan tam olarak birine atamaktır:

{class_names}

Few-shot örneklerini referans olarak kullan.
Sadece sınıf adını yaz.
Açıklama yapma.
Ekstra kelime yazma.
```

Model çıktısını kolay parse etmek için cevap formatı:

```text
Answer: <class_name>
```

veya

```text
Sınıf: <class_name>
```

---

## 8. Kurulum

Yeni sanal ortam önerisi:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Temel paketler:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes pillow pandas scikit-learn tqdm matplotlib
pip install qwen-vl-utils
pip install einops timm sentencepiece protobuf
```

Not:

```text
bitsandbytes Windows üzerinde bazen sorun çıkarabilir.
Sorun yaşarsan WSL2 + Ubuntu veya Google Colab/Vast.ai kullan.
```

---

## 9. Model Konfigürasyonu

`scripts/vlm_fewshot_config.py`

```python
VLM_MODELS = [
    {
        "name": "smolvlm_2b",
        "hf_id": "HuggingFaceTB/SmolVLM-Instruct",
        "family": "smolvlm",
        "load_in_4bit": False,
    },
    {
        "name": "smolvlm_500m",
        "hf_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "family": "smolvlm",
        "load_in_4bit": False,
    },
    {
        "name": "qwen2_5_vl_3b",
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "qwen2_5_vl",
        "load_in_4bit": True,
    },
    {
        "name": "internvl2_5_2b",
        "hf_id": "OpenGVLab/InternVL2_5-2B",
        "family": "internvl",
        "load_in_4bit": True,
    },
    {
        "name": "paligemma_3b_224",
        "hf_id": "google/paligemma-3b-mix-224",
        "family": "paligemma",
        "load_in_4bit": True,
    },
    {
        "name": "llava_1_5_7b",
        "hf_id": "llava-hf/llava-1.5-7b-hf",
        "family": "llava",
        "load_in_4bit": True,
    },
]
```

---

## 10. Test Listesi Oluşturma

```python
from pathlib import Path
import pandas as pd

def build_test_dataframe(test_dir):
    rows = []
    test_dir = Path(test_dir)

    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            for img_path in class_dir.glob(ext):
                rows.append({
                    "image_path": str(img_path),
                    "label": class_dir.name
                })

    return pd.DataFrame(rows)

df_test = build_test_dataframe("data/test")
df_test.to_csv("outputs/vlm_fewshot/test_files.csv", index=False)
print(df_test.head())
```

---

## 11. Model Çalıştırma Mantığı

Her model için:

1. Modeli yükle.
2. Few-shot örneklerini seç.
3. Test görsellerini sırayla işle.
4. Her görsel için tahmin üret.
5. Tahmini CSV olarak kaydet.
6. Modeli bellekten sil.
7. Sonraki modele geç.

Pseudo-code:

```python
for model_cfg in VLM_MODELS:
    model, processor = load_model(model_cfg)

    predictions = []

    for row in test_data:
        prompt = build_prompt(class_names, fewshot_examples)
        pred = run_single_prediction(model, processor, row["image_path"], prompt)

        predictions.append({
            "model": model_cfg["name"],
            "image_path": row["image_path"],
            "true_label": row["label"],
            "pred_label": pred
        })

    save_predictions(predictions, model_cfg["name"])

    del model
    del processor
    torch.cuda.empty_cache()
```

---

## 12. Çıktı Temizleme

VLM modelleri bazen ekstra metin döndürebilir.

Örnek kötü çıktı:

```text
The image belongs to class: cat
```

Bunu sınıf adına çevirmek gerekir.

```python
def normalize_prediction(raw_text, class_names):
    text = raw_text.strip().lower()

    for cls in class_names:
        if cls.lower() == text:
            return cls

    for cls in class_names:
        if cls.lower() in text:
            return cls

    return "UNKNOWN"
```

---

## 13. Metrikler

Her model için ölç:

- Accuracy
- Macro F1
- Weighted F1
- Confusion Matrix
- Sınıf bazlı precision/recall/f1
- Ortalama inference süresi
- Maksimum VRAM kullanımı

Kod:

```python
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

df = pd.read_csv("outputs/vlm_fewshot/predictions/qwen2_5_vl_3b.csv")

y_true = df["true_label"]
y_pred = df["pred_label"]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
print("Weighted F1:", f1_score(y_true, y_pred, average="weighted"))
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

---

## 14. Deney Planı

Her model için şu deneyleri çalıştır:

| Deney | Shot | Prompt Dili | Amaç |
|---|---:|---|---|
| E1 | 0-shot | İngilizce | Temel performans |
| E2 | 1-shot | İngilizce | Az örnek etkisi |
| E3 | 3-shot | İngilizce | Dengeli few-shot |
| E4 | 5-shot | İngilizce | Daha fazla örnek etkisi |
| E5 | 3-shot | Türkçe | Türkçe prompt etkisi |

Toplam:

```text
6 model × 5 deney = 30 deney
```

RTX 4060 için önce küçük test:

```text
6 model × 1 deney × sınıf başı 5 test görseli
```

Sonra tam test:

```text
6 model × 5 deney × tüm test seti
```

---

## 15. Çalıştırma Komutları

Önce mini test:

```bash
python scripts/run_vlm_fewshot.py --model smolvlm_500m --shots 1 --max_samples_per_class 5
```

Sonra tek model tam test:

```bash
python scripts/run_vlm_fewshot.py --model qwen2_5_vl_3b --shots 3
```

Tüm modeller:

```bash
python scripts/run_vlm_fewshot.py --model all --shots 0
python scripts/run_vlm_fewshot.py --model all --shots 1
python scripts/run_vlm_fewshot.py --model all --shots 3
python scripts/run_vlm_fewshot.py --model all --shots 5
```

Sonuçları değerlendir:

```bash
python scripts/evaluate_vlm_results.py
python scripts/compare_models.py
```

---

## 16. Beklenen CSV Formatı

Her model için:

```text
outputs/vlm_fewshot/predictions/{model_name}_shot{n}.csv
```

CSV içeriği:

```csv
model,shots,image_path,true_label,raw_output,pred_label,is_correct,inference_time_sec
qwen2_5_vl_3b,3,data/test/class_1/img001.jpg,class_1,class_1,class_1,1,1.42
```

---

## 17. Karşılaştırma Tablosu

Final karşılaştırma:

| Model | Shot | Accuracy | Macro F1 | Weighted F1 | Avg Time/Image | Max VRAM |
|---|---:|---:|---:|---:|---:|---:|
| SmolVLM 500M | 3 | - | - | - | - | - |
| SmolVLM 2B | 3 | - | - | - | - | - |
| Qwen2.5-VL 3B | 3 | - | - | - | - | - |
| InternVL2.5 2B | 3 | - | - | - | - | - |
| PaliGemma 3B | 3 | - | - | - | - | - |
| LLaVA 1.5 7B | 3 | - | - | - | - | - |

---

## 18. RTX 4060 Bellek Sorunları İçin Çözümler

Eğer CUDA out of memory alırsan:

### 1. Görsel boyutunu düşür

```python
image = image.resize((224, 224))
```

### 2. 4-bit quantization kullan

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### 3. Max token azalt

```python
MAX_NEW_TOKENS = 16
```

### 4. Tek tek çalıştır

```python
BATCH_SIZE = 1
```

### 5. Cache temizle

```python
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

### 6. Ağır modeli sona bırak

Önerilen çalışma sırası:

```text
1. SmolVLM 500M
2. SmolVLM 2B
3. InternVL2.5 2B
4. PaliGemma 3B
5. Qwen2.5-VL 3B
6. LLaVA 1.5 7B
```

---

## 19. Başarı Kriteri

Çalışma başarılı sayılırsa:

- 6 model de aynı test seti üzerinde çalışmış olmalı.
- Her model için 0-shot, 1-shot, 3-shot, 5-shot sonucu alınmalı.
- Sonuçlar CSV olarak kaydedilmeli.
- Accuracy ve Macro F1 hesaplanmalı.
- En iyi model seçilmeli.
- En hızlı model seçilmeli.
- RTX 4060 üzerinde çalışabilirlik notu yazılmalı.

---

## 20. Rapor Şablonu

`reports/vlm_fewshot_report.md`

```markdown
# VLM Few-Shot Karşılaştırma Raporu

## Amaç

Bu çalışmada 6 farklı Vision-Language Model, aynı veri seti üzerinde few-shot sınıflandırma görevi için karşılaştırılmıştır.

## Kullanılan Modeller

1. SmolVLM 500M
2. SmolVLM 2B
3. Qwen2.5-VL 3B
4. InternVL2.5 2B
5. PaliGemma 3B
6. LLaVA 1.5 7B

## Donanım

- GPU: NVIDIA RTX 4060
- VRAM: 8 GB
- Batch size: 1
- Quantization: 4-bit / FP16

## Deneyler

| Deney | Shot | Açıklama |
|---|---:|---|
| E1 | 0 | Örneksiz tahmin |
| E2 | 1 | Her sınıftan 1 örnek |
| E3 | 3 | Her sınıftan 3 örnek |
| E4 | 5 | Her sınıftan 5 örnek |

## Sonuçlar

| Model | Shot | Accuracy | Macro F1 | Avg Time |
|---|---:|---:|---:|---:|
| - | - | - | - | - |

## En İyi Model

Buraya en yüksek Macro F1 değerine sahip model yazılacak.

## En Hızlı Model

Buraya ortalama inference süresi en düşük model yazılacak.

## Yorum

Few-shot örnek sayısı arttıkça performansın nasıl değiştiği yorumlanacak.

## Sonuç

Bu veri seti için en uygun VLM modeli şu şekilde belirlenmiştir:

- Performans odaklı seçim:
- Hız odaklı seçim:
- RTX 4060 uyumluluğu açısından seçim:
```

---

## 21. Codex/Agent İçin Net Görev Listesi

Aşağıdaki görevleri sırayla uygula:

```text
1. Mevcut proje klasör yapısını incele.
2. data/ klasöründeki veri seti formatını tespit et.
3. Eğer veri seti train/val/test şeklinde değilse uygun şekilde dönüştürmek için script yaz.
4. scripts/ klasörünü oluştur.
5. outputs/vlm_fewshot/ klasör yapısını oluştur.
6. VLM model konfigürasyon dosyasını oluştur.
7. Few-shot örnek seçme scriptini oluştur.
8. Test görsellerini CSV’ye çıkaran scripti oluştur.
9. Her model için inference çalıştıran genel scripti oluştur.
10. Model ailesine göre doğru processor/model loading fonksiyonlarını yaz.
11. 4-bit quantization desteği ekle.
12. CUDA OOM riskine karşı batch size 1 kullan.
13. Çıktıları normalize eden fonksiyon yaz.
14. Tahminleri CSV olarak kaydet.
15. Accuracy, Macro F1, Weighted F1 ve confusion matrix hesaplayan script yaz.
16. Tüm modelleri karşılaştıran özet CSV üret.
17. reports/vlm_fewshot_report.md dosyasını otomatik doldur.
18. Önce SmolVLM 500M ile mini test çalıştır.
19. Mini test başarılıysa diğer modelleri sırayla çalıştır.
20. Hata alınan modelleri logs/ klasörüne kaydet ve çalışmaya devam et.
```

---

## 22. Agent'a Verilecek Kısa Talimat

```text
Bu projede mevcut data/ klasöründeki görsel veri setini kullanarak 6 farklı VLM modeli ile few-shot image classification deneyleri yapmak istiyorum.

Donanımım RTX 4060 olduğu için bellek verimli çalışmalısın:
- batch size 1 kullan
- mümkünse 4-bit quantization kullan
- her modeli tek tek yükle/çalıştır/sil
- görselleri 224 veya 336 çözünürlüğe indir
- CUDA OOM alırsan modeli atlama, önce daha düşük ayarlarla tekrar dene

Kullanılacak modeller:
1. HuggingFaceTB/SmolVLM-500M-Instruct
2. HuggingFaceTB/SmolVLM-Instruct
3. Qwen/Qwen2.5-VL-3B-Instruct
4. OpenGVLab/InternVL2_5-2B
5. google/paligemma-3b-mix-224
6. llava-hf/llava-1.5-7b-hf

Deneyler:
- 0-shot
- 1-shot
- 3-shot
- 5-shot

Her deney için:
- true_label
- raw_output
- pred_label
- is_correct
- inference_time_sec

alanlarını içeren CSV üret.

Son olarak:
- Accuracy
- Macro F1
- Weighted F1
- Confusion Matrix
- Ortalama inference süresi

hesapla ve tüm modelleri karşılaştıran rapor oluştur.
```

---

## 23. Öncelikli Başlangıç Komutu

İlk olarak sadece en hafif modelle mini test yap:

```bash
python scripts/run_vlm_fewshot.py --model smolvlm_500m --shots 1 --max_samples_per_class 3
```

Bu başarılı olursa:

```bash
python scripts/run_vlm_fewshot.py --model smolvlm_2b --shots 3 --max_samples_per_class 5
```

Sonra tüm modele geç:

```bash
python scripts/run_vlm_fewshot.py --model all --shots 3
```

---

## 24. Önemli Not

Bu çalışma fine-tuning değildir.

Bu çalışma:

```text
VLM + Prompt + Few-Shot Examples + Evaluation
```

mantığıyla yapılacaktır.

Eğer few-shot sonuçları düşük çıkarsa sonraki aşamada:

```text
LoRA / QLoRA ile VLM fine-tuning
```

planlanabilir. Ancak RTX 4060 için önce few-shot inference ile model karşılaştırması yapmak en mantıklı adımdır.
