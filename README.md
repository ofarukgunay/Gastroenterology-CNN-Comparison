# Gastroentoloji Görüntüleri Üzerine CNN Sınıflandırma Algoritmalarının Karşılaştırılması

Bu proje, gastrointestinal (GI) sistemden alınan endoskopik görüntüler üzerinde modern Konvolüsyonel Sinir Ağı (CNN) mimarilerinin performansını karşılaştırmayı amaçlamaktadır. Proje kapsamında, farklı CNN modelleri kullanılarak polipler, ülserler ve normal doku gibi çeşitli gastrointestinal bulguların sınıflandırılması hedeflenmiştir.

**Yazar:** Ömer Faruk Günay
**Tarih:** Ekim 2025

---

## 📋 İçindekiler
- [Proje Hakkında](#proje-hakkında)
- [Kullanılan Veri Seti](#kullanılan-veri-seti)
- [Karşılaştırılan Mimariler](#karşılaştırılan-mimariler)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Sonuçlar](#sonuçlar)

---

## 🚀 Proje Hakkında

Bu çalışmanın temel amacı, endoskopik görüntülerin otomatik analizinde derin öğrenmenin etkinliğini göstermek ve güncel CNN mimarilerinin bu alandaki başarımını ölçmektir.

---

## 📊 Kullanılan Veri Seti

Projede, halka açık olan **Kvasir-v2** veri seti kullanılmıştır. Bu veri seti, 8 farklı sınıfta toplam 8000 adet endoskopik görüntü içermektedir.

- **Veri Seti Adı:** Kvasir-v2
- **Kaynak:** [Simula Datasets](https://datasets.simula.no/kvasir/)
- **Sınıflar:** 8 (polyps, normal-cecum, ulcerative-colitis vb.)

---

## 🧠 Karşılaştırılan Mimariler

Aşağıdaki modern ve yüksek performanslı CNN mimarileri, Transfer Öğrenme (Transfer Learning) tekniği kullanılarak karşılaştırılmıştır:

* **CVT (Convolutional Vision Transformer):** CNN ve Transformer mimarilerinin hibrit birleşimi, modern ve güçlü bir yaklaşım.
* **ResNeSt:** Split-Attention blokları kullanan gelişmiş bir ResNet varyasyonudur.
* **EfficientNetV2:** Hız ve doğruluk arasında verimli bir denge sunan modern bir mimari.
* **ConvNeXt:** Vision Transformer'lardan ilham alan ve yüksek doğruluk oranları sunan bir CNN.
* **ResNeXt:** ResNet mimarisinin "Cardinality" konsepti ile geliştirilmiş bir versiyonu.
* **MobileNetV3:** Özellikle mobil ve düşük işlem gücüne sahip cihazlar için tasarlanmış hafif bir model.

---

## 📂 Proje Yapısı

Proje dosyaları aşağıdaki gibi organize edilmiştir:
```
├── data/                 # Veri setinin bulunduğu klasör (.gitignore ile hariç tutulmuştur)
├── notebooks/            # Veri analizi ve model eğitimi Jupyter Notebook'ları
├── src/                  # Yardımcı fonksiyonlar ve Python script'leri
├── models/               # Eğitilmiş model ağırlıklarının kaydedildiği klasör (.gitignore ile hariç tutulmuştur)
├── .gitignore            # Git tarafından izlenmeyecek dosyalar
└── README.md             # Bu dosya
```

---

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

1.  **Repoyu klonlayın:**
    ```bash
    git clone [https://github.com/](https://github.com/)[kullanici-adiniz]/[repo-adiniz].git
    cd [repo-adiniz]
    ```

2.  **Veri setini indirin:**
    [Kvasir-v2](https://datasets.simula.no/kvasir/kvasir-dataset-v2.zip) linkinden veri setini indirin ve `data/` klasörünün içine çıkarın.

3.  **Gerekli kütüphaneleri yükleyin:**
    pip install -r requirements.txt
    # Veya manuel kurulum için:
    ```bash
    pip install torch torchvision timm pandas numpy matplotlib scikit-learn seaborn
    ```

---

## ⚡ Kullanım

Model eğitimi ve değerlendirme süreçleri `notebooks/` klasöründeki Jupyter Notebook'ları üzerinden yürütülebilir. Örnek olarak: `notebooks/1_EfficientNetV2_Egitimi.ipynb` dosyasını çalıştırarak eğitime başlayabilirsiniz.

---

