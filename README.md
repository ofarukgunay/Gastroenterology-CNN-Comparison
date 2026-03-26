# Gastroentoloji Goruntuleri Uzerine CNN Karsilastirmasi

Bu proje, Kvasir-v2 endoskopi veri seti uzerinde 6 farkli modern mimariyi ayni veri bolunmesi, benzer egitim ayarlari ve ortak metriklerle karsilastirmayi hedefler.

## Projenin Amaci

Ana hedef:

1. 6 modeli (tiny/small yerine medium/large tercihleriyle) egitmek
2. Her model icin train/val accuracy ve loss grafiklerini uretmek
3. Sonraki asamalarda tum modelleri ortak test setinde benchmark etmek

## Veri Seti

- Veri seti: Kvasir-v2
- Sinif sayisi: 8
- Kaynak: https://datasets.simula.no/kvasir/
- Bu repodaki beklenen hazir veri klasoru: `data/prepared-data/`
  - `data/prepared-data/train`
  - `data/prepared-data/val`
  - `data/prepared-data/test`

## Karsilastirilan 6 Model

1. ResNeSt50 (`resnest50d`)
2. ResNeXt50 (`resnext50_32x4d`)
3. MobileNetV3 Large (`mobilenetv3_large_100`)
4. EfficientNetV2 Medium (`tf_efficientnetv2_m`)
5. CvT-13 (`cvt_13`)
6. ConvNeXt Base (`convnext_base`)

## Guncel Proje Yapisi

```text
.
|-- data/
|   `-- prepared-data/
|-- data_prep_guı.py
|-- ResNeSt/
|   `-- 01_ResNeSt50_Training.ipynb
|-- ResNeXt/
|   `-- 02_ResNeXt50_Training.ipynb
|-- MobileNetV3/
|   `-- 03_MobileNetV3Large_Training.ipynb
|-- EfficienNetV2/
|   `-- 04_EfficientNetV2_Training.ipynb
|-- CVT/
|   `-- 05_CVT_Training.ipynb
|-- ConvNeXt/
|   `-- 06_ConvNeXt_Base_Training.ipynb
|-- evaluation.ipynb
|-- requirements.txt
`-- outputs/
   |-- mobilenetv3_large_100/
   |   |-- ablation_results_mobilenetv3_large_100.csv
   |   |-- plots/
   |   `-- reports/
   |-- tf_efficientnetv2_m/
   |   |-- ablation_results_tf_efficientnetv2_m.csv
   |   |-- plots/
   |   `-- reports/
   `-- ... (diger modeller)
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Ilk Asama Kullanim Akisi

1. Gerekirse `data_prep_guı.py` ile veri hazirlama adimini tamamla.
2. Asagidaki notebooklari sirayla calistir:
   - `ResNeSt/01_ResNeSt50_Training.ipynb`
   - `ResNeXt/02_ResNeXt50_Training.ipynb`
   - `MobileNetV3/03_MobileNetV3Large_Training.ipynb`
   - `EfficienNetV2/04_EfficientNetV2_Training.ipynb`
   - `CVT/05_CVT_Training.ipynb`
   - `ConvNeXt/06_ConvNeXt_Base_Training.ipynb`
3. Her model kendi cikti klasorune su dosyalari yazacaktir:
   - `best_model.pth`
   - `outputs/<model_adi>/plots/training_graph_...png` (train/val accuracy + train/val loss)
   - `outputs/<model_adi>/reports/classification_report_...txt`
   - `outputs/<model_adi>/plots/confusion_matrix_...png`
   - `outputs/<model_adi>/ablation_results_<model_adi>.csv`

## Degerlendirme

Tum modellerin ortak karsilastirmasi icin `evaluation.ipynb` kullanilir. Notebook, `models/pytorch/` altinda model klasorlerini bularak test setinde metrikleri hesaplar.

