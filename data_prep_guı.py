import os
import shutil
import threading
import random
import numpy as np
import cv2  
import albumentations as A  
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from PIL import Image
from sklearn.model_selection import train_test_split

# --- Dizin Yapısı Ayarları ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "kvasir-dataset-v2")
PREPARED_DATA_DIR = os.path.join(DATA_DIR, "data")

# --- Global Değişkenler ---
cancel_processing = False
current_output_dir = None

# --- Albumentations Pipeline ---
albumentations_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3)
])

# --- Ana Pencere ---
root = Tk()
root.title("Gastroenterology Dataset Preparation Tool (v2.0)")
root.geometry("560x700") # Pencere boyutu artırıldı
root.resizable(False, False)

# --- Fonksiyonlar ---
def select_folder():
    folder = filedialog.askdirectory(initialdir=DATA_DIR)
    dataset_path.set(folder)

def set_ui_state(state):
    widgets = [
        dataset_entry,
        train_entry, val_entry, test_entry,
        width_entry, height_entry,
        output_entry, aug_count_entry,  
        normalize_check, grayscale_check, augment_check,
        browse_button, prepare_button
    ]
    for w in widgets:
        w.config(state=state)

def cancel_preprocessing():
    global cancel_processing
    cancel_processing = True
    progress_label.config(text="Cancelling... please wait ⏳")

def start_preprocessing_thread():
    global cancel_processing
    cancel_processing = False
    thread = threading.Thread(target=start_preprocessing)
    thread.start()

def start_preprocessing():
    global current_output_dir
    set_ui_state("disabled")

    folder = dataset_path.get()
    if not folder or not os.path.isdir(folder):
        messagebox.showerror("Error", "Please select a valid dataset folder.")
        set_ui_state("normal")
        return

    try:
        # Kullanıcının hedeflediği NİHAİ oranlar
        target_train_ratio = int(train_var.get()) / 100
        target_val_ratio = int(val_var.get()) / 100
        target_test_ratio = int(test_var.get()) / 100
        
        width = int(width_var.get())
        height = int(height_var.get())
        
        # Augmentation çarpanı (1 Orijinal + N Yeni)
        # Eğer Augmentation kapalıysa çarpan 1'dir.
        aug_multiplier = 1
        if augment_var.get():
            aug_multiplier = int(aug_count_var.get()) + 1 
            
    except ValueError:
        messagebox.showerror("Error", "Invalid ratio, image size, or augmentation count values.")
        set_ui_state("normal")
        return
    
    # Oranların toplamı 1.0 (veya %100) olmalı
    if abs((target_train_ratio + target_val_ratio + target_test_ratio) - 1.0) > 0.001:
        messagebox.showerror("Error", "Ratios must sum to 100.")
        set_ui_state("normal")
        return

    # --- AKILLI ORAN HESAPLAMA (DÜZELTME BURADA) ---
    # Hedeflenen sonuca ulaşmak için BAŞLANGIÇTA nasıl bölmeliyiz?
    # Formül: Train verisi 'aug_multiplier' kadar şişeceği için, başlangıç payını küçültüyoruz.
    
    # Normalize edilmiş paydalar
    weighted_train = target_train_ratio / aug_multiplier
    weighted_val = target_val_ratio
    weighted_test = target_test_ratio
    
    total_weight = weighted_train + weighted_val + weighted_test
    
    # Gerçek (Effective) Bölme Oranları
    effective_train_ratio = weighted_train / total_weight
    effective_val_ratio = weighted_val / total_weight
    # Test oranı, kalan kısımdan otomatik çıkacak ama hesaplayalım:
    effective_test_ratio = weighted_test / total_weight
    
    print(f"Hedef: %{target_train_ratio*100} Train (x{aug_multiplier} büyüyecek)")
    print(f"Hesaplanan Başlangıç Bölmesi: Train: %{effective_train_ratio*100:.2f}, Val: %{effective_val_ratio*100:.2f}, Test: %{effective_test_ratio*100:.2f}")
    # -----------------------------------------------

    folder_name = output_folder_var.get().strip()
    if not folder_name:
        messagebox.showerror("Error", "Please enter a name for the output folder.")
        set_ui_state("normal")
        return

    output_dir = os.path.join(PREPARED_DATA_DIR, folder_name)
    current_output_dir = output_dir

    if os.path.exists(output_dir):
        confirm = messagebox.askyesno("Overwrite Confirmation", f"The folder '{folder_name}' already exists.\nDo you want to overwrite it?")
        if not confirm:
            set_ui_state("normal")
            return
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    total_images = sum(
        len([f for f in os.listdir(os.path.join(folder, c)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        for c in classes
    )
    
    # İlerleme çubuğu için toplam işlem sayısını tahmin et
    estimated_total_ops = int(total_images * effective_train_ratio * aug_multiplier) + int(total_images * (1 - effective_train_ratio))
    
    processed_ops = 0
    progress_bar["maximum"] = estimated_total_ops
    progress_bar["value"] = 0
    progress_label.config(text="Processing started... (Calculating splits)")

    for c in classes:
        if cancel_processing: break
        
        imgs = [os.path.join(folder, c, f) for f in os.listdir(os.path.join(folder, c)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not imgs: continue
        
        # Düzeltilmiş (effective) oranları kullanarak bölme işlemi
        train_files, temp = train_test_split(imgs, test_size=(1 - effective_train_ratio), random_state=42)
        
        # Kalan kısmı Val ve Test arasında orantısal böl (Burası kritik)
        if len(temp) > 0:
            relative_test_ratio = effective_test_ratio / (effective_val_ratio + effective_test_ratio)
            val_files, test_files = train_test_split(temp, test_size=relative_test_ratio, random_state=42)
        else:
            val_files, test_files = [], []

        for split_name, file_list in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            if cancel_processing: break
            
            split_dir = os.path.join(output_dir, split_name, c)
            os.makedirs(split_dir, exist_ok=True)
            
            for fpath in file_list:
                if cancel_processing: break
                
                try:
                    img = Image.open(fpath).convert("RGB")
                    img = img.resize((width, height))

                    if grayscale_var.get():
                        img = img.convert("L").convert("RGB")
                    
                    if normalize_var.get():
                        arr = np.asarray(img).astype(np.float32) / 255.0
                        img = Image.fromarray((arr * 255).astype(np.uint8))

                    if split_name == "train" and augment_var.get():
                        # 1. Orijinal kaydet
                        img.save(os.path.join(split_dir, os.path.basename(fpath)))
                        processed_ops += 1
                        
                        # 2. Çoğalt (Augment)
                        img_np = np.array(img)
                        num_augs = int(aug_count_var.get())
                        
                        for i in range(num_augs):
                            augmented_np = albumentations_pipeline(image=img_np)['image']
                            base_name, ext = os.path.splitext(os.path.basename(fpath))
                            new_filename = f"{base_name}_aug_{i}{ext}"
                            save_path = os.path.join(split_dir, new_filename)
                            cv2.imwrite(save_path, cv2.cvtColor(augmented_np, cv2.COLOR_RGB2BGR))
                            processed_ops += 1
                    
                    else:
                        img.save(os.path.join(split_dir, os.path.basename(fpath)))
                        processed_ops += 1

                    if processed_ops % 10 == 0: 
                        progress_bar["value"] = processed_ops
                        progress_label.config(text=f"Processing images... ({processed_ops}/{estimated_total_ops})")
                        root.update_idletasks()

                except Exception as e:
                    print(f"Error: {fpath} could not be processed ({e})")
        
    if cancel_processing:
        if current_output_dir and os.path.exists(current_output_dir):
            shutil.rmtree(current_output_dir)
        progress_label.config(text="Processing cancelled ❌")
        messagebox.showinfo("Cancelled", "Dataset preparation was cancelled by the user.")
    else:
        progress_bar["value"] = estimated_total_ops
        progress_label.config(text="Completed ✅")
        
        # Sonuç Raporu (Kullanıcıya gerçek dağılımı göster)
        final_msg = f"Dataset prepared successfully!\nSaved in:\n{output_dir}\n\n"
        final_msg += "Approximate Final Distribution:\n"
        final_msg += f"Train: ~{target_train_ratio*100:.1f}% (Initial split adjusted for x{aug_multiplier} augmentation)\n"
        final_msg += f"Val:   ~{target_val_ratio*100:.1f}%\n"
        final_msg += f"Test:  ~{target_test_ratio*100:.1f}%"
        
        messagebox.showinfo("Done", final_msg)
    
    set_ui_state("normal")

# --- UI Elemanları ---
dataset_path = StringVar(value=RAW_DATA_DIR)
train_var = StringVar(value="80")
val_var = StringVar(value="10")
test_var = StringVar(value="10")
width_var = StringVar(value="224")
height_var = StringVar(value="224")
output_folder_var = StringVar(value="prepared_" + str(random.randint(100, 999)))
aug_count_var = StringVar(value="5") # Her resim için 5 yeni versiyon 

Label(root, text="Dataset Folder:", font=("Arial", 11)).pack(pady=5)
dataset_entry = Entry(root, textvariable=dataset_path, width=45)
dataset_entry.pack()
browse_button = Button(root, text="📂 Browse", command=select_folder)
browse_button.pack(pady=5)

Label(root, text="Data Split Ratios (%)", font=("Arial", 11, "bold")).pack(pady=10)
frame_ratio = Frame(root)
frame_ratio.pack()
Label(frame_ratio, text="Train").grid(row=0, column=0)
train_entry = Entry(frame_ratio, textvariable=train_var, width=5)
train_entry.grid(row=0, column=1)
Label(frame_ratio, text="Val").grid(row=0, column=2)
val_entry = Entry(frame_ratio, textvariable=val_var, width=5)
val_entry.grid(row=0, column=3)
Label(frame_ratio, text="Test").grid(row=0, column=4)
test_entry = Entry(frame_ratio, textvariable=test_var, width=5)
test_entry.grid(row=0, column=5)

Label(root, text="Image Size (px)", font=("Arial", 11, "bold")).pack(pady=10)
frame_size = Frame(root)
frame_size.pack()
Label(frame_size, text="Width").grid(row=0, column=0)
width_entry = Entry(frame_size, textvariable=width_var, width=5)
width_entry.grid(row=0, column=1)
Label(frame_size, text="Height").grid(row=0, column=2)
height_entry = Entry(frame_size, textvariable=height_var, width=5)
height_entry.grid(row=0, column=3)

Label(root, text="Output Folder Name:", font=("Arial", 11, "bold")).pack(pady=10)
output_entry = Entry(root, textvariable=output_folder_var, width=25)
output_entry.pack(pady=(0, 10))

Label(root, text="Image Processing Options", font=("Arial", 11, "bold")).pack(pady=10)
frame_opts = Frame(root)
frame_opts.pack()

normalize_var = BooleanVar(value=False)
grayscale_var = BooleanVar(value=False)
augment_var = BooleanVar(value=False)

normalize_check = Checkbutton(frame_opts, text="Normalize (0–1)", variable=normalize_var, state="disabled") # Diske kaydederken anlamsız
grayscale_check = Checkbutton(frame_opts, text="Grayscale", variable=grayscale_var)
augment_check = Checkbutton(frame_opts, text="Augmentation (Train Only)", variable=augment_var)

normalize_check.grid(row=0, column=0, padx=10)
grayscale_check.grid(row=0, column=1, padx=10)
augment_check.grid(row=0, column=2, padx=10)

# --- YENİ UI ELEMANI ---
Label(root, text="Augmentations per Image (if checked):", font=("Arial", 10)).pack(pady=(10, 0))
aug_count_entry = Entry(root, textvariable=aug_count_var, width=5)
aug_count_entry.pack()


button_frame = Frame(root)
button_frame.pack(pady=25)

prepare_button = Button(button_frame, text="Prepare Dataset", bg="#4CAF50", fg="white",
                        font=("Arial", 11, "bold"), command=start_preprocessing_thread)
prepare_button.grid(row=0, column=0, padx=10)

cancel_button = Button(button_frame, text="Cancel", bg="#f44336", fg="white",
                       font=("Arial", 11, "bold"), command=cancel_preprocessing)
cancel_button.grid(row=0, column=1, padx=10)

progress_label = Label(root, text="Progress: Waiting...", fg="gray")
progress_label.pack(pady=5)
progress_bar = ttk.Progressbar(root, length=400, mode='determinate')
progress_bar.pack(pady=10)

Label(root, text="Output will be saved in 'data/prepared-data/<folder_name>'", wraplength=460, fg="gray").pack()

root.mainloop()