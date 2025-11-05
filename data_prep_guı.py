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
        train_ratio = int(train_var.get()) / 100
        val_ratio = int(val_var.get()) / 100
        test_ratio = int(test_var.get()) / 100
        width = int(width_var.get())
        height = int(height_var.get())
        num_augs = int(aug_count_var.get())
    except ValueError:
        messagebox.showerror("Error", "Invalid ratio, image size, or augmentation count values.")
        set_ui_state("normal")
        return
    
    if (train_ratio + val_ratio + test_ratio) != 1.0:
        messagebox.showerror("Error", "Ratios must sum to 100.")
        set_ui_state("normal")
        return

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
    processed = 0

    progress_bar["maximum"] = total_images
    progress_bar["value"] = 0
    progress_label.config(text="Processing started... (Splitting files)")

    for c in classes:
        if cancel_processing: break
        
        imgs = [os.path.join(folder, c, f) for f in os.listdir(os.path.join(folder, c)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not imgs: continue
        
        train_files, temp = train_test_split(imgs, test_size=(1 - train_ratio), random_state=42)
        val_files, test_files = train_test_split(temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

        for split_name, file_list in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            if cancel_processing: break
            
            split_dir = os.path.join(output_dir, split_name, c)
            os.makedirs(split_dir, exist_ok=True)
            
            for fpath in file_list:
                if cancel_processing: break
                
                try:
                    # Görüntüyü oku (PIL ile okuyup RGB'ye çevirmek en güvenlisi)
                    img = Image.open(fpath).convert("RGB")
                    img = img.resize((width, height))

                    # Grayscale ve Normalizasyon seçenekleri
                    if grayscale_var.get():
                        img = img.convert("L").convert("RGB") # 3 Kanallı Grayscale
                    
                    if normalize_var.get():
                        # Bu seçenek artık diske kaydederken mantıklı değil.
                        # Normalizasyon eğitim sırasında (online) yapılmalıdır.
                        # Ama kodda olduğu için bırakıyoruz.
                        arr = np.asarray(img).astype(np.float32) / 255.0
                        img = Image.fromarray((arr * 255).astype(np.uint8))

                    if split_name == "train" and augment_var.get():
                        # Sadece EĞİTİM seti ve Augmentation seçiliyse 1-N çoğaltma yap
                        
                        # 1. Orijinal (sadece resize/gray/norm) görüntüyü kaydet
                        img.save(os.path.join(split_dir, os.path.basename(fpath)))
                        
                        # PIL görüntüsünü OpenCV/Numpy formatına çevir
                        img_np = np.array(img)
                        
                        # N adet yeni versiyon oluştur
                        for i in range(num_augs):
                            augmented_np = albumentations_pipeline(image=img_np)['image']
                            
                            # Yeni dosya adı oluştur
                            base_name, ext = os.path.splitext(os.path.basename(fpath))
                            new_filename = f"{base_name}_aug_{i}{ext}"
                            save_path = os.path.join(split_dir, new_filename)
                            
                            # OpenCV (BGR) formatında kaydet
                            cv2.imwrite(save_path, cv2.cvtColor(augmented_np, cv2.COLOR_RGB2BGR))
                    
                    else:
                        # VAL/TEST setleri veya Augmentation kapalıysa 1-1 kaydet
                        img.save(os.path.join(split_dir, os.path.basename(fpath)))

                    processed += 1
                    progress_bar["value"] = processed
                    progress_label.config(text=f"Processing {processed}/{total_images} input images...")
                    root.update_idletasks()

                except Exception as e:
                    print(f"Error: {fpath} could not be processed ({e})")
        
    if cancel_processing:
        if current_output_dir and os.path.exists(current_output_dir):
            shutil.rmtree(current_output_dir)
        progress_label.config(text="Processing cancelled ❌")
        messagebox.showinfo("Cancelled", "Dataset preparation was cancelled by the user.")
    else:
        progress_label.config(text="Completed ✅")
        messagebox.showinfo("Done", f"Dataset prepared successfully!\nSaved in:\n{output_dir}")
    
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