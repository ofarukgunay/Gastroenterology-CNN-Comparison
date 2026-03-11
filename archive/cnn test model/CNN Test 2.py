import tensorflow as tf
import matplotlib.pyplot as plt

# Görsel ayarlar
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='gray')

# Kenar tespiti filtresi (Laplacian benzeri)
kernel = tf.constant(
    [[-1, -1, -1],
     [-1,  8, -1],
     [-1, -1, -1]], 
    dtype=tf.float32
)

# Görüntüyü oku (grayscale)
image = tf.io.read_file('Resmi Foto.JPG')
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[300, 300])

# Orijinal görüntü göster
img = tf.squeeze(image).numpy()
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Gray Scale Image')
plt.show()

# Normalize ve şekil düzenleme
image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # [0,1] aralığına
image = tf.expand_dims(image, axis=0)  # batch ekle -> [1, H, W, C]
kernel = tf.reshape(kernel, [3, 3, 1, 1])  # [filter_h, filter_w, in_ch, out_ch]

# --- 1. Konvolüsyon ---
conv_output = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=[1, 1, 1, 1],
    padding='SAME'
)

# Görselleştirme için normalize et
conv_img = tf.squeeze(conv_output).numpy()
conv_img = (conv_img - conv_img.min()) / (conv_img.max() - conv_img.min() + 1e-8)

# --- 2. Aktivasyon (ReLU) ---
relu_output = tf.nn.relu(conv_output)
relu_img = tf.squeeze(relu_output).numpy()

# --- 3. Max Pooling ---
pool_output = tf.nn.pool(
    input=relu_output,
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME'
)
pool_img = tf.squeeze(pool_output).numpy()

# --- Çıktıları görselleştir ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(conv_img, cmap='gray')
plt.axis('off')
plt.title('Convolution')

plt.subplot(1, 3, 2)
plt.imshow(relu_img, cmap='gray')
plt.axis('off')
plt.title('Activation (ReLU)')

plt.subplot(1, 3, 3)
plt.imshow(pool_img, cmap='gray')
plt.axis('off')
plt.title('Pooling (Max)')

plt.show()
