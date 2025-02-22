# Chapter 14: Deep Computer Vision Using Convolutional Neural Networks

## **Foundations of Convolutional Neural Networks**

Convolutional Neural Networks (CNNs) are specialized architectures designed for processing visual data. Unlike fully connected neural networks, CNNs exploit spatial hierarchies in images using convolutional layers, which extract local features like edges, textures, and shapes.

### **Key Components of CNNs**

1. **Convolutional Layers**:
    - Convolutional layers apply filters (kernels) to extract features from input images.
    - The output of a convolutional layer is called a **feature map**:
    
    $$
    \text{Feature Map} = \text{Input Image} * \text{Filter}
    
    $$
    
    - Filters detect patterns such as edges, corners, and textures.
        
        Example:
        
        ```python
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(64, 64, 3)))
        ```
        
2. **Pooling Layers**:
    - Pooling reduces the spatial dimensions of feature maps while retaining important information.
    - Common types:
        - **Max Pooling**: *Selects the maximum value in each window.*
            
            ```python
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            ```
            
        - **Average Pooling**: Computes the average value in each window.
3. **Fully Connected Layers:**
    - After extracting hierarchical features, fully connected layers perform classification or regression.
        
        Example:
        
        ```python
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(10, activation="softmax"))  # Output layer
        ```
        
4. **Activation Functions**:
    - ReLU (Rectified Linear Unit) introduces non-linearity:
    
    $$
    f(x) = \max(0, x)
    $$
    
    - *Implementation:*
        
        ```python
        model.add(tf.keras.layers.Activation("relu"))
        ```
        

---

## **Building and Training CNNs**

This section provides hands-on guidance for implementing CNNs using Keras.

### **Step-by-Step CNN Construction**

1. **Define the Model**:
Stack convolutional, pooling, and dense layers:
    
    ```python
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    ```
    
2. **Compile the Model**:
Specify the loss function, optimizer, and metrics:
    
    ```python
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    ```
    
3. **Train the Model**:
Fit the model to the training data:
    
    ```python
    history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
    ```
    

### **Data Augmentation**

Data augmentation improves generalization by creating variations of the training data:

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

augmented_images = data_augmentation(images)
```

### **Transfer Learning**

Leverage pre-trained models like VGG16, ResNet, or Inception for faster training:

```python
base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Freeze base model

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
```

---

## **Advanced Topics in Computer Vision**

This section explores advanced techniques that extend CNN capabilities.

### **Object Detection**

- Algorithms like YOLO (You Only Look Once) and Faster R-CNN identify and localize objects in images.
- Example Use Case: Detecting pedestrians in autonomous driving systems.

### **Semantic Segmentation**

- Pixel-wise classification assigns labels to each pixel in an image.
- Techniques like U-Net and Mask R-CNN are widely used in medical imaging and autonomous vehicles.
    
    Example:
    
    ```python
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")  # Output segmentation mask
    ])
    ```
    

### **Attention Mechanisms**

- Attention mechanisms focus on relevant regions of an image, improving performance in tasks like object detection and segmentation.

---

## **Practical Implementation**

The chapter includes hands-on examples for real-world applications.

**Example:**

*MNIST Classification*

```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

*Transfer Learning for Medical Imaging*

```python
base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Tumor detection
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
```

---

## **Applications and Use Cases**

CNNs have transformative applications across industries:

- **Medical Imaging**: Detecting tumors in brain scans or diagnosing diseases from X-rays.
- **Autonomous Vehicles**: Identifying pedestrians, vehicles, and road signs in real-time.
- **Retail and E-commerce**: Automating product categorization and recommendation systems.

---

## **Conclusion**

This chapter equips readers with the skills to build, train, and deploy CNNs for computer vision tasks. By combining theoretical insights with practical implementation, you can tackle challenges like image classification, object detection, and semantic segmentation.

### **Key Takeaways**

- CNNs use convolutional layers to extract hierarchical features from images.
- Data augmentation and transfer learning improve model performance and reduce training time.
- Advanced techniques like object detection and semantic segmentation extend CNN capabilities.
- Hands-on examples reinforce theoretical concepts and enable real-world applications.

# فصل 14: بینایی کامپیوتری عمیق با استفاده از شبکه‌های عصبی کانولوشنی

## **مبانی شبکه‌های عصبی کانولوشنی**

شبکه‌های عصبی کانولوشنی (CNN) معماری‌های تخصصی هستند که برای پردازش داده‌های تصویری طراحی شده‌اند. برخلاف شبکه‌های عصبی کاملاً متصل، CNNها از سلسله مراتب فضایی در تصاویر با استفاده از لایه‌های کانولوشنی استفاده می‌کنند که ویژگی‌های محلی مانند لبه‌ها، بافت‌ها و اشکال را استخراج می‌کنند.

### **اجزای اصلی CNNها**

1. **لایه‌های کانولوشنی**:
    - لایه‌های کانولوشنی فیلترها (kernels) را برای استخراج ویژگی‌ها از تصاویر ورودی اعمال می‌کنند.
    - خروجی یک لایه کانولوشنی، **نقشه ویژگی (Feature Map)** نامیده می‌شود:
    
    $$
    \text{Feature Map} = \text{Input Image} * \text{Filter}
    $$
    
2. **لایه‌های Pooling**:
    - Pooling ابعاد فضایی نقشه‌های ویژگی را کاهش می‌دهد در حالی که اطلاعات مهم را حفظ می‌کند.
    - انواع رایج:
        - **Max Pooling**: *حداکثر مقدار را در هر پنجره انتخاب می‌کند.*
        - **Average Pooling**: میانگین مقادیر را در هر پنجره محاسبه می‌کند.
3. **لایه‌های کاملاً متصل (Fully Connected):**
    - پس از استخراج ویژگی‌های سلسله مراتبی، لایه‌های کاملاً متصل طبقه‌بندی یا رگرسیون را انجام می‌دهند.
4. **توابع فعال‌سازی**:
    - ReLU (Rectified Linear Unit) غیرخطی بودن را معرفی می‌کند:

## **ساخت و آموزش CNNها**

این بخش راهنمایی عملی برای پیاده‌سازی CNNها با استفاده از Keras ارائه می‌دهد.

### **ساخت گام به گام CNN**

### **Data Augmentation**

Data Augmentation با ایجاد تغییرات در داده‌های آموزشی، عمومیت مدل را بهبود می‌بخشد.

### **Transfer Learning**

استفاده از مدل‌های از پیش آموزش دیده مانند VGG16، ResNet یا Inception برای آموزش سریع‌تر.

## **موضوعات پیشرفته در بینایی کامپیوتر**

این بخش تکنیک‌های پیشرفته‌ای که قابلیت‌های CNN را گسترش می‌دهند، بررسی می‌کند.

### **تشخیص اشیا**

- الگوریتم‌هایی مانند YOLO و Faster R-CNN اشیا را در تصاویر شناسایی و مکان‌یابی می‌کنند.
- مثال کاربردی: تشخیص عابران پیاده در سیستم‌های رانندگی خودکار.

### **قطعه‌بندی معنایی (Semantic Segmentation)**

- طبقه‌بندی پیکسل به پیکسل که به هر پیکسل در تصویر برچسب اختصاص می‌دهد.
- تکنیک‌هایی مانند U-Net و Mask R-CNN به طور گسترده در تصویربرداری پزشکی و خودروهای خودران استفاده می‌شوند.

### **مکانیزم‌های توجه (Attention Mechanisms)**

- مکانیزم‌های توجه بر مناطق مرتبط تصویر تمرکز می‌کنند و عملکرد را در وظایفی مانند تشخیص اشیا و قطعه‌بندی بهبود می‌بخشند.

## **پیاده‌سازی عملی**

این فصل شامل مثال‌های عملی برای کاربردهای دنیای واقعی است.

### **کاربردها و موارد استفاده**

CNNها کاربردهای تحول‌آفرینی در صنایع مختلف دارند:

- **تصویربرداری پزشکی**: تشخیص تومورها در اسکن‌های مغزی یا تشخیص بیماری‌ها از عکس‌های رادیولوژی.
- **خودروهای خودران**: شناسایی عابران پیاده، خودروها و علائم راهنمایی و رانندگی در زمان واقعی.
- **خرده‌فروشی و تجارت الکترونیک**: خودکارسازی دسته‌بندی محصولات و سیستم‌های پیشنهادی.

### **نتیجه‌گیری**

این فصل خوانندگان را با مهارت‌های لازم برای ساخت، آموزش و استقرار CNNها برای وظایف بینایی کامپیوتر مجهز می‌کند. با ترکیب بینش‌های نظری با پیاده‌سازی عملی، می‌توانید چالش‌هایی مانند طبقه‌بندی تصویر، تشخیص اشیا و قطعه‌بندی معنایی را حل کنید.

### **نکات کلیدی**

- CNNها از لایه‌های کانولوشنی برای استخراج ویژگی‌های سلسله مراتبی از تصاویر استفاده می‌کنند.
- Data Augmentation و Transfer Learning عملکرد مدل را بهبود می‌بخشند و زمان آموزش را کاهش می‌دهند.
- تکنیک‌های پیشرفته مانند تشخیص اشیا و قطعه‌بندی معنایی قابلیت‌های CNN را گسترش می‌دهند.
- مثال‌های عملی مفاهیم نظری را تقویت می‌کنند و کاربردهای دنیای واقعی را امکان‌پذیر می‌سازند.

با تسلط بر این مفاهیم، شما برای حل مسائل پیچیده بینایی کامپیوتر آماده خواهید بود.