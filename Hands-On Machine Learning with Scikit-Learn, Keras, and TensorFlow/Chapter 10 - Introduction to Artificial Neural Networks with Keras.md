# Chapter 10: Introduction to Artificial Neural Networks with Keras

## **Introduction to Neural Networks**

Artificial Neural Networks (ANNs) are computational models inspired by the human brain's biological neural networks. They consist of layers of interconnected artificial neurons that process input data to learn patterns and make predictions.

### **Key Concepts**

- **Neurons**: The basic unit of a neural network. Each neuron applies a weighted sum to its inputs and passes the result through an activation function:

$$
z=w⋅x+b
\newline
a=f(z)
$$

*where **$w$** is the weight vector, **$x$** is the input vector, $b$ is the bias, and $f(z)$ is the activation function.*

- **Activation Functions**: *Non-linear functions that introduce non-linearity into the model. Common activation functions include:*
    - Sigmoid:
    
    $$
    f(z)=\frac{1}{1+e^{−z}}
    $$
    
    - ReLU (*Rectified Linear Unit*):
    
    $$
    f(z)=max(0,z)
    $$
    
    - Softmax: *Used for multi-class classification, normalizes outputs into probabilities.*
- **Layers**: Neurons are organized into layers:
    - **Input Layer**: *Receives raw input data.*
    - **Hidden Layers**: *Intermediate layers that extract features.*
    - **Output Layer**: *Produces the final prediction.*

<aside>
💡

### **Why Neural Networks?**

Neural networks excel at modeling complex, non-linear relationships in data, making them suitable for tasks like image recognition, natural language processing, and more.

</aside>

---

## **Building Neural Networks with Keras**

Keras is a high-level API for building and training neural networks. It is integrated into TensorFlow, making it user-friendly and efficient.

### **Steps to Build a Neural Network**

1. **Define the Model** :
    
    Use **`Sequential`** for simple models or **`Functional API`** for more complex architectures.
    
    ```python
    from tensorflow import keras
    
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),  # Input layer
        keras.layers.Dense(300, activation="relu"),  # Hidden layer
        keras.layers.Dense(100, activation="relu"),  # Hidden layer
        keras.layers.Dense(10, activation="softmax")  # Output layer
    ])
    ```
    
2. **Compile the Model**:
    
    Specify the loss function, optimizer, and metrics.
    
    ```python
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",  # Stochastic Gradient Descent
                  metrics=["accuracy"])
    ```
    
3. **Train the Model**:
    
    Fit the model to the training data.
    
    ```python
    history = model.fit(X_train, y_train, epochs=30, validation_split=0.1)
    ```
    
4. **Evaluate the Model**:
    
    Assess performance on test data.
    
    ```python
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy}")
    ```
    

**Example:**

*MNIST Dataset*

*The chapter uses the MNIST dataset (handwritten digit images) to demonstrate how to build a neural network for classification:*

```python
from tensorflow.keras.datasets import mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# Normalize the data
X_train, X_valid, X_test = X_train / 255.0, X_valid / 255.0, X_test / 255.0
```

---

## **Training Deep Neural Networks**

Training deep neural networks involves addressing several challenges:

### **Challenges**

1. **Vanishing/Exploding Gradients** :
    - Gradients can become too small (vanish) or too large (explode) during backpropagation, hindering learning.
    - Solutions:
        - **Weight Initialization** : Use techniques like He initialization for ReLU activations.
        - **Normalization** : Apply batch normalization to stabilize training.
            
            ```python
            keras.layers.BatchNormalization()
            ```
            
2. **Optimization Algorithms** :
    - **SGD (Stochastic Gradient Descent)**: Basic but slow.
    - **Adam**: Combines momentum and adaptive learning rates for faster convergence.
        
        ```python
        optimizer="adam"
        ```
        
3. **Regularization** :
    - Prevent overfitting using dropout layers:
        
        ```python
        keras.layers.Dropout(rate=0.2)
        ```
        

---

## **Evaluating Model Performance**

Evaluating neural networks requires careful consideration of metrics and techniques:

### **Metrics**

- **Accuracy**: Fraction of correct predictions.
- **Confusion Matrix**: Shows true vs. predicted classifications.
- **Precision, Recall, F1-Score**: Useful for imbalanced datasets.

### **Cross-Validation**

Use cross-validation to assess model robustness:

```python
from sklearn.model_selection import cross_val_score

cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
```

---

## **Practical Tips for Training Neural Networks**

- **Data Preprocessing** :
    - Normalize inputs to improve convergence.
    - Augment data (e.g., rotate, flip images) to increase diversity.
- **Hyperparameter Tuning** :
    - Experiment with learning rates, batch sizes, and network architectures.
- **Early Stopping** :
Stop training when validation performance plateaus to prevent overfitting:
    
    ```python
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ```
    

---

## Conclusion

This chapter serves as a foundational guide to neural networks, bridging traditional machine learning techniques with deep learning. By understanding the structure of ANNs, implementing models with Keras, and addressing training challenges, readers gain the skills to tackle real-world problems.

### **Key Takeaways**

- Neural networks are powerful tools for modeling complex data.
- Keras simplifies the process of building and training models.
- Addressing challenges like vanishing gradients and overfitting is crucial for success.
- Evaluation metrics and techniques ensure reliable model performance.

# فصل ۱۰: آشنایی با شبکه‌های عصبی مصنوعی با Keras

## **مقدمه‌ای بر شبکه‌های عصبی**

شبکه‌های عصبی مصنوعی (ANNs) مدل‌های محاسباتی هستند که از شبکه‌های عصبی مغز انسان الهام گرفته‌اند. آنها از لایه‌هایی از نورون‌های مصنوعی متصل به هم تشکیل شده‌اند که داده‌های ورودی را برای یادگیری الگوها و انجام پیش‌بینی پردازش می‌کنند.

### **مفاهیم کلیدی**

- **Neurons**: واحد پایه یک شبکه عصبی. هر نورون مجموع وزن‌دار ورودی‌های خود را محاسبه کرده و نتیجه را از طریق یک تابع فعال‌سازی عبور می‌دهد:

$$
z=w⋅x+b
\newline
a=f(z)
$$

*که در آن **w** بردار وزن، **x** بردار ورودی، b بایاس و f(z) تابع فعال‌سازی است.*

- **توابع فعال‌سازی (Activation Functions)**: *توابع غیرخطی که غیرخطی بودن را به مدل اضافه می‌کنند. توابع فعال‌سازی رایج عبارتند از:*
    - Sigmoid:
    
    $$
    f(z)=\frac{1}{1+e^{−z}}
    $$
    
    - ReLU (*واحد یکسوساز خطی*):
    
    $$
    f(z)=max(0,z)
    $$
    
    - Softmax: *برای دسته‌بندی چند کلاسه استفاده می‌شود و خروجی‌ها را به احتمالات تبدیل می‌کند.*
- **لایه‌ها (Layers)**: نورون‌ها در لایه‌ها سازماندهی می‌شوند:
    - **لایه ورودی (Input Layer)**: *داده‌های خام ورودی را دریافت می‌کند.*
    - **لایه‌های پنهان (Hidden Layers)**: *لایه‌های میانی که ویژگی‌ها را استخراج می‌کنند.*
    - **لایه خروجی (Output Layer)**: *پیش‌بینی نهایی را تولید می‌کند.*

<aside>
💡 **چرا شبکه‌های عصبی؟**

شبکه‌های عصبی در مدل‌سازی روابط پیچیده و غیرخطی در داده‌ها عملکرد عالی دارند و آنها را برای وظایفی مانند تشخیص تصویر، پردازش زبان طبیعی و موارد دیگر مناسب می‌سازد.

</aside>

---

## **ساخت شبکه‌های عصبی با Keras**

Keras یک API سطح بالا برای ساخت و آموزش شبکه‌های عصبی است که در TensorFlow یکپارچه شده و کاربرپسند و کارآمد است.

### **مراحل ساخت یک شبکه عصبی**

1. **تعریف مدل**:از **Sequential** برای مدل‌های ساده یا **Functional API** برای معماری‌های پیچیده‌تر استفاده کنید.
    
    ```python
    from tensorflow import keras
    
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),  # Input layer
        keras.layers.Dense(300, activation="relu"),  # Hidden layer
        keras.layers.Dense(100, activation="relu"),  # Hidden layer
        keras.layers.Dense(10, activation="softmax")  # Output layer
    ])
    ```
    
2. **کامپایل مدل**:تابع خطا، بهینه‌ساز و معیارها را مشخص کنید.
    
    ```python
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",  # Stochastic Gradient Descent
                  metrics=["accuracy"])
    ```
    
3. **آموزش مدل**:مدل را با داده‌های آموزشی تطبیق دهید.
    
    ```python
    history = model.fit(X_train, y_train, epochs=30, validation_split=0.1)
    ```
    
4. **ارزیابی مدل**:عملکرد را روی داده‌های آزمایشی ارزیابی کنید.
    
    ```python
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy}")
    ```
    

**مثال:**

*مجموعه داده MNIST*

این فصل از مجموعه داده MNIST (تصاویر ارقام دست‌نویس) برای نمایش نحوه ساخت یک شبکه عصبی برای دسته‌بندی استفاده می‌کند:

```python
from tensorflow.keras.datasets import mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# Normalize the data
X_train, X_valid, X_test = X_train / 255.0, X_valid / 255.0, X_test / 255.0
```

---

## **آموزش شبکه‌های عصبی عمیق**

آموزش شبکه‌های عصبی عمیق شامل مقابله با چندین چالش است:

### **چالش‌ها**

1. **گرادیان‌های محو شونده/منفجر شونده**:
    - گرادیان‌ها می‌توانند در طول پس‌انتشار خیلی کوچک (محو) یا خیلی بزرگ (منفجر) شوند که یادگیری را مختل می‌کند.
    - راه‌حل‌ها:
        - **مقداردهی اولیه وزن‌ها**: استفاده از تکنیک‌هایی مانند مقداردهی He برای توابع فعال‌سازی ReLU.
        - **نرمال‌سازی**: اعمال نرمال‌سازی دسته‌ای برای پایدارسازی آموزش.
            
            ```python
            keras.layers.BatchNormalization()
            ```
            
2. **الگوریتم‌های بهینه‌سازی**:
    - **SGD (گرادیان نزولی تصادفی)**: ساده اما کند.
    - **Adam**: ترکیب مومنتوم و نرخ یادگیری تطبیقی برای همگرایی سریع‌تر.
        
        ```python
        optimizer="adam"
        ```
        
3. **منظم‌سازی**:
    - جلوگیری از بیش‌برازش با استفاده از لایه‌های Dropout:
        
        ```python
        keras.layers.Dropout(rate=0.2)
        ```
        

---

## **ارزیابی عملکرد مدل**

ارزیابی شبکه‌های عصبی نیازمند بررسی دقیق معیارها و تکنیک‌ها است:

### **معیارها**

- **دقت (Accuracy)**: نسبت پیش‌بینی‌های صحیح.
- **ماتریس درهم‌ریختگی (Confusion Matrix)**: دسته‌بندی‌های واقعی در مقابل پیش‌بینی شده را نشان می‌دهد.
- **Precision، Recall، F1-Score**: برای مجموعه داده‌های نامتوازن مفید هستند.

### **اعتبارسنجی متقاطع**

از اعتبارسنجی متقاطع برای ارزیابی استحکام مدل استفاده کنید:

```python
from sklearn.model_selection import cross_val_score

cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
```

---

## **نکات عملی برای آموزش شبکه‌های عصبی**

- **پیش‌پردازش داده**:
    - نرمال‌سازی ورودی‌ها برای بهبود همگرایی.
    - افزایش داده (مثلاً چرخش، برگرداندن تصاویر) برای افزایش تنوع.
- **تنظیم ابرپارامترها**:
    - آزمایش با نرخ‌های یادگیری، اندازه دسته‌ها و معماری‌های شبکه.
- **توقف زودهنگام**:
توقف آموزش هنگامی که عملکرد اعتبارسنجی به ثبات می‌رسد برای جلوگیری از بیش‌برازش:
    
    ```python
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ```
    

---

## نتیجه‌گیری

این فصل به عنوان یک راهنمای پایه برای شبکه‌های عصبی عمل می‌کند و تکنیک‌های یادگیری ماشین سنتی را با یادگیری عمیق پیوند می‌دهد. با درک ساختار ANNها، پیاده‌سازی مدل‌ها با Keras و مقابله با چالش‌های آموزش، خوانندگان مهارت‌های لازم برای حل مسائل دنیای واقعی را کسب می‌کنند.

### **نکات کلیدی**

- شبکه‌های عصبی ابزارهای قدرتمندی برای مدل‌سازی داده‌های پیچیده هستند.
- Keras فرآیند ساخت و آموزش مدل‌ها را ساده می‌کند.
- مقابله با چالش‌هایی مانند گرادیان‌های محو شونده و بیش‌برازش برای موفقیت ضروری است.
- معیارها و تکنیک‌های ارزیابی، عملکرد قابل اعتماد مدل را تضمین می‌کنند.

با تسلط بر این مفاهیم، شما برای کشف موضوعات پیشرفته در یادگیری عمیق آماده خواهید بود.