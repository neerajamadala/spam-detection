# spam-detection

Here’s a clear and concise **Project Overview** for your **Spam Detection using Machine Learning** project:

---

## 📌 **Project Overview: Spam Detection Using Machine Learning**

### **Objective:**

To build a machine learning model that accurately classifies SMS messages as **spam** or **ham** (not spam).

---

### **Dataset Used:**

* **Source:** SMS Spam Collection Dataset (UCI Machine Learning Repository)
* **Size:** \~5,500 messages
* **Classes:**

  * `ham` (legitimate message)
  * `spam` (unwanted/ad message)

---

### **Data Preprocessing Steps:**

* Label Encoding:

  * `ham` → 0
  * `spam` → 1
* Text cleaning:

  * Lowercasing, removing punctuation, stopwords
* Feature Extraction:

  * **TF-IDF Vectorizer** used to convert text into numeric format

---

### **Models Implemented:**

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest Classifier**

---

### **Techniques Used:**

* **SMOTE** (Synthetic Minority Over-sampling Technique):
  Balanced the dataset due to spam class imbalance.
* **GridSearchCV**:
  Performed hyperparameter tuning for improved accuracy.

---

### **Evaluation Metrics:**

* **Accuracy**
* **Confusion Matrix**
* **ROC Curve & AUC**
* **Precision, Recall, F1-Score**

---

### **Best Performing Model:**

* **Random Forest** with SMOTE and hyperparameter tuning:

  * **Accuracy:** 99.27%
  * **AUC Score:** 0.999
  * Outperformed SVM and Logistic Regression


