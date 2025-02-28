## **Fake News Detection using Machine Learning**  

### 📌 **Project Overview**  
The **Fake News Detection System** is a machine learning-based solution to classify news articles as **real or fake**. The project aims to address the growing issue of misinformation by using various machine learning algorithms for classification.  

---

### 🔍 **Problem Statement**  
Fake news has become a serious issue, influencing public opinion and democracy. The rapid spread of misinformation, especially on social media, necessitates a reliable system to detect and prevent fake news. The project focuses on developing **automated machine learning models** to accurately classify news articles as **real** or **fake**.

---

### 🗃 **Dataset**  
- The dataset consists of **news articles labeled as real or fake**.
- Features include:
  - **Title**: News headline  
  - **Text**: Full article content  
  - **Label**: Binary classification (1 = Fake, 0 = Real)  
- The dataset is split into **training (70%)** and **testing (30%)** sets.

---

### ⚙ **Technologies Used**  
- **Programming Language**: Python  
- **Libraries**:
  - **Natural Language Processing (NLP)**: `NLTK`, `spaCy`
  - **Machine Learning**: `Scikit-learn`, `XGBoost`
  - **Data Processing**: `Pandas`, `NumPy`
  - **Model Evaluation**: `Matplotlib`, `Seaborn`
- **Algorithms Implemented**:
  - Random Forest  
  - Support Vector Machine (SVM)  
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  

---

### 🚀 **Project Workflow**  
1. **Data Preprocessing**:
   - Remove stopwords and punctuation  
   - Tokenization & Lemmatization  
   - Convert text to numerical format using TF-IDF  

2. **Exploratory Data Analysis (EDA)**:
   - Analyzed the distribution of real vs. fake news  
   - Visualized word frequency using graphs  

3. **Model Training & Evaluation**:
   - Trained **Random Forest, SVM, Logistic Regression, and KNN** models  
   - Evaluated using accuracy, precision, recall, and F1-score  

4. **Prediction & Testing**:
   - Deployed models to classify new/unseen news articles  

---

### 📌 **How to Run the Project**  
#### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/fake-news-detection.git  
cd fake-news-detection
```

#### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt  
```

#### 3️⃣ Run the Model  
```bash
python fake_news_detection.py
```

#### 4️⃣ Test with a Sample News Article  
```python
from model import predict_news  
print(predict_news("Breaking: Government announces new policies on AI development."))
```

---

### 📊 **Results & Model Performance**  
- **Naïve Bayes Classifier**: **89.58% accuracy**  
- **Random Forest Classifier**: **91.23% accuracy**  
- **SGD Classifier**: **92.65% accuracy** (Best model)  
- **K-Nearest Neighbors (KNN)**: **77.11% accuracy**  

| **Model** | **Accuracy** |
|-----------|------------|
| Naïve Bayes | 89.58% |
| Random Forest | 91.23% |
| SGD Classifier | **92.65%** |
| KNN | 77.11% |

---

### 📌 **Future Improvements**  
✅ Train on larger and more diverse datasets  
✅ Implement deep learning models (LSTMs, BERT)  
✅ Deploy as a **web application** using Flask or Django  

---

### 📩 **Contributing**  
Want to improve this project? Feel free to submit a **Pull Request** or raise an **Issue**!  

---

### 📜 **License**  
This project is open-source under the **MIT License**.  
