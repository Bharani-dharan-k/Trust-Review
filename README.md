# 🔍 TrustReview - Unmasking Fake Reviews with AI

![TrustReview Banner](demo/Screenshot%202025-04-13%20141705.png)

## 🧩 Problem Statement

**Unmasking Fake Reviews with AI**

People frequently rely on online reviews when deciding to purchase a product, visit a place, or try a new restaurant. However, many reviews are fake or misleading, making it difficult to trust them. So the challenge is to develop an **AI-powered system** that detects **fake reviews** by analyzing **language patterns**, **user review history**, **sentiment analysis**, and other relevant factors to help users make more informed decisions.

---

## 🧠 About the Project

**TrustReview** is an intelligent AI system designed to detect fake reviews. It uses techniques like sentiment analysis, language detection, and machine learning to classify reviews as **genuine or fake**. Our solution supports **multilingual reviews** and provides **real-time predictions** for users and businesses to identify review authenticity.

---

## 🖼️ Preview

![TrustReview UI Screenshot](demo/Screenshot%202025-04-13%20141836.png)

---

## 🛠️ Tech Stack

### Frontend
- ✅ React.js
- ✅ Tailwind CSS
- ✅ Axios for API Integration

### Backend
- ✅ Python Flask
- ✅ XGBoost ML Model
- ✅ TextBlob (Sentiment Analysis)
- ✅ Langdetect (Language Detection)
- ✅ Googletrans (Translation)

---

## ⚙️ Features

- ✔️ Detects whether a review is **Fake** or **Genuine**
- 🌐 Supports **multiple languages**
- 🔁 Automatically **translates** reviews to English if needed
- 💬 Performs **sentiment analysis**
- ⚡ Instant prediction results with a user-friendly UI
- 📈 Trained on multilingual datasets using **XGBoost**

---

## 📦 Folder Structure

```
trustreview/
├── backend/
│   ├── app.py
│   ├── model/
│   ├── utils/
│   └── requirements.txt
├── frontend/
│   ├── public/
│   └── src/
├── images/
│   ├── banner.png
│   └── preview.png
├── README.md
```

---

## 🖥️ How to Run the Project Locally

### 🔹 Clone the Repository

```bash
git clone https://github.com/yourusername/trustreview.git
cd trustreview
```

### 🔹 Backend Setup (Flask)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

> Backend will run on `http://localhost:5000`

### 🔹 Frontend Setup (React)

```bash
cd frontend
npm install
npm start
```

> Frontend will run on `http://localhost:3000`

---

## 📊 Example Output

**Input Review:**  
> "This is the best thing I've ever bought! Highly recommended!!"

**Output:**  
- ❌ **Fake Review Detected**  
- 💬 **Sentiment:** Positive  
- 🌍 **Language:** English

---

## 🌱 Future Enhancements

- 📌 Add Chrome extension for review verification
- 🔗 Integrate with Amazon, Flipkart, and Google Reviews
- 📋 Include reviewer credibility scoring

---





