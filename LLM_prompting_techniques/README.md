# 🧠 Prompt Engineering Techniques with Google Gemini API

## 📌 Overview
This repository demonstrates multiple **prompt engineering strategies** using **Large Language Models (LLMs)** to generate **professional IT support email responses** via the **Google Gemini API**.

The project highlights how different prompting techniques affect:
- Reasoning depth  
- Output quality  
- Structure and clarity of responses  

Each technique is implemented as a separate Python script, and generated outputs are automatically stored for comparison and learning.

---

## 🎯 Project Objectives
- Learn and apply core **prompt engineering patterns**
- Compare LLM reasoning approaches
- Generate real-world, professional IT support emails
- Automate output storage for analysis
- Build a practical GenAI-based automation project

---

## 🧩 Prompting Techniques Covered

### 1️⃣ Single-Shot Prompting
**File:** `single_shot.py`  
Direct instruction without examples.  
Best for simple, fast responses.

---

### 2️⃣ Few-Shot Prompting
**File:** `few_shot.py`  
Uses one example to guide format and tone.

---

### 3️⃣ Multi-Shot Prompting
**File:** `multi_shot.py`  
Uses multiple examples to strongly enforce structure and consistency.

---

### 4️⃣ Chain-of-Thought (CoT)
**File:** `chain_of_thought.py`  
Encourages step-by-step reasoning before producing the final output.

---

### 5️⃣ Tree-of-Thought (ToT)
**File:** `tree_of_thought.py`  
Explores multiple reasoning paths and selects the most comprehensive answer.

---

### 6️⃣ Practical IT Support Use Case
**File:** `app.py`  
Simulates a real-world IT support scenario involving:
- Authentication issues  
- GitHub private repository access  
- Clear, step-by-step resolution  

---

## 📂 Project Structure

```
.
├── app.py
├── single_shot.py
├── few_shot.py
├── multi_shot.py
├── chain_of_thought.py
├── tree_of_thought.py
├── outputs/
│   ├── single_shot_YYYYMMDD_HHMMSS.txt
│   ├── few_shot_YYYYMMDD_HHMMSS.txt
│   ├── multi_shot_YYYYMMDD_HHMMSS.txt
│   ├── Chain_of_thoughtYYYYMMDD_HHMMSS.txt
│   ├── tree_of_thoughts_YYYYMMDD_HHMMSS.txt
│   └── app_YYYYMMDD_HHMMSS.txt
└── README.md
```

---

## 🛠️ Tech Stack
- Python 3
- Google Gemini API (`google-genai`)
- Prompt Engineering Techniques
- File Handling & Automation

---

## ⚙️ Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/your-username/prompt-engineering-gemini.git
cd prompt-engineering-gemini
```

### Install Dependencies
```bash
pip install google-genai
```

### Configure API Key (Recommended)
```bash
export GEMINI_API_KEY="your_api_key_here"
```

> ⚠️ Do not hardcode API keys in production environments.

---

## ▶️ How to Run

Run any script to test a specific prompting technique:

```bash
python single_shot.py
python few_shot.py
python multi_shot.py
python chain_of_thought.py
python tree_of_thought.py
python app.py
```

Each execution:
- Calls the Gemini model  
- Generates a professional IT support email  
- Saves output to the `outputs/` directory  

---

## 📄 Output Details
- Outputs are saved as `.txt` files
- Timestamp-based filenames for easy comparison
- Professional email format including:
  - Subject
  - Greeting
  - Clear explanation
  - Step-by-step guidance
  - Formal closing

---

## 🧠 Key Takeaways
- Prompt design significantly impacts LLM responses
- Reasoning-based prompts improve accuracy and depth
- Example-driven prompts enhance consistency
- Suitable for **GenAI portfolios**, **IT automation**, and **enterprise AI use cases**

---

## 🚀 Future Improvements
- Build a web interface (Streamlit / FastAPI)
- Add output comparison dashboard
- Implement prompt evaluation metrics
- Improve API key security using `.env`
- Add logging and error handling
