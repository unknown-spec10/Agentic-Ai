# Skill Gap Analysis & Job Search 🚀

A powerful tool for analyzing skill gaps, finding job matches, and getting personalized career recommendations.

---

## 📚 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [API Integration](#api-integration)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## 🧠 Overview

This project is a comprehensive career development tool that helps users:
- Analyze their skills against job market requirements
- Find matching job opportunities
- Get personalized learning recommendations
- Track market demand for different roles
- Receive detailed skill gap analysis

---

## ✨ Features

### 1. Resume Analysis 📄
- PDF resume parsing and text extraction
- Automatic skill identification
- Experience level detection
- Certification recognition
- Project and leadership experience analysis

### 2. Skill Gap Analysis 🔍
- Comparison with job requirements
- Weighted skill matching algorithm
- Experience-based scoring
- Detailed gap identification
- Personalized improvement suggestions

### 3. Market Analysis 📊
- Real-time job market data
- Salary range analysis
- Industry demand tracking
- Location-based insights
- Remote work opportunity analysis

### 4. Job Recommendations 💼
- Personalized job matching
- Role-based suggestions
- Experience-level filtering
- Company and location matching
- Salary range optimization

### 5. Learning Recommendations 📚
- Skill-based learning paths
- Course recommendations
- Certification guidance
- Resource suggestions
- Progress tracking

---

## 🧰 Technology Stack

### 🖥️ Frontend
- Streamlit (UI Framework)
- HTML/CSS (Styling)
- JavaScript (Interactivity)

### ⚙️ Backend
- Python 3.8+
- FastAPI or Flask (API Framework)
- PyPDF2 (PDF Processing)
- NumPy (Numerical Processing)
- Pandas (Data Analysis)

### 🤖 AI/ML
- Google Gemini Pro (AI Model)
- FAISS (Vector Similarity)
- Scikit-learn (Machine Learning)

### 🌐 APIs
- Adzuna API (Job Data)
- Gemini API (AI Processing)

### 💾 Storage
- JSON (Data Storage)
- File System (Cache)

---

## 📁 Project Structure

The codebase is organized using a modular structure:

```
Agentic-Ai/
├── app.py                  # Main Streamlit application entry point
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
├── SETUP_GUIDE.md         # Setup instructions
├── .streamlit/            # Streamlit configuration files
│   └── secrets.toml       # API keys and other sensitive data
├── data/                  # Data files and resources
├── database/              # Database for persistent storage
├── cache/                 # Cache directory for API responses
└── src/                   # Source code modules
    ├── api/               # API integration components
    │   └── persistent_db.py  # Database operations for API responses
    ├── core/              # Core analysis functionality
    │   ├── skill_analysis.py # Skill analysis algorithms
    │   ├── tech_utils.py     # Technology utilities
    │   └── update_analysis.py # Analysis update functionality
    └── utils/             # Utility functions
        ├── cache_manager.py  # Cache management utilities
        └── general_utils.py  # General utility functions
```

### 🔑 Core Components

- **app.py**: The main Streamlit application that provides the user interface
- **src/core/skill_analysis.py**: Core logic for analyzing skills and gaps
- **src/utils/general_utils.py**: Utility functions for API access, formatting, etc.
- **src/api/persistent_db.py**: Database management for persistent data storage
