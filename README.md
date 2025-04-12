# Skill Gap Analysis & Job Search 🚀

A powerful tool for analyzing skill gaps, finding job matches, and getting personalized career recommendations.

## Table of Contents
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

## Overview

This project is a comprehensive career development tool that helps users:
- Analyze their skills against job market requirements
- Find matching job opportunities
- Get personalized learning recommendations
- Track market demand for different roles
- Receive detailed skill gap analysis

## Features

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

## Technology Stack

### Frontend
- Streamlit (UI Framework)
- HTML/CSS (Styling)
- JavaScript (Interactivity)

### Backend
- Python 3.8+
- FastAPI/Flask (API Framework)
- PyPDF2 (PDF Processing)
- NumPy (Numerical Processing)
- Pandas (Data Analysis)

### AI/ML
- Google Gemini Pro (AI Model)
- FAISS (Vector Similarity)
- Scikit-learn (Machine Learning)

### APIs
- Adzuna API (Job Data)
- Gemini API (AI Processing)

### Storage
- JSON (Data Storage)
- File System (Cache)

## Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/yourusername/skill-analysis.git
cd skill-analysis
\`\`\`

2. Create a virtual environment:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. Set up environment variables:
\`\`\`bash
cp .env.example .env
# Edit .env with your API keys
\`\`\`

5. Run the application:
\`\`\`bash
streamlit run app.py
\`\`\`

## Usage

1. **Upload Resume**
   - Support for PDF format
   - Automatic text extraction
   - Skill identification

2. **Enter Job Details**
   - Specify target role
   - Select location
   - Input experience level

3. **Review Analysis**
   - Skill match percentage
   - Market demand insights
   - Salary ranges
   - Job recommendations

4. **Get Recommendations**
   - Learning paths
   - Job matches
   - Skill improvement suggestions

## Project Structure

\`\`\`
skill-analysis/
├── app.py              # Main Streamlit application
├── skill_analysis.py   # Core analysis logic
├── utils.py           # Utility functions
├── requirements.txt   # Project dependencies
├── .env              # Environment variables
├── README.md         # Documentation
└── cache/           # Cache directory
    └── embeddings/  # Cached embeddings
\`\`\`

## Core Components

### 1. Resume Parser
- PDF text extraction
- Skill pattern matching
- Experience calculation
- Certification detection

### 2. Skill Analyzer
- Vector embeddings
- Similarity matching
- Weight calculation
- Gap identification

### 3. Market Analyzer
- Job data aggregation
- Salary analysis
- Trend detection
- Demand forecasting

### 4. Recommendation Engine
- Job matching algorithm
- Learning path generation
- Resource curation
- Personalization logic

## API Integration

### Adzuna API
- Job posting retrieval
- Salary data
- Market trends
- Location analysis

### Gemini API
- Text analysis
- Content generation
- Embedding creation
- Similarity matching

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Deep Podder**
- GitHub: [@unknown-spec10](https://github.com/unknown-spec10)
- LinkedIn: [Deep Podder](https://www.linkedin.com/in/deeppodder2005)
- Email: deeppodder2005@gmail.com

---

<div align="center">
  <p>Created with ❤️ by Deep Podder</p>
  <a href="https://github.com/unknown-spec10">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://www.linkedin.com/in/deeppodder2005">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
</div>
