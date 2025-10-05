# 🚀 EXODETECT - Exoplanet Classification & Analysis System  

**Advanced web application for exoplanet data exploration, classification, and batch processing using machine learning**

---

## 🪐 Overview  

**EXODETECT** is a comprehensive **Streamlit-based web application** designed for analyzing and classifying exoplanet candidates from **NASA's Kepler, K2, TESS, and CoRoT missions**.  
It combines **interactive data visualization** with **machine learning capabilities** to help astronomers and researchers efficiently process large datasets of planetary candidates.

---

## ✨ Features  

### 🔍 Data Exploration  
- Interactive sky map with RA/Dec coordinates  
- Advanced filtering by mission, disposition, and stellar parameters  
- Real-time search results with physical properties  
- Multi-parameter query system  

### 🌍 Detailed Planet Analysis  
- Comprehensive physical property display  
- Radar chart comparison with Earth  
- Orbital and stellar characteristics  
- Temperature and habitability metrics  

### 🤖 Machine Learning Classification  
- Single prediction mode with confidence scores  
- Support for custom-trained models (`.joblib` / `.pkl`)  
- Automatic baseline model training  
- Feature importance visualization  
- Multi-class probability distribution  

### 📊 Batch Processing & Triage  
- CSV file upload for bulk analysis  
- Automated predictions with priority scoring  
- Confidence-based ranking system  
- Exportable results (`.csv` format)  
- Distribution analytics and visualizations  

---

## 🧠 Technical Stack  

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (RandomForest, Pipeline, Imputer) |
| **Visualization** | Plotly (Express & Graph Objects) |
| **Model Persistence** | Joblib |

---

## 📂 Data Format  

The system accepts **CSV files** following the **NASA Exoplanet Archive** format.  
Required columns may include:

| Column | Description |
|--------|-------------|
| `koi_period` | Orbital period (days) |
| `koi_duration` | Transit duration (hours) |
| `koi_prad` | Planetary radius (Earth radii) |
| `koi_depth` | Transit depth (ppm) |
| `koi_steff` | Stellar effective temperature (K) |
| `disposition` | Classification label (for training) |

---

## 🧩 Model Training  

You can:
- Upload your own **trained model**, or  
- Let the system **automatically train a baseline RandomForest classifier** using your data’s `disposition` labels.

---

## 🗂️ Project Structure  

exodetect/
│
├── exodetect_app.py # Main application
├── requirements.txt # Python dependencies
├── README.md # Documentation
└── models/ # Trained models (optional)

---

## 👨‍💻 Author  

**Fasih Ur Rehman**  
BSCS (8th Semester) — Khwaja Fareed University of Engineering & Information Technology (KICSIT), Institute of Space Technology (IST)  

📞 03467042773  

---

## 🪪 License  

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgments  

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) for providing publicly accessible exoplanet data  
- [Streamlit](https://streamlit.io/) team for their excellent web framework  
- [Scikit-learn](https://scikit-learn.org/) contributors for robust machine learning tools  

---

## 🤝 Contributing  

Contributions, issues, and feature requests are welcome!  
Feel free to check the **[issues page](../../issues)** for open tasks.

---

## 📬 Contact  

For questions, ideas, or collaborations, reach out at:  
📧 fasihsatti168@gmail.com  
📞 03467042773  
🌐 [GitHub Profile](https://github.com/Fasih-Satti)
