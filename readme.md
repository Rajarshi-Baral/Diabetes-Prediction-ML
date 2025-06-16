# 🩺 Diabetes Prediction using Machine Learning

This project applies a machine learning model to predict whether a patient is likely to have diabetes based on several medical attributes. The system is trained on the well-known Pima Indians Diabetes dataset.

---

## 📌 Overview

The goal of this project is to assist healthcare providers in identifying potential diabetic patients early by analyzing clinical parameters using a Random Forest Classifier.

- 🔍 **Problem Type:** Binary Classification  
- 🧠 **Algorithm Used:** Random Forest (Tuned via GridSearchCV)  
- 💾 **Dataset:** Pima Indians Diabetes Dataset (CSV format)  
- 🛠️ **Tools:** Python, Pandas, Scikit-learn, Jupyter Notebook

---

## 📂 Project Structure

    Diabetes-Prediction-ML/
    ├── dataset/
    │   ├── diabetes_cleaned.csv
    │   ├── diabetes_rf_model.pkl
    │   └── model_testing.csv          # Auto-generated during app usage
    ├── code/
    │   ├── app.py                     # Streamlit web app
    │   ├── model_training.ipynb
    │   ├── model_testing.ipynb
    │   └── Procfile                   # For deployment (e.g., on Render)
    ├── requirements.txt              # Dependencies for local/deployment
    ├── README.md
    



---

## 🧪 Model Training & Evaluation

- Cleaned and preprocessed the dataset (handled zero/missing values).
- Performed train-test split (80/20).
- Tuned hyperparameters using `GridSearchCV`.
- Final model achieved an **accuracy of ~75%**.

### 📉 Evaluation Metrics:
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC-AUC Curve

---

## 🚀 Usage

### 🔬 Predict from Jupyter Notebook:

1. Run `model_training.ipynb` to train and save the model.
2. Use `model_testing.ipynb` to load the `.pkl` model and make predictions on new patient data.

---
### 🌐 Option 2: Use the Streamlit App

#### ▶️ Local Usage:

```bash
pip install -r requirements.txt
```
After installing `requirement.text`
```bash
pip install streamlit
```
Run the below into the "code" directory
```bash
    streamlit run app.py
```

### 📋 Features:

- Intuitive input form for medical attributes
- Displays result: Diabetic or Not Diabetic
- Logs user input and prediction to dataset/model_testing.csv
- Shows last 5 predictions inside the app

### 🌍 Deployment (Render / Streamlit Cloud):

- The app includes a Procfile for deployment on Render.
- Make sure both app.py and Procfile are inside the code/ folder.
- Ensure requirements.txt is in the root directory for build configuration.



📦Requirements

Install required Python packages:

```bash
    pip install pandas numpy scikit-learn joblib matplotlib seaborn

```



✍️ Author

😎 [Rajarshi Baral](https://www.instagram.com/rajarshi__baral/)    |    Aspiring Software Developer & Machine Learning Enthusiast

📧 baralrajarshi35@gmail.com

🔗 [LinkedIn](https://www.linkedin.com/in/rajarshi-baral-r350b01/) | [GitHub](https://github.com/Rajarshi-Baral)




🔮 Future Plans

✅ Build a simple web app using Streamlit or Flask for real-time predictions.

✅ Deploy the app on Streamlit Cloud or Render.

✅ Improve the ML pipeline and accuracy using XGBoost or ensembling.

✅ Add CSV upload support for batch prediction.




🛡️ Disclaimer

This project is for educational purposes only and should not be used for real medical diagnosis.   

