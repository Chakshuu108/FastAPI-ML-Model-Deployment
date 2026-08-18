# 🏥 Insurance Premium Category Predictor

A machine learning powered API that predicts a user's **Insurance Premium Category** (`Low` / `Medium` / `High`) based on their health, lifestyle, and demographic details — served via **FastAPI** and consumed by a **Streamlit** frontend. Also bundled with a separate **Patient Management System API** for BMI-based patient records. 🩺📊

---

## 📁 Project Structure

```
📦 insurance-premium-predictor
├── 🧠 ml_model.ipynb      # Notebook: trains the RandomForest pipeline
├── 📦 model.pkl           # Pickled, trained sklearn Pipeline
├── 📊 insurance.csv       # Training dataset (100 rows)
├── 🚀 app.py              # FastAPI backend — /predict endpoint
├── 🎨 frontend.py         # Streamlit UI that calls the FastAPI backend
├── 🩺 main.py              # Bonus: Patient Management System API
├── 🗂️ patients.json        # JSON "database" for patient records
└── 📄 README.md           # You are here!
```

---

## 🎯 What This Project Does

1. 🧑‍⚕️ User enters their **age, weight, height, income, smoking status, city, and occupation**.
2. ⚙️ The backend engineers extra features (BMI, age group, lifestyle risk, city tier).
3. 🤖 A trained **Random Forest Classifier** predicts the premium category: **Low, Medium,** or **High**.
4. 📤 The result is returned as JSON and shown beautifully on the Streamlit frontend.

---

## 🔄 How It Works — Flowchart

```
                🧑 USER
                  │
                  ▼
      ┌────────────────────────┐
      │  🎨 Streamlit Frontend  │
      │      (frontend.py)      │
      │  Collects: age, weight, │
      │  height, income, smoker,│
      │  city, occupation       │
      └───────────┬─────────────┘
                  │  HTTP POST /predict (JSON)
                  ▼
      ┌────────────────────────┐
      │   🚀 FastAPI Backend    │
      │        (app.py)         │
      │  ✅ Pydantic validates  │
      │     raw user input      │
      └───────────┬─────────────┘
                  │
                  ▼
      ┌────────────────────────┐
      │ 🧮 Computed Fields (auto)│
      │  • bmi                  │
      │  • lifestyle_risk       │
      │  • age_group            │
      │  • city_tier            │
      └───────────┬─────────────┘
                  │  DataFrame built
                  ▼
      ┌────────────────────────┐
      │  📦 model.pkl (Pipeline)│
      │  • OneHotEncoder (cat)  │
      │  • passthrough (num)    │
      │  • RandomForestClassifier│
      └───────────┬─────────────┘
                  │  prediction
                  ▼
      ┌────────────────────────┐
      │ 📤 JSON Response         │
      │ {"predicted_category":  │
      │   "Low/Medium/High"}    │
      └───────────┬─────────────┘
                  │
                  ▼
      ┌────────────────────────┐
      │ 🎉 Displayed on          │
      │    Streamlit UI          │
      └────────────────────────┘
```

---

## 🧪 Machine Learning Pipeline (`ml_model.ipynb`)

### 📚 Libraries Used
| Library | Purpose |
|---|---|
| 🐼 `pandas` | Data loading & manipulation |
| 🔢 `numpy` | Numerical operations |
| 🌲 `scikit-learn` (`RandomForestClassifier`) | The classification model |
| ✂️ `train_test_split` | Splitting data into train/test sets |
| 🔤 `OneHotEncoder` | Encoding categorical features |
| 🔧 `Pipeline` & `ColumnTransformer` | Chaining preprocessing + model into one object |
| 📈 `classification_report`, `accuracy_score` | Model evaluation |
| 🥒 `pickle` | Saving the trained model to `model.pkl` |

### 🗂️ Dataset — `insurance.csv`
- **100 rows**, 8 columns.
- Raw columns: `age`, `weight`, `height`, `income_lpa`, `smoker`, `city`, `occupation`, `insurance_premium_category` (target).

### 🛠️ Feature Engineering
The notebook engineers 4 new features from the raw data before training:

| # | Feature | Logic |
|---|---|---|
| 1️⃣ | **bmi** | `weight / (height ** 2)` |
| 2️⃣ | **age_group** | `<25` → `young`, `<45` → `adult`, `<60` → `middle_aged`, else `senior` |
| 3️⃣ | **lifestyle_risk** | `smoker & bmi>30` → `high`; `smoker OR bmi>27` → `medium`; else `low` |
| 4️⃣ | **city_tier** | City in Tier-1 list → `1`, Tier-2 list → `2`, else → `3` |

### 🎯 Final Model Inputs & Target
- **Features (X):** `bmi`, `age_group`, `lifestyle_risk`, `city_tier`, `income_lpa`, `occupation`
- **Target (y):** `insurance_premium_category` → `Low` / `Medium` / `High`

### 🧱 Model Pipeline
```
ColumnTransformer
 ├── cat → OneHotEncoder → [age_group, lifestyle_risk, occupation, city_tier]
 └── num → passthrough    → [bmi, income_lpa]
        │
        ▼
RandomForestClassifier(random_state=42)
```

- 🔀 **Train/Test split:** 80/20 (`test_size=0.2`, `random_state=1`)
- ✅ **Accuracy achieved:** ~**90%** on the test set
- 💾 **Saved as:** `model.pkl` via `pickle.dump()`

---

## 🚀 Backend — FastAPI (`app.py`)

### 🎯 Endpoint
| Method | Route | Description |
|---|---|---|
| `POST` | `/predict` | Takes user details → returns predicted premium category |

### ✅ Input Validation — `UserInput` (Pydantic Model)
| Field | Type | Constraint |
|---|---|---|
| `age` | `int` | `0 < age < 120` |
| `weight` | `float` | `> 0` |
| `height` | `float` | `0 < height < 2.5` (meters) |
| `income_lpa` | `float` | `> 0` |
| `smoker` | `bool` | — |
| `city` | `str` | — |
| `occupation` | `Literal[...]` | one of: `retired`, `freelancer`, `student`, `government_job`, `business_owner`, `unemployed`, `private_job` |

### 🧮 Computed Fields (auto-calculated via `@computed_field`)
These are **derived automatically** the moment the `UserInput` model is created — the user never sends them directly:

| Computed Field | 🧠 Logic |
|---|---|
| `bmi` | `weight / height²` |
| `lifestyle_risk` | `high` if smoker & bmi>30 · `medium` if smoker OR bmi>27 · else `low` |
| `age_group` | `young` (<25) · `adult` (<45) · `middle_aged` (<60) · `senior` (60+) |
| `city_tier` | `1` (Tier-1 metro) · `2` (Tier-2 city) · `3` (everything else) |

### 🌆 City Tier Lists
- **Tier 1 (7 cities):** Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Pune
- **Tier 2 (48 cities):** Jaipur, Chandigarh, Indore, Lucknow, Patna, Ludhiana, Noida, and more...
- **Tier 3:** Any city not in the above two lists

### 📦 How Prediction Works Internally
```python
input_df = pd.DataFrame([{
    'bmi': data.bmi,
    'age_group': data.age_group,
    'lifestyle_risk': data.lifestyle_risk,
    'city_tier': data.city_tier,
    'income_lpa': data.income_lpa,
    'occupation': data.occupation
}])
prediction = model.predict(input_df)[0]
```

### 📤 Sample Response
```json
{
  "predicted_category": "Medium"
}
```

---

## 🎨 Frontend — Streamlit (`frontend.py`)

- 🖥️ A simple, clean form UI titled **"Insurance Premium Category Predictor"**.
- 📝 Collects: age, weight, height, income, smoker status, city, occupation.
- 🌐 Sends a `POST` request to the FastAPI server at:
  ```
  http://34.226.152.222:8000/predict
  ```
- ✅ On success → shows the **predicted category**.
- ⚠️ Note: the current frontend also expects `confidence` and `class_probabilities` keys in the response, but `app.py` currently only returns `predicted_category` — this would need to be added to `app.py` for full compatibility (see 🐛 Known Issues below).
- ❌ On connection failure → shows a friendly error message.

---

## 🩺 Bonus: Patient Management System API (`main.py`)

A completely separate mini-project — a full CRUD API for managing patient BMI records, stored in `patients.json`.

### 📌 Endpoints
| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Welcome message |
| `GET` | `/about` | API description |
| `GET` | `/view` | View all patients |
| `GET` | `/patient/{patient_id}` | View a single patient by ID |
| `GET` | `/sort?sort_by=&order=` | Sort patients by `height`, `weight`, or `bmi` |
| `POST` | `/create` | Add a new patient |
| `PUT` | `/edit/{patient_id}` | Update an existing patient |
| `DELETE` | `/delete/{patient_id}` | Remove a patient |

### 🧮 Computed Fields — `Patient` Model
| Field | Logic |
|---|---|
| `bmi` | `round(weight / height², 2)` |
| `verdict` | `<18.5` → `Underweight` · `<30` → `Normal` · `≥30` → `Obese` |

### 🗂️ Data Storage
- Simple **JSON file** (`patients.json`) acting as a mini database.
- 5 sample patients pre-loaded (Ananya Verma, Ravi Mehta, Sneha Kulkarni, Arjun Verma, Neha Sinha).

---

## ⚙️ Installation & Setup

### 1️⃣ Clone / download the project files

### 2️⃣ Install dependencies
```bash
pip install fastapi uvicorn pydantic pandas scikit-learn streamlit requests
```

### 3️⃣ Run the FastAPI backend
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4️⃣ Run the Streamlit frontend
```bash
streamlit run frontend.py
```
> ⚠️ Update `API_URL` in `frontend.py` if not running the backend locally.

### 5️⃣ (Optional) Run the Patient Management API
```bash
uvicorn main:app --reload --port 8001
```

---

## 🐛 Known Issues / Improvement Ideas
- ⚠️ `frontend.py` expects `confidence` and `class_probabilities` in the response, but `/predict` in `app.py` doesn't return them yet — either update the frontend or extend the backend with `model.predict_proba()`.
- 🔓 `app.py` currently has no CORS middleware — add `fastapi.middleware.cors` if the frontend and backend are hosted on different domains.
- 📉 Dataset is small (100 rows) — more data would likely improve model robustness.
- 🌍 The hardcoded `API_URL` (`34.226.152.222`) should ideally be moved to an environment variable.
- 🔐 No authentication on either API — fine for a demo, but add auth before any real deployment.

---

## 🛠️ Tech Stack Summary
| Layer | Technology |
|---|---|
| 🤖 ML Model | scikit-learn (RandomForestClassifier) |
| 🚀 Backend API | FastAPI + Pydantic |
| 🎨 Frontend | Streamlit |
| 📦 Serialization | Pickle |
| 🗃️ Data Storage | CSV (training) + JSON (patients) |

---

✨ **Made with FastAPI, scikit-learn & Streamlit** ✨
