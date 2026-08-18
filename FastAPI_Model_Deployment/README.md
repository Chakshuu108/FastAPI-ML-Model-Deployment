# Insurance Premium Category Predictor

A machine learning API that predicts a user's **Insurance Premium Category** (`Low` / `Medium` / `High`) based on health, lifestyle, and demographic details — served via **FastAPI** and consumed by a **Streamlit** frontend. Also includes a separate **Patient Management System API** for BMI-based patient records.

---

## 📁 Project Structure

```
├── ml_model.ipynb    # Notebook: trains the RandomForest pipeline
├── model.pkl         # Pickled, trained sklearn Pipeline
├── insurance.csv     # Training dataset (100 rows)
├── app.py            # FastAPI backend — /predict endpoint
├── frontend.py       # Streamlit UI that calls the FastAPI backend
├── main.py           # Patient Management System API
└── patients.json     # JSON "database" for patient records
```

---

## 🔄 How It Works

```
        USER
          │
          ▼
 Streamlit Frontend (frontend.py)
 collects: age, weight, height,
 income, smoker, city, occupation
          │  POST /predict (JSON)
          ▼
 FastAPI Backend (app.py)
 Pydantic validates raw input
          │
          ▼
 Computed Fields (auto-derived)
 bmi · lifestyle_risk · age_group · city_tier
          │
          ▼
 model.pkl (sklearn Pipeline)
 OneHotEncoder + RandomForestClassifier
          │  prediction
          ▼
 JSON Response
 {"predicted_category": "Low/Medium/High"}
          │
          ▼
 Result shown on Streamlit UI
```

---

## Machine Learning Pipeline (`ml_model.ipynb`)

### Libraries Used
| Library | Purpose |
|---|---|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` (`RandomForestClassifier`) | The classification model |
| `train_test_split` | Splitting data into train/test sets |
| `OneHotEncoder` | Encoding categorical features |
| `Pipeline` & `ColumnTransformer` | Chaining preprocessing + model into one object |
| `classification_report`, `accuracy_score` | Model evaluation |
| `pickle` | Saving the trained model to `model.pkl` |

### Dataset — `insurance.csv`
100 rows, 8 columns: `age`, `weight`, `height`, `income_lpa`, `smoker`, `city`, `occupation`, `insurance_premium_category` (target).

### Feature Engineering
The notebook engineers 4 new features from the raw data before training:

| # | Feature | Logic |
|---|---|---|
| 1️⃣ | **bmi** | `weight / (height ** 2)` |
| 2️⃣ | **age_group** | `<25` → `young`, `<45` → `adult`, `<60` → `middle_aged`, else `senior` |
| 3️⃣ | **lifestyle_risk** | `smoker & bmi>30` → `high`; `smoker OR bmi>27` → `medium`; else `low` |
| 4️⃣ | **city_tier** | City in Tier-1 list → `1`, Tier-2 list → `2`, else → `3` |

### Final Model Inputs & Target
- **Features (X):** `bmi`, `age_group`, `lifestyle_risk`, `city_tier`, `income_lpa`, `occupation`
- **Target (y):** `insurance_premium_category` → `Low` / `Medium` / `High`

### Model Pipeline
```
ColumnTransformer
 ├── cat → OneHotEncoder → [age_group, lifestyle_risk, occupation, city_tier]
 └── num → passthrough    → [bmi, income_lpa]
        │
        ▼
RandomForestClassifier(random_state=42)
```

- **Train/Test split:** 80/20 (`test_size=0.2`, `random_state=1`)
- **Accuracy achieved:** ~90% on the test set
- **Saved as:** `model.pkl` via `pickle.dump()`

---

## Backend — FastAPI (`app.py`)

### Endpoint
| Method | Route | Description |
|---|---|---|
| `POST` | `/predict` | Takes user details → returns predicted premium category |

### Input Validation — `UserInput` (Pydantic Model)
| Field | Type | Constraint |
|---|---|---|
| `age` | `int` | `0 < age < 120` |
| `weight` | `float` | `> 0` |
| `height` | `float` | `0 < height < 2.5` (meters) |
| `income_lpa` | `float` | `> 0` |
| `smoker` | `bool` | — |
| `city` | `str` | — |
| `occupation` | `Literal[...]` | one of: `retired`, `freelancer`, `student`, `government_job`, `business_owner`, `unemployed`, `private_job` |

### Computed Fields (`@computed_field`)
Derived automatically the moment `UserInput` is created — the client never sends these directly:

| Computed Field | Logic |
|---|---|
| `bmi` | `weight / height²` |
| `lifestyle_risk` | `high` if smoker & bmi>30 · `medium` if smoker OR bmi>27 · else `low` |
| `age_group` | `young` (<25) · `adult` (<45) · `middle_aged` (<60) · `senior` (60+) |
| `city_tier` | `1` (Tier-1 metro) · `2` (Tier-2 city) · `3` (everything else) |

City tier is looked up against a hardcoded list of 7 Tier-1 cities (Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Pune) and 48 Tier-2 cities; anything else defaults to Tier 3.

### Sample Response
```json
{
  "predicted_category": "Medium"
}
```

---

## Frontend — Streamlit (`frontend.py`)

- Form UI collecting: age, weight, height, income, smoker status, city, occupation.
- Sends a `POST` request to the FastAPI server (`API_URL` in the file).
- On success, shows the predicted category.
- **Note:** the frontend also expects `confidence` and `class_probabilities` keys, but `app.py` currently only returns `predicted_category` — see Known Issues below.

---

## Patient Management System API (`main.py`)

A separate CRUD API for managing patient BMI records, stored in `patients.json`.

### Endpoints
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

### Computed Fields — `Patient` Model
| Field | Logic |
|---|---|
| `bmi` | `round(weight / height², 2)` |
| `verdict` | `<18.5` → `Underweight` · `<30` → `Normal` · `≥30` → `Obese` |

---

## Installation & Setup

```bash
pip install fastapi uvicorn pydantic pandas scikit-learn streamlit requests
```

Run the FastAPI backend:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Run the Streamlit frontend:
```bash
streamlit run frontend.py
```
> Update `API_URL` in `frontend.py` if the backend isn't running locally.

Run the Patient Management API (separate service):
```bash
uvicorn main:app --reload --port 8001
```

---

## Known Issues
- `frontend.py` expects `confidence` and `class_probabilities` in the response, but `/predict` in `app.py` doesn't return them — either update the frontend or extend the backend with `model.predict_proba()`.
- `app.py` has no CORS middleware — needed if frontend and backend are on different domains.
- The hardcoded `API_URL` in `frontend.py` should ideally be an environment variable.
- No authentication on either API.
