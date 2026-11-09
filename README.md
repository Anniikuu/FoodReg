# AI-Based Food Recognition and Personalized Nutrition Tracking System

An AI-driven system for food image recognition and personalized nutritional tracking.

## Current Phase

**Phase 3: Nutrition Database (Source Verification & Compilation)**

Current status: Local nutrition database created (`data/nutrition_database.csv`) with 13 columns.
- 18 classes have verified prepared-food nutrition values (from USDA FNDDS and INDB) per 100 g.
- 5 classes remain unresolved (`chole_bhature`, `jalebi`, `momos`, `pani_puri`, `vada_pav`) with null values due to absence of authoritative prepared-food records.

## Getting Started

### 1. Python Environment

Ensure Python 3.10+ is installed. Create and activate a virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

Install the minimal dependencies required for Phase 1:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Health Check

Launch the Streamlit health-check application:

```bash
streamlit run app/main.py
```

### 4. Run Automated Tests

To run the automated verification test:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

