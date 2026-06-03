import os
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

app = Flask(__name__, static_folder='static', static_url_path='')

DATA_FILE = 'credit_data.csv'

def generate_synthetic_data(num_samples=5000, seed=42, noise_level=0.15, base_risk_shift=0.0):
    """Generates a realistic synthetic credit risk dataset."""
    np.random.seed(seed)
    
    # 1. Age (18 to 75)
    age = np.random.randint(18, 76, size=num_samples)
    
    income = np.random.lognormal(mean=10.8, sigma=0.5, size=num_samples)
    
    income = np.clip(income, 15000, 250000).round(-2)
    
    employment_years = np.zeros(num_samples)
    for i in range(num_samples):
        max_emp = max(0, age[i] - 18)
       
        employment_years[i] = round(np.random.beta(a=1.5, b=5.0) * max_emp, 1)
        
    home_ownership_choices = ['OWN', 'MORTGAGE', 'RENT', 'OTHER']
    home_ownership = np.random.choice(home_ownership_choices, size=num_samples, p=[0.25, 0.45, 0.25, 0.05])
    
    

    credit_score = np.random.normal(loc=660, scale=80, size=num_samples)
    credit_score = np.clip(credit_score, 300, 850).astype(int)
    

    dti = np.random.beta(a=2.0, b=4.0, size=num_samples) * 0.9 + 0.05
    dti = np.round(dti, 3)

    payment_history = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        # Lambda varies from 0.1 (high credit score) to 4.5 (low credit score)
        score_factor = (850 - credit_score[i]) / 550.0
        lam = max(0.1, score_factor * 4.0)
        payment_history[i] = np.random.poisson(lam=lam)
    payment_history = np.clip(payment_history, 0, 12)
        
    # 8. Loan Amount ($1,000 to $50,000)
    # Higher income individuals get approved for larger loans
    loan_amount = np.zeros(num_samples)
    for i in range(num_samples):
        max_loan = min(50000, income[i] * 0.4)
        loan_amount[i] = round(np.random.uniform(1000, max_loan), -2)
        
    # 9. Loan Purpose
    loan_purpose_choices = ['EDUCATION', 'HOME_IMPROVEMENT', 'VENTURE', 'PERSONAL', 'MEDICAL', 'DEBT_CONSOLIDATION']
    loan_purpose = np.random.choice(loan_purpose_choices, size=num_samples, p=[0.15, 0.20, 0.15, 0.15, 0.10, 0.25])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'EmploymentYears': employment_years,
        'HomeOwnership': home_ownership,
        'CreditScore': credit_score,
        'DebtToIncomeRatio': dti,
        'PaymentHistory': payment_history,
        'LoanAmount': loan_amount,
        'LoanPurpose': loan_purpose
    })
    
    # 10. Default Probability (Logit model to create logical target classification)
    # High DTI, lower CreditScore, higher Late Payments, RENT status, lower income increase default risk.
    risk_logit = (
        -1.5 
        + base_risk_shift
        + 4.2 * (df['DebtToIncomeRatio'] - 0.35)
        - 0.014 * (df['CreditScore'] - 650)
        + 0.55 * df['PaymentHistory']
        - 0.06 * df['EmploymentYears']
        - 0.000006 * (df['Income'] - 60000)
        + 0.4 * (df['HomeOwnership'] == 'RENT').astype(int)
        - 0.4 * (df['HomeOwnership'] == 'OWN').astype(int)
        + 0.8 * (df['LoanAmount'] / df['Income'])
        + 0.2 * (df['LoanPurpose'] == 'MEDICAL').astype(int)
        + 0.3 * (df['LoanPurpose'] == 'DEBT_CONSOLIDATION').astype(int)
    )
    
    # Convert logit to probability
    prob_default = 1 / (1 + np.exp(-risk_logit))
    
    # Generate labels
    df['Default'] = np.random.binomial(1, prob_default)
    
    # Introduce noise by flipping labels randomly
    if noise_level > 0.0:
        flip_mask = np.random.rand(num_samples) < noise_level
        df.loc[flip_mask, 'Default'] = 1 - df.loc[flip_mask, 'Default']
    
    return df

def get_or_create_dataset():
    """Retrieves dataset from disk if it exists, or creates and saves a new one."""
    if os.path.exists(DATA_FILE):
        try:
            return pd.read_csv(DATA_FILE)
        except Exception:
            pass
    df = generate_synthetic_data()
    df.to_csv(DATA_FILE, index=False)
    return df

# Global variables for models and preprocessor
models = {}
preprocessor = None
X_train_processed = None
y_train_arr = None
feature_names = []

def get_preprocessor_and_features(df):
    """Initializes the ColumnTransformer pipeline."""
    numeric_features = ['Age', 'Income', 'EmploymentYears', 'CreditScore', 'DebtToIncomeRatio', 'PaymentHistory', 'LoanAmount']
    categorical_features = ['HomeOwnership', 'LoanPurpose']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    return preprocessor, numeric_features, categorical_features

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/data-summary', methods=['GET'])
def get_data_summary():
    """Generates aggregate statistics and data distribution summaries for the frontend dashboard."""
    df = get_or_create_dataset()
    
    # General statistics
    total_records = len(df)
    defaults = int(df['Default'].sum())
    non_defaults = total_records - defaults
    default_rate = round(defaults / total_records * 100, 2)
    
    # Numeric column stats
    numeric_cols = ['Age', 'Income', 'EmploymentYears', 'CreditScore', 'DebtToIncomeRatio', 'PaymentHistory', 'LoanAmount']
    summary_stats = {}
    for col in numeric_cols:
        summary_stats[col] = {
            'mean': round(float(df[col].mean()), 2),
            'min': round(float(df[col].min()), 2),
            'max': round(float(df[col].max()), 2),
            'std': round(float(df[col].std()), 2)
        }
        
    # Class distribution by CreditScore bins
    bins = [300, 580, 670, 740, 800, 850]
    labels = ['Poor (300-579)', 'Fair (580-669)', 'Good (670-739)', 'Very Good (740-799)', 'Exceptional (800-850)']
    df['CreditScoreRange'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, include_lowest=True)
    score_dist = df.groupby(['CreditScoreRange', 'Default'], observed=False).size().unstack(fill_value=0)
    score_chart_data = {
        'labels': labels,
        'non_defaults': [int(x) for x in score_dist[0].values],
        'defaults': [int(x) for x in score_dist[1].values]
    }
    
    # DTI vs Default distribution
    dti_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    dti_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    df['DTIRange'] = pd.cut(df['DebtToIncomeRatio'], bins=dti_bins, labels=dti_labels, include_lowest=True)
    dti_dist = df.groupby(['DTIRange', 'Default'], observed=False).size().unstack(fill_value=0)
    dti_chart_data = {
        'labels': dti_labels,
        'non_defaults': [int(x) for x in dti_dist[0].values],
        'defaults': [int(x) for x in dti_dist[1].values]
    }
    
    # Home ownership vs Default
    home_dist = df.groupby(['HomeOwnership', 'Default']).size().unstack(fill_value=0)
    home_chart_data = {
        'labels': list(home_dist.index),
        'non_defaults': [int(x) for x in home_dist[0].values],
        'defaults': [int(x) for x in home_dist[1].values]
    }

    # Clean temporary columns
    df.drop(columns=['CreditScoreRange', 'DTIRange'], inplace=True, errors='ignore')
    
    return jsonify({
        'total_records': total_records,
        'defaults': defaults,
        'non_defaults': non_defaults,
        'default_rate': default_rate,
        'summary_stats': summary_stats,
        'charts': {
            'credit_score_dist': score_chart_data,
            'dti_dist': dti_chart_data,
            'home_ownership': home_chart_data
        }
    })

@app.route('/api/generate-data', methods=['POST'])
def generate_custom_data():
    """Regenerates dataset based on user-provided size, noise, and default risk shift."""
    global models, preprocessor
    # Clear models so they must be retrained on new data
    models = {}
    preprocessor = None
    
    try:
        data = request.json or {}
        size = int(data.get('size', 5000))
        noise = float(data.get('noise', 0.15))
        risk_level = data.get('risk_level', 'balanced')
        
        # Map risk level to logit shifts
        risk_shifts = {'low': -1.2, 'balanced': 0.0, 'high': 1.2}
        shift = risk_shifts.get(risk_level, 0.0)
        
        df = generate_synthetic_data(num_samples=size, noise_level=noise, base_risk_shift=shift)
        df.to_csv(DATA_FILE, index=False)
        
        # Return data summary response
        return get_data_summary()
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/train', methods=['POST'])
def train_models():
    """Trains Logistic Regression, Decision Tree, and Random Forest. Returns metrics, ROC curves, feature importances."""
    global models, preprocessor, X_train_processed, y_train_arr, feature_names
    
    df = get_or_create_dataset()
    
    X = df.drop(columns=['Default'])
    y = df['Default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_train_arr = y_train.values
    
    # Fit preprocessor
    preprocessor, num_cols, cat_cols = get_preprocessor_and_features(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(cat_cols))
    feature_names = num_cols + encoded_cat_cols
    
    # Define models
    clf_models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in clf_models.items():
        # Train model
        model.fit(X_train_processed, y_train)
        models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        # Downsample ROC coordinates for performance
        downsample_factor = max(1, len(fpr) // 30)
        roc_data = []
        for idx in range(0, len(fpr), downsample_factor):
            roc_data.append({'fpr': round(float(fpr[idx]), 3), 'tpr': round(float(tpr[idx]), 3)})
        # Make sure endpoints (1,1) are added
        if roc_data[-1] != {'fpr': 1.0, 'tpr': 1.0}:
            roc_data.append({'fpr': 1.0, 'tpr': 1.0})
            
        # Feature Importance / Coefficients
        importance_list = []
        if name == 'Logistic Regression':
            importances = model.coef_[0]
        else:
            importances = model.feature_importances_
            
        for feat, val in zip(feature_names, importances):
            importance_list.append({'feature': feat, 'value': round(float(val), 4)})
            
        # Sort importances by absolute value
        importance_list = sorted(importance_list, key=lambda x: abs(x['value']), reverse=True)[:10]
        
        results[name] = {
            'metrics': {
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1-Score': round(f1, 4),
                'ROC-AUC': round(auc, 4)
            },
            'confusion_matrix': {
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn),
                'TP': int(tp)
            },
            'roc_curve': roc_data,
            'feature_importance': importance_list
        }
        
    return jsonify(results)

@app.route('/api/predict', methods=['POST'])
def predict_risk():
    """Predicts default probability for custom user input and provides model explanation contributions."""
    global models, preprocessor, feature_names
    
    if not models or preprocessor is None:
        return jsonify({'error': 'Models have not been trained yet. Please train models first.'}), 400
        
    try:
        data = request.json
        # Convert incoming JSON data into pandas DataFrame
        input_data = pd.DataFrame([{
            'Age': int(data['Age']),
            'Income': float(data['Income']),
            'EmploymentYears': float(data['EmploymentYears']),
            'HomeOwnership': data['HomeOwnership'],
            'CreditScore': int(data['CreditScore']),
            'DebtToIncomeRatio': float(data['DebtToIncomeRatio']),
            'PaymentHistory': int(data['PaymentHistory']),
            'LoanAmount': float(data['LoanAmount']),
            'LoanPurpose': data['LoanPurpose']
        }])
        
        # Preprocess input data
        input_processed = preprocessor.transform(input_data)
        
        predictions = {}
        for name, model in models.items():
            prob = model.predict_proba(input_processed)[0, 1]
            pred = int(model.predict(input_processed)[0])
            predictions[name] = {
                'probability': round(float(prob), 4),
                'prediction': pred
            }
            
        # Explanations using Random Forest feature importances scaled by input standardization & risk direction
        rf_model = models.get('Random Forest')
        explanation = []
        if rf_model:
            scaler = preprocessor.named_transformers_['num']
            numeric_features = ['Age', 'Income', 'EmploymentYears', 'CreditScore', 'DebtToIncomeRatio', 'PaymentHistory', 'LoanAmount']
            
            # The standardized single sample processed values
            val_processed = input_processed[0]
            importances = rf_model.feature_importances_
            
            # Default directional mapping (negative = reduces risk, positive = raises risk)
            directions = {}
            for col in feature_names:
                if 'CreditScore' in col or 'Income' in col or 'EmploymentYears' in col or 'Age' in col:
                    directions[col] = -1.0
                elif 'HomeOwnership_OWN' in col:
                    directions[col] = -1.0
                else:
                    directions[col] = 1.0
                    
            contributions = []
            for i, feat in enumerate(feature_names):
                importance = importances[i]
                val_std = val_processed[i]
                direction = directions.get(feat, 1.0)
                contrib = val_std * importance * direction
                
                display_name = feat
                actual_val = ""
                
                if feat in numeric_features:
                    raw_val = input_data.iloc[0][feat]
                    if feat == 'Income' or feat == 'LoanAmount':
                        actual_val = f"${int(raw_val):,}"
                    elif feat == 'DebtToIncomeRatio':
                        actual_val = f"{round(raw_val * 100, 1)}%"
                    else:
                        actual_val = str(raw_val)
                else:
                    actual_val = "Yes" if val_std > 0.5 else "No"
                    display_name = feat.replace('cat__', '').replace('_', ': ')
                    
                contributions.append({
                    'feature': display_name,
                    'value': round(contrib, 4),
                    'actual_value': actual_val
                })
                
            contributions = sorted(contributions, key=lambda x: abs(x['value']), reverse=True)[:6]
            explanation = contributions
            
        return jsonify({
            'predictions': predictions,
            'explanations': explanation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Ensure static folder exists
    os.makedirs('static', exist_ok=True)
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=True)
