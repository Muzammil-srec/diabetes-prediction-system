# ==============================================================
#   DIABETES PREDICTION SYSTEM — ALL GENDERS
#   Dataset : Combined Male + Female (1000 patients)
#   Models  : Logistic Regression + Random Forest
#   Run     : python diabetes_all_genders.py
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.impute          import SimpleImputer
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, roc_curve,
                                     confusion_matrix, classification_report)

warnings.filterwarnings('ignore')
os.makedirs("outputs_gender", exist_ok=True)

BLUE, RED, GREEN, PURPLE = '#1565C0', '#C62828', '#1B5E20', '#6A1B9A'

# ══════════════════════════════════════════════════════════════
# PART 1 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════
def train_models():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║   DIABETES PREDICTION — ALL GENDERS              ║")
    print("  ║   Training models — please wait...               ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    # ── STEP 1: Load Dataset ──────────────────────────────────
    print("  [1/6] Loading dataset...")
    df = pd.read_csv("diabetes_all_genders.csv")
    print(f"        Total patients  : {len(df)}")
    print(f"        Males           : {(df['Gender']=='Male').sum()}")
    print(f"        Females         : {(df['Gender']=='Female').sum()}")
    print(f"        Diabetic        : {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
    print(f"        Non-Diabetic    : {(df['Outcome']==0).sum()} ({(df['Outcome']==0).mean()*100:.1f}%)")

    # ── STEP 2: EDA Charts ────────────────────────────────────
    print("  [2/6] Creating EDA charts...")

    # Chart 1 — Gender distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle("Dataset Overview — All Genders Diabetes Dataset",
                 fontsize=14, fontweight='bold')

    # Pie — Gender split
    gender_counts = df['Gender'].value_counts()
    axes[0].pie(gender_counts, labels=gender_counts.index,
                colors=[BLUE, RED], autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor':'white','linewidth':2})
    axes[0].set_title("Gender Distribution", fontweight='bold')

    # Bar — Diabetes by Gender
    gender_diab = df.groupby(['Gender','Outcome']).size().unstack()
    gender_diab.plot(kind='bar', ax=axes[1], color=[GREEN, RED],
                     edgecolor='white', rot=0)
    axes[1].set_title("Diabetes by Gender", fontweight='bold')
    axes[1].set_xlabel("Gender")
    axes[1].set_ylabel("Count")
    axes[1].legend(['Not Diabetic','Diabetic'])
    axes[1].spines[['top','right']].set_visible(False)

    # Box — Glucose by Gender and Outcome
    colors_box = {(0,'Female'):'#90CAF9',(1,'Female'):'#EF9A9A',
                  (0,'Male'):'#A5D6A7',(1,'Male'):'#FFCC80'}
    for i, (gender, grp) in enumerate(df.groupby('Gender')):
        for j, (outcome, sub) in enumerate(grp.groupby('Outcome')):
            label = f"{gender} - {'Diabetic' if outcome==1 else 'Healthy'}"
            axes[2].boxplot(sub['Glucose'], positions=[i*3+j],
                           patch_artist=True,
                           boxprops=dict(facecolor=colors_box.get((outcome,gender),'gray')),
                           medianprops=dict(color='black',linewidth=2),
                           widths=0.6)
    axes[2].set_xticks([0,1,3,4])
    axes[2].set_xticklabels(['F-Healthy','F-Diabetic','M-Healthy','M-Diabetic'], fontsize=8)
    axes[2].set_title("Glucose by Gender & Outcome", fontweight='bold')
    axes[2].set_ylabel("Glucose mg/dL")
    axes[2].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig("outputs_gender/01_gender_overview.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    # Chart 2 — Feature distributions
    features = ['Age','Glucose','BloodPressure','BMI',
                'Insulin','SkinThickness','DiabetesPedigreeFunction','Pregnancies']
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle("Feature Distributions by Outcome", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
    for i, feat in enumerate(features):
        ax = fig.add_subplot(gs[i//4, i%4])
        for outcome, color, label in [(0,BLUE,'Not Diabetic'),(1,RED,'Diabetic')]:
            ax.hist(df[df['Outcome']==outcome][feat], bins=20,
                    alpha=0.65, color=color, label=label, edgecolor='white')
        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.spines[['top','right']].set_visible(False)
    plt.savefig("outputs_gender/02_feature_distributions.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("        Charts saved!")

    # ── STEP 3: Preprocessing ─────────────────────────────────
    print("  [3/6] Preprocessing data...")

    # Encode Gender: Male=1, Female=0
    df_clean = df.copy()
    df_clean['Gender'] = LabelEncoder().fit_transform(df_clean['Gender'])

    # Impute zeros
    zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)
    imputer = SimpleImputer(strategy='median')
    df_clean[zero_cols] = imputer.fit_transform(df_clean[zero_cols])

    feature_cols = ['Gender','Age','Pregnancies','Glucose','BloodPressure',
                    'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']
    X = df_clean[feature_cols]
    y = df_clean['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"        Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"        Features: {list(feature_cols)}")

    # ── STEP 4: Train Logistic Regression ────────────────────
    print("  [4/6] Training Logistic Regression...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=42),
        {'C':[0.01,0.1,1,10], 'solver':['lbfgs','liblinear'], 'penalty':['l2']},
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    lr_grid.fit(X_train_sc, y_train)
    lr_model = lr_grid.best_estimator_
    y_pred_lr = lr_model.predict(X_test_sc)
    y_prob_lr = lr_model.predict_proba(X_test_sc)[:,1]
    lr_acc = accuracy_score(y_test, y_pred_lr) * 100
    lr_auc = roc_auc_score(y_test, y_prob_lr) * 100
    print(f"        Accuracy: {lr_acc:.1f}%  AUC: {lr_auc:.1f}%")

    # ── STEP 5: Train Random Forest ───────────────────────────
    print("  [5/6] Training Random Forest (takes ~2-3 min)...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        {'n_estimators':[100,200], 'max_depth':[None,5,10],
         'min_samples_split':[2,5], 'max_features':['sqrt','log2']},
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    rf_grid.fit(X_train_sc, y_train)
    rf_model = rf_grid.best_estimator_
    y_pred_rf = rf_model.predict(X_test_sc)
    y_prob_rf = rf_model.predict_proba(X_test_sc)[:,1]
    rf_acc = accuracy_score(y_test, y_pred_rf) * 100
    rf_auc = roc_auc_score(y_test, y_prob_rf) * 100
    print(f"        Accuracy: {rf_acc:.1f}%  AUC: {rf_auc:.1f}%")

    # ── STEP 6: Evaluation Charts ─────────────────────────────
    print("  [6/6] Saving evaluation charts & models...")

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight='bold')
    for ax, y_pred, title, cmap in zip(axes,
        [y_pred_lr, y_pred_rf],
        ['Logistic Regression', 'Random Forest'], ['Blues','Reds']):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=['Not Diabetic','Diabetic'],
                    yticklabels=['Not Diabetic','Diabetic'],
                    linewidths=1, linecolor='white', annot_kws={"size":12})
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs_gender/03_confusion_matrices.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    # ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    for name, y_prob, color in [('Logistic Regression',y_prob_lr,BLUE),
                                  ('Random Forest',y_prob_rf,RED)]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC={auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.07, color=color)
    ax.plot([0,1],[0,1],'k--', lw=1.2, label='Random (AUC=0.500)')
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(alpha=0.25)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs_gender/04_roc_curves.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    # Feature Importance
    fi = pd.Series(rf_model.feature_importances_,
                   index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    colors = [RED if v==fi.max() else '#90CAF9' for v in fi]
    fi.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    for i, v in enumerate(fi):
        ax.text(v+0.001, i, f" {v:.4f}", va='center', fontsize=9)
    ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance Score"); ax.grid(axis='x', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs_gender/05_feature_importance.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    # Save models
    joblib.dump(lr_model, "outputs_gender/model_lr.pkl")
    joblib.dump(rf_model, "outputs_gender/model_rf.pkl")
    joblib.dump(scaler,   "outputs_gender/scaler.pkl")
    joblib.dump(imputer,  "outputs_gender/imputer.pkl")
    joblib.dump(feature_cols, "outputs_gender/features.pkl")

    print()
    print("  ✅ Training complete!")
    print(f"  Logistic Regression → Accuracy: {lr_acc:.1f}%  AUC: {lr_auc:.1f}%")
    print(f"  Random Forest       → Accuracy: {rf_acc:.1f}%  AUC: {rf_auc:.1f}%")
    print()

    return lr_model, rf_model, scaler, imputer, feature_cols


# ══════════════════════════════════════════════════════════════
# PART 2 — USER INPUT
# ══════════════════════════════════════════════════════════════
def get_input(label, min_val, max_val, hint=""):
    while True:
        try:
            val = input(f"  {label}{' ('+hint+')' if hint else ''}: ").strip()
            if val == "":
                print(f"    Please enter a value between {min_val} and {max_val}")
                continue
            val = float(val)
            if val < min_val or val > max_val:
                print(f"    Must be between {min_val} and {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("    Invalid! Please enter a number.")


def predict_patient(values, lr_model, rf_model, scaler, imputer, feature_cols, model_type='rf'):
    input_df = pd.DataFrame([values], columns=feature_cols)
    zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    input_df[zero_cols] = input_df[zero_cols].replace(0, np.nan)
    input_df[zero_cols] = imputer.transform(input_df[zero_cols])
    scaled  = scaler.transform(input_df)
    model   = rf_model if model_type=='rf' else lr_model
    pred    = model.predict(scaled)[0]
    prob    = model.predict_proba(scaled)[0][1]
    risk    = 'HIGH' if prob>0.6 else 'MEDIUM' if prob>0.4 else 'LOW'
    label   = 'DIABETIC' if pred==1 else 'NOT DIABETIC'
    return label, prob, risk


def show_result(label, prob, risk, model_name, gender):
    print()
    print("  " + "═"*52)
    print("    PREDICTION RESULT")
    print("  " + "═"*52)
    print(f"    Patient Gender : {gender}")
    if label == "DIABETIC":
        print("    Diagnosis      :  ⚠  DIABETIC")
    else:
        print("    Diagnosis      :  ✓  NOT DIABETIC")
    print(f"    Probability    :  {prob*100:.1f}%")
    if risk == "HIGH":
        print("    Risk Level     :  🔴 HIGH   — Please consult a doctor")
    elif risk == "MEDIUM":
        print("    Risk Level     :  🟡 MEDIUM — Monitor your health")
    else:
        print("    Risk Level     :  🟢 LOW    — Keep up healthy habits!")
    print(f"    Model Used     :  {model_name}")
    print("  " + "═"*52)
    bar_len = 40
    filled  = int(prob * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)
    print(f"    Risk : [{bar}] {prob*100:.1f}%")
    print("  " + "═"*52)
    print()
    print("    ⚕  Educational purposes only.")
    print("       Always consult a qualified doctor.")
    print()


def prediction_loop(lr_model, rf_model, scaler, imputer, feature_cols):
    while True:
        print()
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║     ENTER PATIENT DETAILS                        ║")
        print("  ╚══════════════════════════════════════════════════╝")
        print()
        print("  Tip: Enter 0 for unknown values — auto-filled!")
        print("  " + "-"*52)
        print()

        # Gender selection
        print("  Select Gender:")
        print("  [1] Male")
        print("  [2] Female")
        while True:
            g = input("  Enter 1 or 2: ").strip()
            if g == "1":
                gender_val = 1
                gender_name = "Male"
                break
            elif g == "2":
                gender_val = 0
                gender_name = "Female"
                break
            else:
                print("  Please enter 1 or 2")
        print()

        age            = get_input("Age in years           (1–120)",       1,  120)

        # Pregnancies only for females
        if gender_val == 0:
            pregnancies = get_input("Pregnancies            (0–15)",        0,   15)
        else:
            pregnancies = 0
            print("  Pregnancies            : 0  (auto-set for Male)")

        glucose        = get_input("Glucose mg/dL          (0–300)",       0,  300, "normal: 70-99")
        blood_pressure = get_input("Blood Pressure mmHg    (0–200)",       0,  200, "normal: 60-80")
        skin_thickness = get_input("Skin Thickness mm      (0–100)",       0,  100, "0 = unknown")
        insulin        = get_input("Insulin μU/mL          (0–900)",       0,  900, "0 = unknown")
        bmi            = get_input("BMI                    (0–70)",        0,   70, "normal: 18.5-24.9")
        dpf            = get_input("Diabetes Pedigree Fn   (0.0–2.5)",   0.0, 2.5, "family history")

        values = [gender_val, age, pregnancies, glucose, blood_pressure,
                  skin_thickness, insulin, bmi, dpf]

        print()
        print("  Select model:")
        print("  [1] Random Forest        — Higher accuracy")
        print("  [2] Logistic Regression  — More transparent")
        print()
        while True:
            choice = input("  Enter 1 or 2 (default=1): ").strip()
            if choice in ["","1"]:
                model_type = 'rf'; model_name = 'Random Forest'; break
            elif choice == "2":
                model_type = 'lr'; model_name = 'Logistic Regression'; break
            else:
                print("  Please enter 1 or 2")

        print()
        print("  Analyzing...")
        label, prob, risk = predict_patient(values, lr_model, rf_model,
                                            scaler, imputer, feature_cols, model_type)
        show_result(label, prob, risk, model_name, gender_name)

        again = input("  Predict another patient? (yes / no): ").strip().lower()
        if again not in ["yes","y"]:
            print()
            print("  Thank you for using the Diabetes Prediction System!")
            print()
            break


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    models_exist = all(os.path.exists(f"outputs_gender/{f}") for f in [
        "model_rf.pkl","model_lr.pkl","scaler.pkl","imputer.pkl","features.pkl"
    ])

    if models_exist:
        print()
        print("  ✅ Models already trained — loading instantly...")
        lr_model     = joblib.load("outputs_gender/model_lr.pkl")
        rf_model     = joblib.load("outputs_gender/model_rf.pkl")
        scaler       = joblib.load("outputs_gender/scaler.pkl")
        imputer      = joblib.load("outputs_gender/imputer.pkl")
        feature_cols = joblib.load("outputs_gender/features.pkl")
        print("  Ready!")
    else:
        lr_model, rf_model, scaler, imputer, feature_cols = train_models()

    prediction_loop(lr_model, rf_model, scaler, imputer, feature_cols)