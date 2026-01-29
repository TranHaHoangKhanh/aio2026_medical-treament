import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# PART 1: HELPER FUNCTIONS (Ex 1-6)
# ==========================================

def proportion_treated(df):
    """Ex 1: Proportion of treated patients"""
    return sum(df.TRTMT == 1) / len(df.TRTMT)

def event_rate(df):
    """Ex 2: Empirical death rates"""
    treated_prob = sum((df.TRTMT == 1) & (df.outcome == 1)) / sum((df.TRTMT == 1))
    control_prob = sum((df.TRTMT == 0) & (df.outcome == 1)) / sum((df.TRTMT == 0))
    return treated_prob, control_prob

def extract_treatment_effect(lr, data):
    """Ex 3: Extract Theta and Odds Ratio"""
    coeffs = {data.columns[i]: lr.coef_[0][i] for i in range(len(data.columns))}
    theta_TRTMT = coeffs['TRTMT']
    TRTMT_OR = np.exp(theta_TRTMT)
    return theta_TRTMT, TRTMT_OR

def OR_to_ARR(p, OR):
    """Ex 4: Theoretical ARR from Odds Ratio"""
    odds_baseline = p / (1 - p)
    odds_trtmt = OR * odds_baseline
    p_trtmt = odds_trtmt / (1 + odds_trtmt)
    return p - p_trtmt

def get_base_risks(X, lr_model):
    """Ex 5: Calculate Baseline Risk (Risk if Untreated)"""
    X_copy = X.copy()
    X_copy.TRTMT = 0
    return lr_model.predict_proba(X_copy)[:, 1]

def lr_ARR_quantile(X, y, lr):
    """Ex 6: Empirical ARR by Quantile"""
    X = X.copy()
    baseline_risk = get_base_risks(X, lr)
    
    df = X.copy()
    df['baseline_risk'] = baseline_risk
    df['risk_group'] = pd.qcut(baseline_risk, 10, duplicates='drop')
    df['y'] = y
    
    # Calculate observed death rates in groups
    baseline_mean = df[df.TRTMT == 0].groupby('risk_group', observed=False)['y'].mean()
    treatment_mean = df[df.TRTMT == 1].groupby('risk_group', observed=False)['y'].mean()
    
    # Empirical ARR = Control Rate - Treated Rate
    arr_actual = baseline_mean - treatment_mean
    mean_risk = df.groupby('risk_group', observed=False)['baseline_risk'].mean()
    
    return mean_risk, arr_actual

# ==========================================
# PART 2: C-STATISTIC FOR BENEFIT (Ex 7, 8, 11)
# ==========================================

def c_statistic_for_benefit(y_true, T_true, pred_TE):
    """
    Ex 7, 8 & 11: Calculates concordance between predicted benefit and observed benefit.
    Measures how well the model identifies WHO benefits.
    """
    y = np.array(y_true)
    T = np.array(T_true)
    TE = np.array(pred_TE)
    
    idx_T = np.where(T == 1)[0]
    idx_C = np.where(T == 0)[0]
    
    concordant = 0
    discordant = 0
    ties = 0
    
    # Compare every Treated patient with every Control patient
    for i in idx_T:
        for j in idx_C:
            # Observed Benefit: 
            # If Control died (1) and Treated lived (0) -> Benefit = +1
            # If Control lived (0) and Treated died (1) -> Harm = -1
            obs_diff = y[j] - y[i] 
            
            if obs_diff == 0:
                continue # Outcomes are the same, skip pair
            
            # Predicted Benefit Difference
            # Did the model predict higher benefit for 'i' (who actually benefited) vs 'j'?
            pred_diff = TE[i] - TE[j]
            
            if obs_diff > 0: # Benefit Observed
                if pred_diff > 0: concordant += 1
                elif pred_diff < 0: discordant += 1
                else: ties += 1
            elif obs_diff < 0: # Harm Observed
                if pred_diff < 0: concordant += 1
                elif pred_diff > 0: discordant += 1
                else: ties += 1
                
    total_pairs = concordant + discordant + ties
    if total_pairs == 0: return 0.5
    
    return (concordant + 0.5 * ties) / total_pairs

# ==========================================
# PART 3: T-LEARNER (Ex 9, 10)
# ==========================================

class TLearner:
    def __init__(self):
        # Using Random Forest as base learners
        self.m0 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.m1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
    def fit(self, X, y, T):
        """Ex 9: Fit two separate models"""
        self.m0.fit(X[T==0], y[T==0])
        self.m1.fit(X[T==1], y[T==1])
        
    def predict_cate(self, X):
        """Ex 10: Predict Benefit (Risk Control - Risk Treated)"""
        r0 = self.m0.predict_proba(X)[:, 1]
        r1 = self.m1.predict_proba(X)[:, 1]
        return r0 - r1

# ==========================================
# PART 4: MAIN LOGIC
# ==========================================

def run_full_pipeline(file_obj):
    if file_obj is None: return "Please upload CSV", "", "", "", None, None

    # --- Load Data ---
    try:
        data = pd.read_csv(file_obj.name, index_col=0)
    except:
        return "Error reading CSV", "", "", "", None, None

    data_clean = data.dropna(axis=0)
    y = data_clean.outcome
    X = data_clean.drop('outcome', axis=1)
    
    # Split
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    # --- 1. Basic Stats (Ex 1-2) ---
    stats_txt = (
        f"**Dataset Analysis**\n"
        f"Total Patients: {len(data)}\n"
        f"Proportion Treated: {proportion_treated(data):.2%}\n"
        f"Death Rate (Control): {event_rate(data)[1]:.2%}\n"
        f"Death Rate (Treated): {event_rate(data)[0]:.2%}"
    )

    # --- 2. Logistic Regression (Ex 3-6) ---
    lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000)
    lr.fit(X_dev, y_dev)
    
    theta, OR = extract_treatment_effect(lr, X_dev)
    mean_risks, arr_actual = lr_ARR_quantile(X_dev, y_dev, lr)
    
    # ARR Plot
    fig_arr, ax = plt.subplots(figsize=(6, 4))
    ps = np.arange(0.001, 0.999, 0.001)
    diffs = [OR_to_ARR(p, OR) for p in ps]
    ax.plot(ps, diffs, label='Theoretical (LR)', color='blue')
    ax.scatter(mean_risks, arr_actual, color='red', label='Empirical (Actual)', zorder=5)
    ax.set_xlabel('Baseline Risk'); ax.set_ylabel('ARR (Benefit)')
    ax.set_title('Ex 6: Theoretical vs Empirical Benefit')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    lr_txt = (
        f"**Logistic Regression**\n"
        f"Theta: {theta:.3f}\n"
        f"Odds Ratio: {OR:.3f}\n"
        f"Interpretation: Treatment reduces odds of death by {(1-OR):.1%}."
    )

    # --- 3. T-Learner (Ex 9-10) ---
    # Prepare data for ML (remove TRTMT column)
    X_dev_ml = X_dev.drop('TRTMT', axis=1)
    X_test_ml = X_test.drop('TRTMT', axis=1)
    T_dev = X_dev.TRTMT
    
    tl = TLearner()
    tl.fit(X_dev_ml, y_dev, T_dev)
    
    # Predict Benefit (CATE)
    cate_pred_test = tl.predict_cate(X_test_ml)
    
    # CATE Histogram
    fig_cate, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(cate_pred_test, bins=15, color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Benefit (Risk Reduction)')
    ax2.set_ylabel('Count')
    ax2.set_title('Ex 10: T-Learner Predicted Effects')
    ax2.axvline(0, color='red', linestyle='--', label='No Effect')
    ax2.legend()
    
    ml_txt = (
        f"**T-Learner Results**\n"
        f"Avg Predicted Benefit: {np.mean(cate_pred_test):.3f}\n"
        f"Max Benefit Predicted: {np.max(cate_pred_test):.3f}\n"
        f"Patients Predicted to be Harmed: {sum(cate_pred_test < 0)}"
    )

    # --- 4. Evaluation & Comparison (Ex 7, 8, 11) ---
    # Calculate LR C-stat
    r0_lr = get_base_risks(X_test, lr)
    X_test_T1 = X_test.copy(); X_test_T1.TRTMT = 1
    r1_lr = lr.predict_proba(X_test_T1)[:, 1]
    pred_te_lr = r0_lr - r1_lr # Benefit = Risk_Control - Risk_Treated
    
    c_stat_lr = c_statistic_for_benefit(y_test, X_test.TRTMT, pred_te_lr)
    
    # Calculate T-Learner C-stat
    c_stat_tl = c_statistic_for_benefit(y_test, X_test.TRTMT, cate_pred_test)
    
    eval_txt = (
        f"**Model Comparison (Ex 11)**\n\n"
        f"**1. Logistic Regression C-stat:** {c_stat_lr:.3f}\n"
        f"**2. T-Learner C-stat:** {c_stat_tl:.3f}\n\n"
        f"**Conclusion:** "
        f"{'The T-Learner is better' if c_stat_tl > c_stat_lr else 'Logistic Regression is better'} "
        f"at identifying which patients benefit most."
    )

    return stats_txt, lr_txt, ml_txt, eval_txt, fig_arr, fig_cate

# ==========================================
# PART 5: GRADIO UI
# ==========================================

with gr.Blocks(theme=gr.themes.Soft(), title="Medical AI Ex 1-11") as demo:
    gr.Markdown("# 🏥 Assignment C3M1: Ex 1 to 11")
    gr.Markdown("Full pipeline: Data Stats -> Logistic Regression -> T-Learner -> Comparison")
    
    with gr.Row():
        file_in = gr.File(label="Upload 'levamisole_data.csv'", height=150)
        btn = gr.Button("Run Analysis", variant="primary")
    
    with gr.Tabs():
        with gr.TabItem("📊 Data & LR (Ex 1-6)"):
            with gr.Row():
                with gr.Column():
                    stats_out = gr.Markdown("Waiting...")
                    lr_out = gr.Markdown("Waiting...")
                plot_arr = gr.Plot(label="ARR Plot")

        with gr.TabItem("🤖 T-Learner (Ex 9-10)"):
            with gr.Row():
                ml_out = gr.Markdown("Waiting...")
                plot_cate = gr.Plot(label="CATE Hist")
        
        with gr.TabItem("🏆 Evaluation (Ex 7,8,11)"):
            eval_out = gr.Markdown("### Waiting for results...")

    btn.click(
        fn=run_full_pipeline,
        inputs=file_in,
        outputs=[stats_out, lr_out, ml_out, eval_out, plot_arr, plot_cate]
    )

if __name__ == "__main__":
    demo.launch()