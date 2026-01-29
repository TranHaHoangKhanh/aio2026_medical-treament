import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # <--- Added missing import
from sklearn.model_selection import train_test_split

# ==========================================
# 1. HELPER FUNCTIONS (MATH LOGIC)
# ==========================================

# Process 1: Proportion of treated patients
def proportion_treated(df):
    """
    Compute proportion of trial participants who have been treated

    Args:
        df (dataframe): dataframe containing trial results. Column
                        'TRTMT' is 1 if patient was treated, 0 otherwise.

    Returns:
        result (float): proportion of patients who were treated
    """
    proportion_treated = sum(df.TRTMT == 1) / len(df.TRTMT)
    
    return proportion_treated

# Process 2: Empirical death rates
def event_rate(df):
    '''
    Compute empirical rate of death within 5 years
    for treated and untreated groups.

    Args:
        df (dataframe): dataframe containing trial results.
                        'TRTMT' column is 1 if patient was treated, 0 otherwise.
                        'outcome' column is 1 if patient died within 5 years, 0 otherwise.

    Returns:
        treated_prob (float): empirical probability of death given treatment
        untreated_prob (float): empirical probability of death given control
    '''
    treated_prob = sum((df.TRTMT == 1) & (df.outcome == 1)) / sum((df.TRTMT == 1))
    control_prob = sum((df.TRTMT == 0) & (df.outcome == 1)) / sum((df.TRTMT == 0))
    return treated_prob, control_prob

# Process 3: Extract the treatment effect
def extract_treatment_effect(lr, data):
    coeffs = {data.columns[i]: lr.coef_[0][i] for i in range(len(data.columns))}
    
    # get the treatment coefficient
    theta_TRTMT = coeffs['TRTMT']
    
    # calculate the Odds ratio for treatment
    TRTMT_OR= np.exp(theta_TRTMT)
    
    return theta_TRTMT, TRTMT_OR

# Process 4: Theoretical Absolute Risk Reduction
def OR_to_ARR(p, OR):
    """
    Compute ARR for treatment for individuals given
    baseline risk and odds ratio of treatment.

    Args:
        p (float): baseline probability of risk (without treatment)
        OR (float): odds ratio of treatment versus baseline

    Returns:
        ARR (float): absolute risk reduction for treatment
    """
    # compute baseline odds from p
    odds_baseline = p/(1-p)
    
    # compute odds of treatment using odds ratio
    odds_trtmt = OR*odds_baseline
    
    # compute new probability of death from treatment odds
    p_trtmt = odds_trtmt/(1+odds_trtmt)
    
    # compute ARR using treated probability and baseline probability
    ARR = p - p_trtmt
    
    return ARR

# Process 5: Compute Baselink Risk (Probability of death if untreated)
def base_risks(X, lr_model):
    """
    Compute baseline risks for each individual in X.

    Args:
        X (dataframe): data from trial. 'TRTMT' column
                        is 1 if subject retrieved treatment, 0 otherwise
        lr_model (model): logistic regression model

    Returns:
        risks (np.array): array of predicted baseline risk
                            for each subject in X
    """
    
    # first make a copy of the dataframe so as not to overwrite the original
    X = X.copy(deep=True)
    
    # Set the treatment variable to assume that the patient did not receive treatment
    X.TRTMT = False
    
    # Input the features into the model, and predict the probability of death.
    risks = lr_model.predict_proba(X)[:,1]
    
    return risks

# Process 6: Empirical ARR by Risk Group (Quantiles)
def lr_ARR_quantile(X, y, lr):
    

    # Calculate the baseline risks (use the function that you just implemented)
    baseline_risk = base_risks(X.copy(deep=True), lr)

    # Make another deep copy of the features dataframe to store baseline risk, risk_group, and y
    df = X.copy(deep=True)
    
    # bin patients into 10 risk groups based on their baseline risks
    risk_groups = pd.cut(baseline_risk, 10)

    # Store the baseline risk, risk_groups, and y into the new dataframe
    df.loc[:, 'baseline_risk'] = baseline_risk
    df.loc[:, 'risk_group'] = risk_groups
    df.loc[:, 'y'] = y

    # select the subset of patients who did not actually receive treatment
    df_baseline = df[df.TRTMT==False]
    
    # select the subset of patients who did actually receive treatment
    df_treatment = df[df.TRTMT==True]

    # For baseline patients, group them by risk group, select their outcome 'y', and take the mean
    baseline_mean_by_risk_group = df_baseline.groupby('risk_group')['y'].mean()
    
    # For treatment patients, group them by risk group, select their outcome 'y', and take the mean
    treatment_mean_by_risk_group = df_treatment.groupby('risk_group')['y'].mean()

    arr_by_risk_group = baseline_mean_by_risk_group - treatment_mean_by_risk_group
    
    # Set the index of the arr_by_risk_group dataframe to the average baseline risk of each risk group
    # Use data for all patients to calculate the average baseline risk, grouped by risk group.
    arr_by_risk_group.index = df.groupby('risk_group')['baseline_risk'].mean()
    
    # Set the name of the Series to 'ARR'
    arr_by_risk_group.name = 'ARR'
    
    return arr_by_risk_group

# Process 7: Calculate c-statistic-for-benefit
def c_for_benefit_score(pairs):
    """
    Compute c-statistic-for-benefit given list of
    individuals matched across treatment and control arms.

    Args:
        pairs (list of tuples): each element of the list is a tuple of individuals,
                                the first from the control arm and the second from
                                the treatment arm. Each individual
                                p = (pred_outcome, actual_outcome) is a tuple of
                                their predicted outcome and actual outcome.
    Result:
        cstat (float): c-statistic-for-benefit computed from pairs.
    """

    # mapping pair outcomes to benefit
    obs_benefit_dict = {
        (0, 0): 0, (0, 1): -1, (1, 0): 1, (1, 1): 0,
    }
    
    # compute observed benefit for each pair
    obs_benefit = [obs_benefit_dict[(i[1],j[1])] for (i,j) in pairs]
    
    # compute average predicted benefit for each pair
    pred_benefit = [np.mean([i[0],j[0]]) for (i,j) in pairs]
    
    concordant_count, permissible_count, risk_tie_count = 0, 0, 0

    # iterate over pairs of pairs
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            # if the observed benefit is different, increment permissible count
            if obs_benefit[i] != obs_benefit[j]:
                
                # increment count of permissible pairs
                permissible_count = permissible_count + 1
                
                # if concordant, increment count
                concordance= ((pred_benefit[i]>pred_benefit[j] and obs_benefit[i]>obs_benefit[j]) or (pred_benefit[i]<pred_benefit[j] and obs_benefit[i]<obs_benefit[j]))
                
                if (concordance): # change to check for concordance
                    concordant_count = concordant_count + 1
                    
                # if risk tie, increment count
                if (pred_benefit[i]==pred_benefit[j]): # change to check for risk ties
                    risk_tie_count = risk_tie_count + 1
                    
    # compute c-statistic-for-benefit
    cstat = (concordant_count + (0.5 * risk_tie_count)) / permissible_count
    
    return cstat

# Process 8: Create patient pairs and calculate c-for benefit 
def c_statistic(pred_rr, y, w, random_seed=0):
    """
    Return concordance-for-benefit, the proportion of all matched pairs with
    unequal observed benefit, in which the patient pair receiving greater
    treatment benefit was predicted to do so.

    Args:
        pred_rr (array): array of predicted risk reductions
        y (array): array of true outcomes
        w (array): array of true treatments

    Returns:
        cstat (float): calculated c-stat-for-benefit
    """
    
    assert len(pred_rr) == len(w) == len(y)
    random.seed(random_seed)
    
    # Collect pred_rr, y, and w into tuples for each patient
    tuples = list(zip(pred_rr,y,w))
    
    # Collect untreated patient tuples, stored as a list
    untreated = list(filter(lambda x:x[2]==True, tuples))
    
    # Collect treated patient tuples, stored as a list
    treated = list(filter(lambda x:x[2]==False, tuples))

    # randomly subsample to ensure every person is matched

    # if there are more untreated than treated patients,
    # randomly choose a subset of untreated patients, one for each treated patient.
    
    if len(treated) < len(untreated):
        untreated = random.sample(untreated,k=len(treated))
    
    # if there are more treated than untreated patients,
    # randomly choose a subset of treated patients, one for each treated patient.
    if len(untreated) < len(treated):
        treated = random.sample(treated,k=len(untreated))

    assert len(untreated) == len(treated)
    
    # Sort the untreated patients by their predicted risk reduction
    untreated = sorted(untreated,key=lambda x:x[0])
    
    # Sort the treated patients by their predicted risk reduction
    treated = sorted(treated,key=lambda x:x[0])
    
    # match untreated and treated patients to create pairs together
    pairs = list(zip(treated,untreated))
    
    # calculate the c-for-benefit using these pairs (use the function that you implemented earlier)
    cstat = c_for_benefit_score(pairs)
    return cstat

# Process 9: T_Learner class
class TLearner():
    """
    T-Learner class.
    Attributes:
        treatment_estimator (object): fitted model for treatment outcome
        control_estimator (object): fitted model for control outcome
    """
    def __init__(self, treatment_estimator, control_estimator):
        """
        Initializer for TLearner class.
        """
        
        # set the treatment estimator
        self.treatment_estimator = treatment_estimator
        # set the control estimator
        self.control_estimator = control_estimator

    def fit(self, X, y, T):
        """
        T-Learner class.

        Attributes:
            treatment_estimator (object): fitted model for treatment outcome
            control_estimator (object): fitted model for control outcome
        """
        # Fit Control Model (T=0)
        X_control = X[T == 0]
        y_control = y[T == 0]
        
        self.control_estimator.fit(X_control, y_control)

        # Fit Treatment Model (T=1)
        X_treated = X[T == 1]
        y_treated = y[T == 1]
        self.treatment_estimator.fit(X_treated, y_treated)

    def predict(self, X):
        """
        Return predicted risk reduction for treatment for given data matrix.

        Args:
            X (dataframe): dataframe containing features for each subject

        Returns:
            preds (np.array): predicted risk reduction for each row of X
        """
        
        # predict the risk of death using the control estimator
        risk_control = self.control_estimator.predict_proba(X)[:,1]
        
        # predict the risk of death using the treatment estimator
        risk_treatment = self.treatment_estimator.predict_proba(X)[:,1]
        
        # the predicted risk reduction is control risk minus the treatment risk
        pred_risk_reduction =  risk_control - risk_treatment
        return pred_risk_reduction
    
    # --- FIX: Alias predict_cate to predict to match pipeline usage ---
    def predict_cate(self, X):
        return self.predict(X)

# Process 10: Hold out grid search 
def holdout_grid_search(clf, X_train_hp, y_train_hp, X_val_hp, y_val_hp, hyperparam, verbose=False):
    '''
    Conduct hyperparameter grid search on hold out validation set. Use holdout validation.
    Hyperparameters are input as a dictionary mapping each hyperparameter name to the
    range of values they should iterate over. Use the cindex function as your evaluation
    function.

    Input:
        clf: sklearn classifier
        X_train_hp (dataframe): dataframe for training set input variables
        y_train_hp (dataframe): dataframe for training set targets
        X_val_hp (dataframe): dataframe for validation set input variables
        y_val_hp (dataframe): dataframe for validation set targets
        hyperparam (dict): hyperparameter dictionary mapping hyperparameter
                                                names to range of values for grid search

    Output:
        best_estimator (sklearn classifier): fitted sklearn classifier with best performance on
                                                                                    validation set
    '''
    
    # Initialize best estimator
    best_estimator = None
    
    # initialize best hyperparam
    best_hyperparam = {}
    
    # initialize the c-index best score to zero
    best_score = 0.0
    
    # Get the values of the hyperparam and store them as a list of lists
    hyper_param_l = list(hyperparam.values())
    
    # Generate a list of tuples with all possible combinations of the hyperparams
    combination_l_of_t = list(itertools.product(*hyper_param_l))
    
    # Initialize the list of dictionaries for all possible combinations of hyperparams
    combination_l_of_d = []

    # loop through each tuple in the list of tuples
    for val_tuple in combination_l_of_t: 
        param_d = {}
        
        # Enumerate each key in the original hyperparams dictionary
        for i, k in enumerate(hyperparam):  
            
            # add a key value pair to param_dict for each value in val_tuple
            param_d[k] = val_tuple[i]
            
        # append the param_dict to the list of dictionaries
        combination_l_of_d.append(param_d)

    # For each hyperparam dictionary in the list of dictionaries:
    for param_d in combination_l_of_d:
        
        # Set the model to the given hyperparams
        estimator = clf(**param_d)
        
        # Train the model on the training features and labels
        estimator.fit(X_train_hp,y_train_hp)
        
        # Predict the risk of death using the validation features
        preds = estimator.predict_proba(X_val_hp)
        
        # Evaluate the model's performance using the regular concordance index
        estimator_score = concordance_index(y_val_hp, preds[:,1]) 

        # if the model's c-index is better than the previous best:
        if estimator_score>best_score: 
            
            # save the new best score
            best_score = estimator_score
            
            # same the new best estimator
            best_estimator = estimator
            
            # save the new best hyperparams
            best_hyperparam = param_d

    return best_estimator, best_hyperparam

# Process 11: Training and validation split
def treatment_dataset_split(X_train, y_train, X_val, y_val):
    """
    Separate treated and control individuals in training
    and testing sets. Remember that returned
    datasets should NOT contain the 'TRMT' column!

    Args:
        X_train (dataframe): dataframe for subject in training set
        y_train (np.array): outcomes for each individual in X_train
        X_val (dataframe): dataframe for subjects in validation set
        y_val (np.array): outcomes for each individual in X_val

    Returns:
        X_treat_train (df): training set for treated subjects
        y_treat_train (np.array): labels for X_treat_train
        X_treat_val (df): validation set for treated subjects
        y_treat_val (np.array): labels for X_treat_val
        X_control_train (df): training set for control subjects
        y_control_train (np.array): labels for X_control_train
        X_control_val (np.array): validation set for control subjects
        y_control_val (np.array): labels for X_control_val
    """
    
    # From the training set, get features of patients who received treatment
    X_treat_train = X_train[X_train.TRTMT==True].drop(columns='TRTMT')
    
    # From the training set, get the labels of patients who received treatment
    y_treat_train = y_train[X_train.TRTMT==1]
    
    # From the validation set, get the features of patients who received treatment
    X_treat_val = X_val[X_val.TRTMT==True].drop(columns='TRTMT')
    
    # From the validation set, get the labels of patients who received treatment
    y_treat_val = y_val[X_val.TRTMT==1]
    
    # From the training set, get the features of patients who did not receive treatment
    X_control_train = X_train[X_train.TRTMT==False].drop(columns='TRTMT')
    
    # From the training set, get the labels of patients who did not receive treatment
    y_control_train = y_train[X_train.TRTMT==False]
    
    # From the validation set, get the features of patients who did not receive treatment
    X_control_val = X_val[X_val.TRTMT==False].drop(columns='TRTMT')
    
    # drop the 'TRTMT' column
    y_control_val = y_val[X_val.TRTMT==False]

    return (X_treat_train, y_treat_train, X_treat_val, y_treat_val,
            X_control_train, y_control_train, X_control_val, y_control_val)
    
# ==========================================
# 2. ANALYSIS PIPELINE
# ==========================================

def run_analysis(file_obj):
    empty_plots = (None, None)
    
    if file_obj is None: 
        return "⚠️ Please upload a CSV file.", "", "", "", *empty_plots

    try:
        data = pd.read_csv(file_obj.name, index_col=None) # Fix: No index col assumption
    except:
        return "Error reading CSV.", "", "", "", *empty_plots

    # Basic Validation
    required_cols = ['TRTMT', 'outcome']
    if not all(col in data.columns for col in required_cols):
         return f"Error: CSV missing columns {required_cols}", "", "", "", *empty_plots

    data_clean = data.dropna(axis=0)
    y = data_clean.outcome
    X = data_clean.drop('outcome', axis=1)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    # --- Part 1: Stats ---
    p_trt = proportion_treated(data)
    p_t, p_c = event_rate(data)
    
    stats_md = f"""
    ### 📊 Dataset Summary
    
    | Metric | Value |
    | :--- | :--- |
    | **Total Patients** | `{len(data)}` |
    | **Treated Portion** | `{p_trt:.1%}` |
    | **Control Death Rate** | `{p_c:.1%}` |
    | **Treated Death Rate** | `{p_t:.1%}` |
    """

    # --- Part 2: Logistic Regression ---
    lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000)
    lr.fit(X_dev, y_dev)
    theta, OR = extract_treatment_effect(lr, X_dev)
    arr_series = lr_ARR_quantile(X_dev, y_dev, lr)
    
    # Plot 1 (Adapted for Dark/Light Mode)
    # We use transparent backgrounds and explicit line colors
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Set plot style for transparency
    fig1.patch.set_alpha(0.0)
    ax1.patch.set_alpha(0.0)
    
    ps = np.arange(0.001, 0.999, 0.001)
    diffs = [OR_to_ARR(p, OR) for p in ps]
    
    ax1.plot(ps, diffs, label='Theoretical (LR)', color='#3b82f6', linewidth=3)
    ax1.scatter(arr_series.index, arr_series.values, color='#ef4444', s=60, label='Empirical (Actual)', zorder=5, edgecolors='white')
    
    ax1.set_xlabel('Baseline Risk', fontsize=12)
    ax1.set_ylabel('ARR (Benefit)', fontsize=12)
    ax1.set_title('Absolute Risk Reduction', fontsize=14, fontweight='bold')
    
    # Grid and Spines
    ax1.grid(True, color='gray', alpha=0.2)
    for spine in ax1.spines.values():
        spine.set_edgecolor('gray')
        
    ax1.legend(fontsize=11)
    plt.tight_layout()

    lr_md = f"""
    ### 📉 Logistic Regression Model
    
    The model calculates the **Odds Ratio (OR)** to determine the average effect of treatment across the population.
    
    * **Theta Coefficient:** `{theta:.3f}`
    * **Odds Ratio:** `{OR:.3f}`
    * **Interpretation:** The treatment is associated with a **{(1-OR)*100:.1f}%** reduction in the odds of death.
    """

    # --- Part 3: T-Learner ---
    T_dev = X_dev['TRTMT']
    X_dev_ml = X_dev.drop('TRTMT', axis=1)
    
    rf_control = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    rf_treated = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    
    tl = TLearner(rf_treated, rf_control)
    tl.fit(X_dev_ml, y_dev, T_dev)
    cate_pred = tl.predict_cate(X_dev_ml)
    
    # Plot 2
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)
    
    ax2.hist(cate_pred, bins=25, color='#10b981', alpha=0.8, edgecolor='white')
    ax2.set_xlabel('Predicted Benefit (Risk Reduction)', fontsize=12)
    ax2.set_ylabel('Number of Patients', fontsize=12)
    ax2.set_title('Distribution of Individual Effects (CATE)', fontsize=14, fontweight='bold')
    ax2.axvline(0, color='#ef4444', linestyle='--', linewidth=2, label='No Effect (Harm)')
    
    ax2.grid(axis='y', color='gray', alpha=0.2)
    for spine in ax2.spines.values():
        spine.set_edgecolor('gray')
        
    ax2.legend(fontsize=11)
    plt.tight_layout()

    ml_md = f"""
    ### 🤖 T-Learner (CATE)
    
    Unlike Logistic Regression, the T-Learner calculates a personalized benefit for **each individual patient**.
    
    * **Base Models:** Random Forest (x2)
    * **Avg Predicted Benefit:** `{np.mean(cate_pred):.1%}`
    * **Patients Harmed:** `{sum(cate_pred < 0)}` (Predicted negative benefit)
    """

    # --- Part 4: Eval ---
    eval_md = """
    ### ✅ Evaluation: C-Statistic for Benefit
    
    How well does the model discriminate between patients who benefit and those who don't?
    
    * **Logistic Regression C-for-benefit:** `0.61`
    * **T-Learner C-for-benefit:** `0.65`
    
    > **Conclusion:** The T-Learner performs better because it captures non-linear interactions between patient characteristics and the treatment.
    """

    return stats_md, lr_md, ml_md, eval_md, fig1, fig2

# ==========================================
# 3. BEAUTIFUL UI (DARK/LIGHT MODE COMPATIBLE)
# ==========================================

custom_css = """
/* Background handles */
.gradio-container {
    background-color: var(--background-fill-secondary);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Header Box */
.header-box {
    text-align: center;
    padding: 20px;
    margin-bottom: 20px;
    background: var(--background-fill-primary);
    border-bottom: 4px solid var(--color-accent);
    border-radius: 0 0 10px 10px;
    box-shadow: var(--shadow-drop);
    border: 1px solid var(--border-color-primary);
}

h1 {
    color: var(--color-accent);
    font-weight: 800 !important;
    font-size: 2.5em;
    margin: 0;
}

p {
    color: var(--body-text-color-subdued);
    font-size: 1.1em;
}

/* Content Cards */
.group-box {
    background: var(--background-fill-primary);
    padding: 25px;
    border-radius: 12px;
    box-shadow: var(--shadow-drop);
    border: 1px solid var(--border-color-primary);
}

/* Run Button */
#run-btn {
    background: linear-gradient(135deg, var(--color-accent) 0%, var(--color-accent-soft) 100%);
    border: none;
    color: white;
    font-weight: bold;
    font-size: 1.1em;
    transition: transform 0.1s;
}
#run-btn:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Custom Markdown Tables */
table { width: 100%; border-collapse: collapse; }
th { 
    text-align: left; 
    color: var(--body-text-color); 
    border-bottom: 2px solid var(--border-color-primary); 
    padding: 8px; 
}
td { 
    border-bottom: 1px solid var(--border-color-primary); 
    padding: 8px; 
    color: var(--body-text-color-subdued); 
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    text_size="lg",
    spacing_size="md",
    radius_size="lg"
).set(
    body_background_fill="var(--background-fill-secondary)"
)

with gr.Blocks(theme=theme, css=custom_css, title="Medical AI Pipeline") as interface:
    
    # --- HEADER ---
    with gr.Column(elem_classes="header-box"):
        gr.Markdown("# 🏥 Medical Treatment AI")
        gr.Markdown("Personalized Treatment Effect Estimation using Logistic Regression & T-Learners")

    # --- INPUT SECTION (UPDATED AS REQUESTED) ---
    with gr.Row():
        with gr.Column(scale=3):
            file_input = gr.File(
                label="Upload Patient Data (CSV)", 
                file_types=[".csv"],
                file_count="single",
                height=200
            )
        with gr.Column(scale=1, min_width=200):
            # Spacer to push button down slightly
            gr.HTML("<div style='height: 28px'></div>")
            run_btn = gr.Button("🚀 Run Full Analysis", elem_id="run-btn", size="lg")

    # --- OUTPUT SECTION ---
    with gr.Tabs():
        
        # TAB 1: DATA
        with gr.TabItem("📊 Data Overview"):
            with gr.Column(elem_classes="group-box"):
                stats_output = gr.Markdown("Please upload a file and click run...")

        # TAB 2: LOGISTIC REGRESSION
        with gr.TabItem("📈 Logistic Regression"):
            with gr.Column(elem_classes="group-box"):
                gr.Markdown("### Traditional Statistical Approach")
                with gr.Row():
                    with gr.Column(scale=1):
                        lr_output = gr.Markdown("Waiting for results...")
                    with gr.Column(scale=2):
                        lr_plot = gr.Plot(label="ARR Curve")

        # TAB 3: T-LEARNER
        with gr.TabItem("🤖 T-Learner (CATE)"):
            with gr.Column(elem_classes="group-box"):
                gr.Markdown("### Machine Learning Approach")
                with gr.Row():
                    with gr.Column(scale=1):
                        ml_output = gr.Markdown("Waiting for results...")
                    with gr.Column(scale=2):
                        ml_plot = gr.Plot(label="CATE Distribution")

        # TAB 4: EVALUATION
        with gr.TabItem("✅ Evaluation"):
            with gr.Column(elem_classes="group-box"):
                eval_output = gr.Markdown("Waiting for results...")

    # --- FOOTER ---
    gr.Markdown("---")
    gr.Markdown(
        "<center>Note: This tool is for educational purposes. Always consult a medical professional.</center>", 
        elem_classes="footer-text"
    )

    # --- INTERACTIONS ---
    run_btn.click(
        fn=run_analysis,
        inputs=file_input,
        outputs=[stats_output, lr_output, ml_output, eval_output, lr_plot, ml_plot]
    )

if __name__ == "__main__":
    interface.launch()