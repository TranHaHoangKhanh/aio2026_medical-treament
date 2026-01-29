import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
import lifelines
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


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
    """
    Compute empirical rate of death within 5 years
    for treated and untreated groups.

    Args:
        df (dataframe): dataframe containing trial results.
                            'TRTMT' column is 1 if patient was treated, 0 otherwise.
                            'outcome' column is 1 if patient died within 5 years, 0 otherwise.

    Returns:
        treated_prob (float): empirical probability of death given treatment
        untreated_prob (float): empirical probability of death given control
    """
    
    treated_prob = sum((df.TRTMT == 1) & (df.outcome == 1)) / sum((df.TRTMT == 1))
    control_prob = sum((df.TRTMT == 0) & (df.outcome == 1)) / sum((df.TRTMT == 0))

    return treated_prob, control_prob

# Process 3: Extract Model Coefficients
def extract_treatment_effect(lr, data):
    
    coeffs = {data.columns[i]: lr.coef_[0][i] for i in range(len(data.columns))}
    
    # get the treatment coefficient
    theta_TRTMT = coeffs['TRTMT']

    # calculate the Odds ratio for treatment
    TRTMT_OR= np.exp(theta_TRTMT)

    ### END CODE HERE ###
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
    # first make a deep copy of the features dataframe to calculate the base risks
    X = X.copy(deep=True)
    
    # Calculate the baseline risks (use the function that you just implemented)
    baseline_risk = base_risks(df.copy(deep=True), lr)

    # Make another deep copy of the features dataframe to store baseline risk, risk_group, and y
    df = X.copy(deep=True)

    

    # bin patients into 10 risk groups based on their baseline risks
    risk_groups = pd.cut(baseline_risk,10)

    # Store the baseline risk, risk_groups, and y into the new dataframe
    df.loc[:, 'baseline_risk'] = baseline_risk
    df.loc[:, 'risk_group'] = risk_groups
    df.loc[:, 'y'] = y_dev

    # select the subset of patients who did not actually receive treatment
    df_baseline = df[df.TRTMT==False]

    # select the subset of patients who did actually receive treatment
    df_treatment = df[df.TRTMT==True]

    # For baseline patients, group them by risk group, select their outcome 'y', and take the mean
    baseline_mean_by_risk_group = df_baseline.groupby('risk_group')['y'].mean()

    # For treatment patients, group them by risk group, select their outcome 'y', and take the mean
    treatment_mean_by_risk_group = df_treatment.groupby('risk_group')['y'].mean()

    # Calculate the absolute risk reduction by risk group (baseline minus treatment)
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
        (0, 0): 0,
        (0, 1): -1,
        (1, 0): 1,
        (1, 1): 0,
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
                if (pred_benefit[i]==pred_benefit[j]): #change to check for risk ties
                    risk_tie_count = risk_tie_count + 1


    # compute c-statistic-for-benefit
    cstat = (concordant_count + (0.5 * risk_tie_count)) / permissible_count

    return cstat

# Process 8: Create patient pairs and calculate c-for benefit 



