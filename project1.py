import streamlit as st
import pandas as pd
import pickle
import numpy as np
def load_model(filename):
    with open(filename,"rb") as file:
        return pickle.load(file)
xgboost_model = load_model("xgb_model.pkl")
naive_bayes_model = load_model("nb_model.pkl")
random_forest_model = load_model("rf_model.pkl")
decision_tree_model = load_model("dt_model.pkl")
svm_model = load_model("svm_model.pkl")
knn_model = load_model("knn_model.pkl")
voting_classifier_model = load_model("voting_clf.pkl")
xgboost_SMOTE_model = load_model("xgboost-SMOTE.pkl")
xgboost_featureEngineered_model = load_model("xgboost-featureEngineered.pkl")

def prepare_input(credit_score, location,gender,age, tenure, balance, num_product, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        "CreditScore" : credit_score,
        "Age" : age,
        "Tenure" : tenure,
        "Balance" : balance,
        "NumOfProducts" : num_product,
        "HasCrCard" : has_credit_card,
        "IsActiveMember" : is_active_member,
        "EstimateSalary" : estimated_salary,
        "Geography_France" : 1 if "location" == "France" else 0,
        "Geography_Germany" : 1 if "location" == "Germany" else 0,
        "Geography_Spain" : 1 if "location" == "Spain" else 0,
        "gender_Male": 1 if "Gender" == "Male" else 0,
        "gender_female": 1 if "Gender" == "Female" else 0,
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

def make_predictions(input_df, input_dict):
    probs = {
        "XGBoost" : xgboost_model.predict_proba(input_df)[0][1],
        "Naive Bayes": naive_bayes_model.predict_proba(input_df)[0][1] ,
        "Random Forest": random_forest_model.predict_proba(input_df)[0][1],
        "Decision Tree": decision_tree_model.predict_proba(input_df)[0][1],
        # "SVM": svm_model.predict_proba(input_df)[0][1],
        "KNN": knn_model.predict_proba(input_df)[0][1] ,
        # "Voting classifier": voting_classifier_model.predict_proba(input_df)[0][1] ,
        # "XGBoost SMOTE": xgboost_SMOTE_model.predict_proba(input_df)[0][1] ,
        # "XGBoost feature Engineered": xgboost_featureEngineered_model.predict_proba(input_df)[0][1] ,
    }
    avg_prob = np.mean(list(probs.values()))
    st.markdown("### Model Probabilities")
    for model, prob in probs.items():
        st.write(f"{model} {prob}")
    st.write(f"Average Probability: {avg_prob}")

st.title("Customer Churn Prediction")
df = pd.read_csv("churn.csv")
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]
selected_customer_option = st.selectbox('Select a customer',customers)

if selected_customer_option:
    selected_customer_Id = int(selected_customer_option.split(" - ")[0])
    print("Selected Customer Id:", selected_customer_Id)
    selected_surname = selected_customer_option.split(" - ")[1]
    print("Surname:", selected_surname)
    selected_customer = df.loc[df['CustomerId'] == selected_customer_Id].iloc[0]
    print('Selected Customer:', selected_customer)
    
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value = 300,
            max_value = 850,
            value = int(selected_customer['CreditScore'])
        )
        location = st.selectbox(
            "Location", ["Spain","France", "Germany"],
            index=["Spain","France", "Germany"].index(
                selected_customer["Geography"]
            )
        )
        gender = st.radio("Gender", ["Male", "Female"],
                          index =0 if selected_customer["Gender"] == 'Male' else 1)
        age = st.number_input(
            "Age",
            min_value = 18,
            max_value = 100,
            value = int(selected_customer['Age'])
        )
        tenure = st.number_input(
            "Tenure (years)",
            min_value = 0,
            max_value = 50,
            value = int(selected_customer['Tenure'])
        )
    with col2:
        balance = st.number_input(
            "Balance",
            min_value = 0.0,
            value = float(selected_customer['Balance'])
        )
        num_products = st.number_input(
            "Number of Products",
            min_value = 0,
            max_value = 50,
            value = int(selected_customer['NumOfProducts'])
        )
        has_credit_card = st.checkbox(
            "Has Credit Card",
            value = bool(selected_customer['HasCrCard'])
        )
        is_active_member = st.checkbox(
            "Is Active member",
            value = bool(selected_customer['IsActiveMember'])
        )
        estimated_salary = st.number_input(
            "Estimated",
            min_value = 0.0,
            value = float(selected_customer['EstimatedSalary'])
        )
    input_df, input_dict = prepare_input(credit_score, location, gender,age, tenure,balance, num_products, has_credit_card, is_active_member,estimated_salary)
    make_predictions(input_df, input_dict)
 