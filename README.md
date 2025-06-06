# Customer Churn Prediction with Machine Learning

Work with a data set with US Bank to predict how likely a customer is to churn or no longer be a customer at the bank. 
All U.S banks track this customer data, and if they can identify which customers are likely to churn and reach out to them beforehand, they can retain those customers and ensure they stay loyal to the bank.
This project involves classical machine learning, where multiple machine learning models are trained and average their predictions together to determine how likely a customer is to churn.
This project will also involve  generative AI. We will use Llama 3.1 through Groq to explain the model's prediction so it can be easily interpreted and understand why the customer likely is to churn.
We will also use Llama 3.1 to generate a personalized email to send to them to incentivize them to stay at the bank.

### Models used:
 - Random Forest
 - XGBoost
 - K Nearest Neighbors
 - Decision Tree
 - Naive Bayes
 - SVM
 - Voting Classifiers
 - XGBoost SMOTE

---

## Customer Churn Prediction with Machine Learning

### User Stories

#### REQUIRED (10pts)
- [X] Train ML models on the Kaggle dataset to predict customer churn

- [X] Build the web app using Replit to get the model's predictions given customer data

- [X] Use an LLM to explain the model's predictions and generate a personalized email for the customer

#### BONUS
- [ ] Retrain the ML models with different feature engineering and data pre-processing techniques to increase accuracy

- [ ] Train different ML models on the data and see how they perform. E.g. GradientBoostingClassifier, StackingClassifier

- [ ] Use different LLMs and prompting techniques to generate better explanations of the model's predictions and emails to customers

- [ ] Host the ML models on the cloud to be served via an API so that they can be used in other web apps

- [ ] Train ML models on a different dataset and see what's the highest accuracy you can achieve

### Notes
Some of the classifiers give an error because they were not able to find the function called (voting).
Unable to host on streamlit, confict with importing dotenv

### Video
[Youtube](https://youtu.be/Q-TcfAkj5Eg)

### Open-source libraries used
- [SkLearn](https://scikit-learn.org/stable/) -  Machine learning in python
- [GROQ](https://groq.com/) - fast AI inference (Generative AI)
- [Llama](https://www.llama.com/) - Meta's open-source AI model (Generative AI)
- [Open AI](https://openai.com/) - Open AI, AI model (Generative AI)
