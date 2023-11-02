# Heart-Failure-Readmission-Prediction


Aims: Patients visiting the emergency department (ED) or hospitalized for heart failure (HF) are at increased risk for subsequent adverse outcomes, however effective risk stratification remains challenging. We aimed to utilize a machine-learning (ML)-based approach to identify patients with HF at risk of adverse outcomes after an ED visit or hospitalization in a large regional healthcare system.

Methods and Results: Patients visiting the ED or hospitalized with HF between 2002-2016 in Alberta, Canada were included. Outcomes of interest were 30-day and 1-year HF-related ED visits, HF hospital readmission or all-cause mortality. We applied a feature extraction method using deep feature synthesis from multiple sources of health data and compared performance of a gradient boosting algorithm (CatBoost) with logistic regression modelling. The area under receiver operating characteristic curve (AUC-ROC) was used to assess model performance. We included 50,630 patients with 93,552 HF ED visits/hospitalizations. At 30-day follow-up in the external validation cohort, the AUC-ROC for the combined endpoint of HF ED visit, HF hospital readmission or death was for the Catboost and logistic regression models was 74.16 versus 62.25, respectively. At 1-year follow-up corresponding values were 76.80 versus 69.52, respectively. AUC-ROC values for the endpoint of all-cause death alone at 30-days and 1-year follow-up were 83.21 versus 69.53, and 85.73 versus 69.40, for the CatBoost and logistic regression models, respectively. 

Conclusions: ML-based modelling with deep feature synthesis provided superior risk stratification for patients with HF at both 30-days and 1-year follow-up after an ED visit or hospitalization compared with logistic regression. 

Key Words: heart failure; machine learning; artificial intelligence; outcomes; hospital readmission;
