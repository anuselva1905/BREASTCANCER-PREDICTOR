import base64
import warnings
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# PAGE CONFIG
st.set_page_config(
    page_title="Breast Cancer Predictor", 
    page_icon="🔬", 
    layout='centered', 
    initial_sidebar_state='auto'
)

st.title('Breast Cancer Prediction Using Artificial Intelligence 🤖')

# ----------------------------- INTRO -----------------------------
'''
This web app uses machine learning to predict whether a person has breast cancer using some of their clinical data.

❗ **Not a diagnostic tool**  
This is just a demo application of machine learning.

🚧 **Limitations**  
The dataset used to train the model is small and there might be better variables that could have been used.

*Original dataset available here: [Breast Cancer Coimbra](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra)*

*Details: [Patrício et al., BMC Cancer 18, 29 (2018)](https://doi.org/10.1186/s12885-017-3877-1)*
'''

# ---------------------------- LOAD DATA ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

data = load_data()

st.write('###### *click to show/hide')
if st.checkbox('Data'):
    '''## Data Used for Training'''
    data
    st.write(f'###### {data.shape[0]} rows and {data.shape[1]} columns.')

    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download as CSV</a>'
        return href

    st.markdown(filedownload(data), unsafe_allow_html=True)
    st.write('Under the classification column: 0 = healthy, 1 = breast cancer')

    st.header('Feature Correlations')
    corr_features = data.drop('Classification', axis=1).apply(lambda x: x.corr(data['Classification']))
    corr_df = pd.DataFrame(corr_features.sort_values(ascending=False), columns=['correlation to breast cancer classification'])
    st.table(corr_df)
    st.markdown('---')

# ---------------------------- LOAD MODEL ----------------------------
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# ------------------------- MODEL METRICS ---------------------------
if st.checkbox('Model Metrics'):
    features = data.drop('Classification', axis=1)
    training_features, testing_features, training_target, testing_target = \
        train_test_split(features, data['Classification'], random_state=42, test_size=0.2)

    y_pred = model.predict(testing_features)
    y_true = testing_target

    accuracy = accuracy_score(y_true, y_pred) * 100
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tnr = tn / (tn + fn) * 100
    fpr = fp / (tp + fp) * 100
    fnr = fn / (tn + fn) * 100
    tpr = tp / (fp + tp) * 100

    '''# 📊'''
    st.write(f"Accuracy: {accuracy:.2f}%")
    st.write(f"True negative rate: {tnr:.2f}%")
    st.write(f"False Positive rate: {fpr:.2f}%")
    st.write(f"False Negative rate: {fnr:.2f}%")
    st.write(f"True Positive rate: {tpr:.2f}%")

st.markdown('---')
st.title("Get a Prediction🔮")
'''
###### Note: Use this on Desktop for the best experience. If you're on mobile, use the arrow on the upper left to access the sidebar.
'''

# -------------------------- INPUT SECTION --------------------------
input_type = st.sidebar.selectbox('Input Method', ['Move Sliders', 'Enter Values'], index=1)

if input_type == 'Enter Values':
    BMI = st.sidebar.number_input('BMI (kg/m2)', format="%.4f", step=0.0001)
    Glucose = st.sidebar.number_input('Glucose (mg/dL)', format="%.0f")
    Insulin = st.sidebar.number_input('Insulin (µU/mL)', format="%.4f", step=0.0001)
    HOMA = st.sidebar.number_input('HOMA', format="%.4f", step=0.0001)
    Resistin = st.sidebar.number_input('Resistin (ng/mL)', format="%.4f", step=0.0001)

    st.sidebar.info("💡 Change to 'Move Sliders' to explore the model predictions interactively.")

    '''
    ## These are the values you entered
    '''
    st.write(f"**BMI**: {BMI:.4f} kg/m2")
    st.write(f"**Glucose**: {Glucose:.0f} mg/dL")
    st.write(f"**Insulin**: {Insulin:.4f} µU/mL")
    st.write(f"**HOMA**: {HOMA:.4f}")
    st.write(f"**Resistin**: {Resistin:.4f} ng/mL")

    if st.button("submit ✅"):
        dataframe = pd.DataFrame({
            'BMI': [BMI],
            'Glucose': [Glucose],
            'Insulin': [Insulin],
            'HOMA': [HOMA],
            'Resistin ': [Resistin]
        })

elif input_type == 'Move Sliders':
    BMI = st.sidebar.slider('BMI (kg/m2)', 10.0, 50.0, float(data['BMI'][0]), step=0.01)
    Glucose = st.sidebar.slider('Glucose (mg/dL)', 25, 250, int(data['Glucose'][0]), step=1)
    Insulin = st.sidebar.slider('Insulin (µU/mL)', 1.0, 75.0, float(data['Insulin'][0]), step=0.01)
    HOMA = st.sidebar.slider('HOMA', 0.25, 30.0, float(data['HOMA'][0]), step=0.01)
    Resistin = st.sidebar.slider('Resistin (ng/mL)', 1.0, 100.0, float(data['Resistin'][0]), step=0.01)

    dataframe = pd.DataFrame({
        'BMI': [BMI],
        'Glucose': [Glucose],
        'Insulin': [Insulin],
        'HOMA': [HOMA],
        'Resistin ': [Resistin]
    })

    '''
    ## Move the Sliders to Update Results ↔
    '''

# ---------------------------- PREDICTION ----------------------------
try:
    '''
    ## Results 📋
    '''
    prediction = model.predict(dataframe)[0]
    probas = model.predict_proba(dataframe)[0]

    if prediction == 0:
        st.success('Prediction: **NO BREAST CANCER 🙌**')
    else:
        st.error('Prediction: **BREAST CANCER PRESENT**')

    st.table(pd.DataFrame({
        'healthy': [f"{probas[0]*100:.2f}%"],
        'has breast cancer': [f"{probas[1]*100:.2f}%"]
    }, index=['Probability']))

except Exception as e:
    st.warning('*Press submit to compute results*')
    st.write(f'Debug info: {e}')

# ---------------------------- FOOTER ----------------------------
st.markdown('---')
st.info('Source code available [here](https://github.com/batmanscode/breastcancer-predictor) — contributions welcome! 😊')

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
