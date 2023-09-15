import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, PrecisionRecallDisplay, precision_score, recall_score 

@st.cache_data
def load_data():
    data = pd.read_csv('./mushrooms.csv')
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data

@st.cache_data
def split_data(df):
    y = df.type
    x = df.drop(columns=['type'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return  x_train, x_test, y_train, y_test


def plot_metrics(metrics_list, y_test, y_pred, labels):
    if 'Confusion Matrix' in metrics_list:
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(cm).plot()
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader('ROC Curve')
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        st.pyplot()

    if 'Precision-Recall curve' in metrics_list:
        st.subheader('Precision-Recall curve')
        PrecisionRecallDisplay.from_predictions(y_test, y_pred)
        st.pyplot()


def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Binary classification Web App')
    st.sidebar.title('Binary classification Web App')
    st.markdown('Are your mushrooms edible or poisonous? 🍄 ')
    st.sidebar.markdown('Are your mushrooms edible or poisonous? 🍄 ')
    data_load_state = st.text('Loading data...')
    df = load_data()
    x_train, x_test, y_train, y_test = split_data(df)
    labels = ['edible', 'poisonous']
    data_load_state.text('Loading data...Done!')

    if st.sidebar.checkbox('Show raw data', False):
        st.subheader('Mushroom data set')
        st.write(df)
    
    st.sidebar.subheader('Choose Classifier')
    classifier = st.sidebar.selectbox('CLassifier', ('Support Vector Machine (SVM)', 'Logistic Regression', 'Random Forest'))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader('Model Hyper Parameters')
        C = st.sidebar.number_input('C (Regularization Parameter)', 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio('Kernels', ('rbf', 'linear'), key='kernel')
        gamma = st.sidebar.radio('Gamma (Kernel cofficients)', ('scale', 'auto'), key='gamma')
        metrics = st.sidebar.multiselect('Select Metrics to plot', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Support Vector Machine (SVM) Results')
            model = SVC(C=C,gamma=gamma, kernel=kernel)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write(f'Accuracy: {accuracy.round(2)}')
            st.write(f'Precision: {precision_score(y_test, y_pred, labels=labels).round(2)}')
            st.write(f'Recall: {recall_score(y_test, y_pred, labels=labels).round(2)}')
            plot_metrics(metrics, y_test, y_pred, labels)


    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Hyper Parameters')
        C = st.sidebar.number_input('C (Regularization Parameter)', 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider('Maximum number of iterations', 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect('Select Metrics to plot', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Logistic Regression Results')
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write(f'Accuracy: {accuracy.round(2)}')
            st.write(f'Precision: {precision_score(y_test, y_pred, labels=labels).round(2)}')
            st.write(f'Recall: {recall_score(y_test, y_pred, labels=labels).round(2)}')
            plot_metrics(metrics, y_test, y_pred, labels)

    
    if classifier == 'Random Forest':
        st.sidebar.subheader('Model Hyper Parameters')
        n_estimators = st.sidebar.number_input('The number of trees in the forest', 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input('The maximun depth of the tree', 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio('Bootstrap samples when building trees', ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect('Select Metrics to plot', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Random Forest Results')
            model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth, bootstrap=np.bool_(bootstrap), n_jobs=-1)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write(f'Accuracy: {accuracy.round(2)}')
            st.write(f'Precision: {precision_score(y_test, y_pred, labels=labels).round(2)}')
            st.write(f'Recall: {recall_score(y_test, y_pred, labels=labels).round(2)}')
            plot_metrics(metrics, y_test, y_pred, labels)



if __name__ == "__main__":
    main()