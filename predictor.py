import streamlit as st
import autogluon.core as ag
from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np

# 加载训练好的模型
predictor = TabularPredictor.load("AutogluonModels/ag-20250316_222810")
predictor.set_model_best('CatBoost')

# 读取feature_names.csv
feature_names = pd.read_csv('feature_names.csv')['feature_name'].tolist()


# Streamlit 页面设置
st.title("AutoGluon Multiclass Prediction")
st.write("This app allows you to predict the class of an input sample using an AutoGluon multiclass model.")

# 设置特征的默认值
default_values = {
    feature_names[0]: 0.064686,
    feature_names[1]: 0.047185,
    feature_names[2]: 0.264631,
    feature_names[3]: 0.530666,
    feature_names[4]: 2.322381,
    feature_names[5]: 0.177883,
    feature_names[6]: 4.103680,
    feature_names[7]: 3.158713,
    feature_names[8]: 0.527210,
    feature_names[9]: 0.582900
}

# 创建输入控件，用户可以修改默认值
st.sidebar.header("Input Features")
features = {}
for feature, default_value in default_values.items():
    features[feature] = st.sidebar.number_input(
        feature, min_value=-10.0, max_value=10.0, value=default_value, step=0.1
    )

# 将用户输入转换为 DataFrame 格式
input_data = pd.DataFrame([features])

# 显示用户输入
st.write("User input features:")
st.dataframe(input_data, hide_index=True)

data = pd.DataFrame({'Class':['Description'],'1':['Very Poor'],'2':['Poor'],'3':['Normal'],'4':['Good'],'5':['Very Good']})

st.write("Classification meaning:")
st.dataframe(data,hide_index=True)


# 预测按钮
if st.button("Predict"):
    # 获取预测结果
    prediction = predictor.predict(input_data)
    
    
    # 显示预测的类别概率（如果有的话）
    prediction_proba = predictor.predict_proba(input_data)

    st.write(f"Predicted class: {prediction[0]}")
    
    st.write("Class probabilities:")
    st.dataframe(prediction_proba, hide_index=True)

