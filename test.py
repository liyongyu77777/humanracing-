import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE

# 读取数据
df = pd.read_csv("data/Uncleaned_employees_final_dataset (1).csv")

# 数据预览
print(df.head())

# 处理缺失值
df['previous_year_rating'].fillna(df['previous_year_rating'].mean(), inplace=True)
df['no_of_trainings'].fillna(df['no_of_trainings'].median(), inplace=True)
df['age'].fillna(df['age'].median(), inplace=True)
df['length_of_service'].fillna(df['length_of_service'].median(), inplace=True)
df['KPIs_met_more_than_80'].fillna(df['KPIs_met_more_than_80'].mode()[0], inplace=True)
df['awards_won'].fillna(df['awards_won'].mode()[0], inplace=True)
df['avg_training_score'].fillna(df['avg_training_score'].median(), inplace=True)

# 处理异常值
df = df[(df['avg_training_score'] >= 0) & (df['avg_training_score'] <= 100)]
df = df[(df['age'] >= 18) & (df['age'] <= 65)]

# 特征工程：创建新的特征
df['age_service_ratio'] = df['age'] / df['length_of_service']

# 特征和目标变量
X = df.drop(columns=['employee_id', 'previous_year_rating'])
y = df['previous_year_rating']

# 定义预处理步骤
numeric_features = ['no_of_trainings', 'age', 'length_of_service', 'avg_training_score', 'age_service_ratio']
categorical_features = ['department', 'region', 'education', 'gender', 'recruitment_channel']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 使用递归特征消除（RFE）进行特征选择
model = RandomForestRegressor(random_state=42)
rfe = RFE(model, n_features_to_select=10)

# 创建Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', rfe),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
pipeline.fit(X_train, y_train)

# 预测并计算均方误差
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Random Forest Mean Squared Error: {mse}')

# 特征重要性
# 获取预处理后的特征名称
preprocessor.fit(X_train)
preprocessed_features = preprocessor.get_feature_names_out()

# 获取RFE选择的特征
selected_features = preprocessed_features[rfe.support_]

# 获取特征重要性
feature_importances = pipeline.named_steps['regressor'].feature_importances_
importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

# 使用不同的模型进行比较
# 线性回归模型
linear_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', rfe),
    ('regressor', LinearRegression())
])
linear_pipeline.fit(X_train, y_train)
y_pred_linear = linear_pipeline.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f'Linear Regression Mean Squared Error: {mse_linear}')

# 梯度提升回归模型
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', rfe),
    ('regressor', GradientBoostingRegressor(random_state=42))
])
gb_pipeline.fit(X_train, y_train)
y_pred_gb = gb_pipeline.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
print(f'Gradient Boosting Mean Squared Error: {mse_gb}')

# 使用交叉验证评估模型
scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
mse_cv = -scores.mean()
print(f'Cross-Validated Mean Squared Error: {mse_cv}')
