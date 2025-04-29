# data processor.py
import pandas as pd
import numpy as np
from datetime import datetime
import re

class DataProcessor:
    def __init__(self):
        """初始化数据处理器"""
        pass
    
    def preprocess_data(self, data):
        """预处理医疗数据"""
        # 创建数据副本，避免修改原始数据
        df = data.copy()
        
        # 处理缺失值
        self._handle_missing_values(df)
        
        # 转换日期列
        self._convert_date_columns(df)
        
        # 清洗和标准化文本列
        self._clean_text_columns(df)
        
        # 计算派生特征
        self._compute_derived_features(df)
        
        return df
    
    def _handle_missing_values(self, df):
        """处理缺失值"""
        # 用中位数填充数值型缺失值
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # 用众数填充分类型缺失值
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                most_frequent = df[col].mode()[0]
                df[col].fillna(most_frequent, inplace=True)
    
    def _convert_date_columns(self, df):
        """转换日期列"""
        # 识别和转换日期列
        date_columns = ['Date of Birth', 'Admit Date', 'Discharge Date']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 计算住院天数
        if 'Admit Date' in df.columns and 'Discharge Date' in df.columns:
            df['Stay Duration'] = (df['Discharge Date'] - df['Admit Date']).dt.days
    
    def _clean_text_columns(self, df):
        """清洗和标准化文本列"""
        # 处理文本列
        text_columns = ['Medical Condition', 'Treatments', 'Doctor\'s Notes']
        
        for col in text_columns:
            if col in df.columns:
                # 转换为小写
                df[col] = df[col].str.lower()
                
                # 移除多余空格
                df[col] = df[col].str.strip()
    
    def _compute_derived_features(self, df):
        """计算派生特征"""
        # 计算年龄（如果有出生日期）
        if 'Date of Birth' in df.columns:
            current_date = datetime.now()
            df['Age'] = (current_date - df['Date of Birth']).dt.days // 365
            
            # 创建年龄组
            bins = [0, 18, 30, 45, 60, 75, 100]
            labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
            df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
        
        # 从治疗中提取主要治疗方法
        if 'Treatments' in df.columns:
            df['Primary Treatment'] = df['Treatments'].str.split(',').str[0]