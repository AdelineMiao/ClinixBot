# visualization_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from translations import translations

class VisualizationDashboard:
    def __init__(self, data, lang="zh"):
        self.data = data
        self.lang = lang
        self.t = translations
        
    def _preprocess_data(self):
        """数据预处理"""
        # 转换日期列
        date_columns = ['Admit Date', 'Discharge Date']
        for col in date_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col])
        
        # 计算住院天数
        if 'Admit Date' in self.data.columns and 'Discharge Date' in self.data.columns:
            self.data['Stay Duration'] = (self.data['Discharge Date'] - self.data['Admit Date']).dt.days
        
        # 提取年月信息
        if 'Admit Date' in self.data.columns:
            self.data['Admit Year'] = self.data['Admit Date'].dt.year
            self.data['Admit Month'] = self.data['Admit Date'].dt.month
            self.data['Admit YearMonth'] = self.data['Admit Date'].dt.strftime('%Y-%m')
        
        return self.data
    
    def _plot_medical_conditions_distribution(self):
        """绘制医疗条件分布图"""
        t = self.t.dashboard[self.lang]
        
        if 'Medical Condition' not in self.data.columns:
            # 根据语言选择错误消息
            error_msg = "数据中缺少'Medical Condition'列" if self.lang == "zh" else "Missing 'Medical Condition' column in data"
            return st.error(error_msg)
        
        # 统计Top 10医疗条件
        condition_counts = self.data['Medical Condition'].value_counts().nlargest(10)
        
        # 使用Plotly创建条形图
        fig = px.bar(
            x=condition_counts.values,
            y=condition_counts.index,
            orientation='h',
            title=t["condition_distribution"],
            labels={'x': t["patient_count"], 'y': t["medical_condition"]},
            color=condition_counts.values,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title=t["patient_count"],
            yaxis_title=t["medical_condition"],
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_billing_analysis(self):
        """绘制账单分析图表"""
        t = self.t.dashboard[self.lang]
        
        if 'Bill Amount' not in self.data.columns or 'Medical Condition' not in self.data.columns:
            # 根据语言选择错误消息
            error_msg = "数据中缺少必要的列" if self.lang == "zh" else "Missing required columns in data"
            return st.error(error_msg)
        
        # 按医疗条件统计平均账单金额
        avg_bill_by_condition = self.data.groupby('Medical Condition')['Bill Amount'].mean().nlargest(10)
        
        # 创建条形图
        fig = px.bar(
            x=avg_bill_by_condition.index,
            y=avg_bill_by_condition.values,
            title=t["avg_bill_by_condition"],
            labels={'x': t["medical_condition"], 'y': t["avg_bill_amount"]},
            color=avg_bill_by_condition.values,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title=t["medical_condition"],
            yaxis_title=t["avg_bill_amount"],
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render(self):
        """渲染可视化仪表盘"""
        t = self.t.dashboard[self.lang]
        
        st.header(t["header"])
        
        # 预处理数据
        self._preprocess_data()
        
        # 显示关键指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t["total_patients"], f"{len(self.data):,}")
        with col2:
            avg_bill = self.data['Bill Amount'].mean()
            st.metric(t["avg_bill"], f"${avg_bill:.2f}")
        with col3:
            if 'Stay Duration' in self.data.columns:
                avg_stay = self.data['Stay Duration'].mean()
                days_text = "天" if self.lang == "zh" else "days"
                st.metric(t["avg_stay"], f"{avg_stay:.1f} {days_text}")
        with col4:
            if 'Medical Condition' in self.data.columns:
                unique_conditions = self.data['Medical Condition'].nunique()
                st.metric(t["disease_count"], f"{unique_conditions}")
        
        # 创建选项卡
        tab1, tab2 = st.tabs([t["disease_tab"], t["billing_tab"]])
        
        with tab1:
            self._plot_medical_conditions_distribution()
        
        with tab2:
            self._plot_billing_analysis()