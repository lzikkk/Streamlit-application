# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:57:06 2025

@author: Administrator
"""

# clustering_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import os
import re

# ---------- 公共函数 ----------
@st.cache_data
def load_dataset(dataset_name, dataset_path):
    """加载指定数据集"""
    try:
        # 构建完整文件路径
        file_path = Path(dataset_path) / f"{dataset_name}.csv"
        print(f"尝试加载文件: {file_path}")
        
        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        df = pd.read_csv(file_path)
        return df.iloc[:, :-1], df.iloc[:, -1]
    except Exception as e:
        st.error(f"加载数据集时出错: {str(e)}")
        return None, None

def run_clustering(X, algo, params):
    """执行聚类算法"""
    if X is None:
        return None
    
    try:
        if algo == "KMeans":
            model = KMeans(n_clusters=params["k"])
        elif algo == "DBSCAN":
            model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        else:  # HAC
            model = AgglomerativeClustering(n_clusters=params["k"])
        labels = model.fit_predict(X)
        return labels
    except Exception as e:
        st.error(f"聚类执行出错: {str(e)}")
        return None

def evaluate(X, y_true, y_pred):
    """评估聚类结果"""
    if X is None or y_true is None or y_pred is None:
        return None, None
    
    try:
        ari = adjusted_rand_score(y_true, y_pred)
        sil = silhouette_score(X, y_pred) if len(set(y_pred)) > 1 else float("nan")
        return ari, sil
    except Exception as e:
        st.error(f"评估结果出错: {str(e)}")
        return None, None

def save_labels(dataset, algo, labels):
    """保存聚类标签"""
    if labels is None:
        return
    
    try:
        Path("results").mkdir(exist_ok=True)
        out = Path("results") / f"{dataset}_{algo}.csv"
        pd.DataFrame(labels, columns=["label"]).to_csv(out, index=False)
        return True
    except Exception as e:
        st.error(f"保存结果出错: {str(e)}")
        return False

def get_subdirectories(path):
    """获取指定路径下的子目录列表"""
    try:
        if os.path.exists(path):
            return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        return []
    except Exception:
        return []

def find_matching_directories(root_path, current_input):
    """查找匹配的目录名作为自动补全建议"""
    if not current_input:
        return []
    
    # 分割路径为基础路径和最后一部分
    parts = current_input.rstrip(os.sep).split(os.sep)
    if len(parts) <= 1:
        base_path = root_path
        search_term = current_input
    else:
        base_path = os.sep.join(parts[:-1])
        search_term = parts[-1]
    
    # 如果基础路径不存在，尝试逐级向上查找
    while not os.path.exists(base_path) and base_path != "":
        base_path = os.path.dirname(base_path)
    
    # 获取匹配的子目录
    subdirs = get_subdirectories(base_path)
    matches = [d for d in subdirs if d.lower().startswith(search_term.lower())]
    
    # 构建完整路径建议
    full_matches = []
    for match in matches:
        if base_path == "":
            full_matches.append(match)
        else:
            full_matches.append(os.path.join(base_path, match))
    
    return full_matches

# ---------- Streamlit 页面 ----------
st.title("🧩 聚类实验平台 (Streamlit One-File)")

# 侧边栏：数据集路径设置
with st.sidebar:
    st.header("⚙️ 系统设置")
    
    # 默认数据集根目录
    DEFAULT_ROOT = "D:\\lzk\\@PG\\03.@G2S2\\论文修改\\0620\\对比算法\\Dataset"
    DEFAULT_UCI_PATH = os.path.join(DEFAULT_ROOT, "UCI")
    
    # 自动补全功能实现
    st.markdown("### 数据集根目录")
    
    # 初始化session_state
    if 'dataset_path' not in st.session_state:
        st.session_state.dataset_path = DEFAULT_UCI_PATH
    
    # 使用session_state的值作为输入框的初始值
    root_input = st.text_input(
        "", 
        value=st.session_state.dataset_path,
        key="dataset_path_input",
        help="输入数据集根目录路径，支持自动补全"
    )
    
    # 当输入框值改变时，更新session_state
    if root_input != st.session_state.dataset_path:
        st.session_state.dataset_path = root_input
    
    # 显示自动补全建议
    if root_input:
        suggestions = find_matching_directories(DEFAULT_ROOT, root_input)
        if suggestions:
            st.markdown("**自动补全建议:**")
            for i, suggestion in enumerate(suggestions[:5]):  # 只显示前5个建议
                # 点击按钮时，更新session_state并触发重新渲染
                if st.button(f"选择: {suggestion}", key=f"suggestion_{i}"):
                    st.session_state.dataset_path = suggestion
    
    st.header("📊 聚类设置")
    dataset = st.selectbox(
        "选择数据集", 
        ["iris", "glass", "Lsun"],
        help="请选择要使用的数据集(需确保CSV文件存在)"
    )
    algo    = st.selectbox(
        "选择算法", 
        ["KMeans", "DBSCAN", "HAC"],
        help="请选择聚类算法"
    )
    
    # 算法参数设置
    if algo == "KMeans" or algo == "HAC":
        k = st.slider("簇数 k", 2, 10, 3)
        params = {"k": k}
    else:
        eps = st.slider("eps", 0.1, 5.0, 0.5, step=0.1)
        mns = st.slider("min_samples", 1, 20, 5)
        params = {"eps": eps, "min_samples": mns}

    run_btn = st.button("🚀 运行聚类")

# 主面板：执行与展示
if run_btn:
    # 加载数据集（使用session_state中的路径）
    X, y = load_dataset(dataset, st.session_state.dataset_path)
    
    if X is not None and y is not None:
        # 执行聚类
        pred = run_clustering(X, algo, params)
        
        if pred is not None:
            # 评估结果
            ari, sil = evaluate(X, y, pred)
            
            if ari is not None and sil is not None:
                st.subheader("📊 评估指标")
                st.write(f"- **ARI**: {ari:.4f}  \n- **Silhouette**: {sil:.4f}")
                
                # 二维可视化（仅展示前两维）
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=pred, cmap="tab10", s=50)
                ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
                ax.set_title(f"{dataset} | {algo} 聚类结果")
                st.pyplot(fig)
                
                # 保存结果
                save_status = save_labels(dataset, algo, pred)
                if save_status:
                    st.success("结果已保存到 /results 目录")
    else:
        st.warning("请检查数据集路径是否正确，或数据集文件是否存在")