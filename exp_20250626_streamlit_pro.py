# -*- coding: utf-8 -*-
"""

"""
import streamlit as st
import pandas as pd, json, datetime as dt
from pathlib import Path
from sklearn.cluster import *
from sklearn.metrics import adjusted_rand_score
import plotly.express as px
import os
import numpy as np

# 初始化目录
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
LOG = RESULTS_DIR / "log.json"

# ---- 工具函数 ----
def get_subdirectories(path):
    """获取指定路径下的子目录列表"""
    try:
        if os.path.exists(path):
            return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        return []
    except:
        return []

def find_matching_paths(root_path, current_input, max_suggestions=5):
    """查找匹配的路径作为自动补全建议"""
    if not current_input:
        return []
    
    parts = current_input.rstrip(os.sep).split(os.sep)
    if len(parts) <= 1:
        base_path = root_path
        search_term = current_input
    else:
        base_path = os.sep.join(parts[:-1])
        search_term = parts[-1]
    
    while not os.path.exists(base_path) and base_path != "":
        base_path = os.path.dirname(base_path)
    
    subdirs = get_subdirectories(base_path)
    matches = [d for d in subdirs if d.lower().startswith(search_term.lower())]
    
    full_matches = []
    for match in matches:
        if base_path == "":
            full_matches.append(match)
        else:
            full_matches.append(os.path.join(base_path, match))
    
    return full_matches[:max_suggestions]

@st.cache_data
def load_data(dataset_name, dataset_path):
    """从指定路径加载数据集"""
    try:
        file_path = Path(dataset_path) / f"{dataset_name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        df = pd.read_csv(file_path)
        return df.iloc[:, :-1], df.iloc[:, -1]
    except Exception as e:
        st.error(f"加载数据出错: {str(e)}")
        return None, None

def ensure_utf8(s):
    """强制将数据转换为UTF-8编码的字符串"""
    if isinstance(s, bytes):
        # 尝试多种编码解码
        for encoding in ['utf-8', 'iso-8859-1', 'gbk', 'utf-16']:
            try:
                return s.decode(encoding, errors='replace')
            except:
                continue
        return s.decode('utf-8', errors='replace')
    elif isinstance(s, (np.ndarray, list, tuple)):
        return type(s)(ensure_utf8(item) for item in s)
    elif isinstance(s, dict):
        return {ensure_utf8(k): ensure_utf8(v) for k, v in s.items()}
    else:
        return str(s)

def clean_log_data(data):
    """清洗日志数据，确保所有内容可转为UTF-8"""
    cleaned = {}
    for key, value in data.items():
        try:
            # 特别处理numpy类型
            if isinstance(value, np.generic):
                cleaned[key] = ensure_utf8(value.item())
            else:
                cleaned[key] = ensure_utf8(value)
        except Exception as e:
            cleaned[key] = f"[编码错误: {str(e)}]"
    return cleaned

def log_run(info: dict):
    """记录运行日志，强制转换为UTF-8编码"""
    try:
        # 清洗数据，确保所有内容可编码为UTF-8
        cleaned_info = clean_log_data(info)
        cleaned_info["time"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 读取现有日志（使用宽容的编码处理）
        log = []
        if LOG.exists():
            with open(LOG, 'r', encoding='utf-8', errors='replace') as f:
                try:
                    log = json.load(f)
                except json.JSONDecodeError:
                    st.warning("日志文件格式错误，将创建新日志")
        
        # 添加新记录
        log.append(cleaned_info)
        
        # 写入日志（使用UTF-8编码，确保ASCII=False）
        with open(LOG, 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"记录日志出错: {str(e)}")
        # 应急：将日志直接保存为文本
        with open(LOG.with_suffix('.txt'), 'a', encoding='utf-8') as f:
            f.write(f"[{dt.datetime.now()}] {str(cleaned_info)}\n")

def read_logs():
    """读取日志文件，自动处理编码问题"""
    if not LOG.exists():
        return pd.DataFrame()
    
    # 尝试读取日志，使用错误替换模式
    with open(LOG, 'r', encoding='utf-8', errors='replace') as f:
        try:
            log_data = json.load(f)
            return pd.DataFrame(log_data)
        except json.JSONDecodeError:
            st.error("日志文件格式损坏，已尝试修复")
            # 尝试修复损坏的JSON
            fixed_data = []
            for line in f:
                try:
                    fixed_data.append(json.loads(line))
                except:
                    continue
            return pd.DataFrame(fixed_data) if fixed_data else pd.DataFrame()

# ---- Streamlit 界面 ----
st.set_page_config(page_title="Clustering Dashboard", layout="wide")
tab_main, tab_history = st.tabs(["🚀 运行实验", "📜 历史记录"])

# 初始化session_state
if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = "D:\\lzk\\@PG\\03.@G2S2\\论文修改\\0620\\对比算法\\Dataset\\UCI"

with tab_main:
    st.header("新实验")
    
    with st.sidebar:
        st.subheader("数据集设置")
        st.markdown("### 数据集根目录")
        
        root_input = st.text_input(
            "",
            value=st.session_state.dataset_path,
            key="dataset_path_input",
            help="输入数据集根目录，支持自动补全"
        )
        
        if root_input:
            suggestions = find_matching_paths(
                os.path.dirname(st.session_state.dataset_path), 
                root_input
            )
            if suggestions:
                st.markdown("**自动补全建议:**")
                for i, suggestion in enumerate(suggestions):
                    if st.button(f"选择: {suggestion}", key=f"path_suggestion_{i}"):
                        st.session_state.dataset_path = suggestion
    
    col1, col2 = st.columns(2)
    try:
        dataset_files = [f.stem for f in Path(st.session_state.dataset_path).glob("*.csv")]
        dataset = col1.selectbox("数据集", dataset_files if dataset_files else ["iris", "glass", "Lsun"])
    except:
        dataset = col1.selectbox("数据集", ["iris", "glass", "Lsun"])
    
    algo    = col1.selectbox("算法", ["KMeans", "DBSCAN"])
    
    if algo == "KMeans":
        k = col2.slider("簇数", 2, 10, 3)
        params = {"n_clusters": k}
    else:
        eps = col2.slider("eps", 0.1, 5.0, 0.5)
        ms  = col2.slider("min_samples", 1, 20, 5)
        params = {"eps": eps, "min_samples": ms}

    if st.button("开始"):
        X, y = load_data(dataset, st.session_state.dataset_path)
        if X is None:
            st.error("数据加载失败，请检查路径和文件")
            st.stop()
        
        if algo == "KMeans":
            labels = KMeans(**params).fit_predict(X)
        else:
            labels = DBSCAN(**params).fit_predict(X)
        
        ari = adjusted_rand_score(y, labels)
        run_time = dt.datetime.now().strftime("%Y-%m-%d %H%M%S")
        
        fig = px.scatter(
            x=X.iloc[:,0], y=X.iloc[:,1], color=labels.astype(str),
            title=f"{dataset}-{algo}", labels={"x":"特征1", "y":"特征2"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("ARI (调整兰德指数)", f"{ari:.4f}")
        
        fname = RESULTS_DIR / f"{dataset}_{algo}_{run_time}.csv"
        pd.DataFrame(labels, columns=["label"]).to_csv(fname, index=False)
        st.download_button(
            "下载聚类标签", 
            data=fname.read_bytes(), 
            file_name=fname.name
        )
        
        log_run({
            "dataset": dataset,
            "algo": algo,
            "params": params,
            "ari": ari,
            "file": fname.name,
            "dataset_path": st.session_state.dataset_path
        })
        st.success("实验完成并记录到历史")

with tab_history:
    st.header("历史记录")
    history = read_logs()
    
    if not history.empty:
        st.dataframe(history)
        st.subheader("按数据集筛选")
        selected_dataset = st.selectbox(
            "选择数据集", 
            ["全部"] + sorted(history["dataset"].unique())
        )
        if selected_dataset != "全部":
            st.dataframe(history[history["dataset"] == selected_dataset])
    else:
        st.info("暂无实验记录，前往『运行实验』选项卡开始第一个实验")