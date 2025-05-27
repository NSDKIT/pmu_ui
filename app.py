import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import glob
import os
from datetime import datetime
import time

def load_data_from_files(freq_file, phase_file):
    """アップロードされたCSVファイルを読み込む"""
    try:
        # データ読み込み
        data_f = pd.read_csv(freq_file)
        data_p = pd.read_csv(phase_file)
        
        # 時刻データの処理
        time_f = pd.to_datetime(data_f.iloc[:, 0], errors='coerce')
        time_p = pd.to_datetime(data_p.iloc[:, 0], errors='coerce')
        
        # データの長さを合わせる
        N = min(len(data_p), len(data_f))
        
        # 必要なデータを抽出
        phase_target = data_p.iloc[:N, 1].values  # 2列目（対象カラム）
        phase_ref = data_p.iloc[:N, 2].values     # 3列目（基準カラム）
        freq = data_f.iloc[:N, 1].values          # 2列目（対象カラム）
        t = time_p[:N]
        
        return {
            'phase_target': phase_target,
            'phase_ref': phase_ref,
            'freq': freq,
            'time': t,
            'N': N
        }
        
    except Exception as e:
        st.error(f'ファイル読み込みエラー: {str(e)}')
        return None

def load_data():
    """ローカルディレクトリからCSVファイルを検索して読み込む（バックアップ機能）"""
    # f_*.csvファイルを検索
    freq_files = glob.glob('f_*.csv')
    phase_files = glob.glob('p_*.csv')
    
    if not freq_files or not phase_files:
        return None, (None, None)
    
    # 最初のファイルを使用
    freq_file = freq_files[0]
    phase_file = phase_files[0]
    
    try:
        # データ読み込み
        data_f = pd.read_csv(freq_file)
        data_p = pd.read_csv(phase_file)
        
        # 時刻データの処理
        time_f = pd.to_datetime(data_f.iloc[:, 0], errors='coerce')
        time_p = pd.to_datetime(data_p.iloc[:, 0], errors='coerce')
        
        # データの長さを合わせる
        N = min(len(data_p), len(data_f))
        
        # 必要なデータを抽出
        phase_target = data_p.iloc[:N, 1].values  # 2列目（対象カラム）
        phase_ref = data_p.iloc[:N, 2].values     # 3列目（基準カラム）
        freq = data_f.iloc[:N, 1].values          # 2列目（対象カラム）
        t = time_p[:N]
        
        return {
            'phase_target': phase_target,
            'phase_ref': phase_ref,
            'freq': freq,
            'time': t,
            'N': N
        }, (freq_file, phase_file)
        
    except Exception as e:
        st.error(f'ファイル読み込みエラー: {str(e)}')
        return None, (None, None)

def process_data(data, theta0):
    """データを処理する"""
    # 連続位相角の計算（unwrap相当）
    phase_target_cont = np.degrees(np.unwrap(np.radians(data['phase_target'])))
    phase_ref_cont = np.degrees(np.unwrap(np.radians(data['phase_ref'])))
    
    # 位相差の計算
    phase_diff = phase_target_cont - phase_ref_cont
    
    # 基準角度でのwrap処理
    x = ((phase_diff + theta0 + 180) % 360) - 180
    theta_plot = np.radians(x)
    y = 2 * np.pi * data['freq']
    
    return x, theta_plot, y, phase_diff

def create_polar_plot(theta_plot, curr_idx, title):
    """極座標プロットを作成"""
    fig = go.Figure()
    
    # 現在の点
    fig.add_trace(go.Scatterpolar(
        r=[1],
        theta=[np.degrees(theta_plot[curr_idx])],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='現在位置'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1.1], visible=True),
            angularaxis=dict(
                direction='clockwise',
                rotation=90
            )
        ),
        title=title,
        height=300
    )
    
    return fig

def create_xy_scatter(x, y, curr_idx, title):
    """XY散布図を作成"""
    fig = go.Figure()
    
    # 履歴データ
    fig.add_trace(go.Scatter(
        x=x[:curr_idx+1],
        y=y[:curr_idx+1],
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.6),
        name='履歴'
    ))
    
    # 現在の点
    fig.add_trace(go.Scatter(
        x=[x[curr_idx]],
        y=[y[curr_idx]],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='現在位置'
    ))
    
    # 円（標準偏差）
    if curr_idx > 0:
        center_x = np.mean(x[:curr_idx+1])
        center_y = np.mean(y[:curr_idx+1])
        radius = np.std(x[:curr_idx+1])
        
        theta_circle = np.linspace(0, 2*np.pi, 100)
        circle_x = center_x + radius * np.cos(theta_circle)
        circle_y = center_y + radius * np.sin(theta_circle)
        
        fig.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(dash='dash', color='black'),
            name='標準偏差円'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='相対位相 [deg]（±180°wrap）',
        yaxis_title='周波数 [rad/s]',
        xaxis=dict(range=[-180, 180]),
        height=300
    )
    
    return fig

def create_time_series(t, x, freq, curr_idx):
    """時系列プロットを作成"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 位相データ
    fig.add_trace(
        go.Scatter(x=t, y=x, mode='lines', name='相対位相角', 
                  line=dict(color='blue')),
        secondary_y=False,
    )
    
    # 現在位置（位相）
    fig.add_trace(
        go.Scatter(x=[t.iloc[curr_idx]], y=[x[curr_idx]], 
                  mode='markers', marker=dict(size=10, color='red'),
                  name='現在位置（位相）'),
        secondary_y=False,
    )
    
    # 周波数データ
    fig.add_trace(
        go.Scatter(x=t, y=freq, mode='lines', name='周波数',
                  line=dict(color='green')),
        secondary_y=True,
    )
    
    # 現在位置（周波数）
    fig.add_trace(
        go.Scatter(x=[t.iloc[curr_idx]], y=[freq[curr_idx]], 
                  mode='markers', marker=dict(size=10, color='orange', symbol='square'),
                  name='現在位置（周波数）'),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="時刻")
    fig.update_yaxes(title_text="相対位相角 [deg]", secondary_y=False, range=[-180, 180])
    fig.update_yaxes(title_text="周波数 [Hz]", secondary_y=True)
    
    fig.update_layout(
        title="時系列上の相対位相角と周波数",
        height=400
    )
    
    return fig

def create_power_angle_curve(x, curr_idx, E=1.0, V=1.0, X=0.3):
    """Power Angle Curveを作成"""
    fig = go.Figure()
    
    # 理論曲線
    delta_curve = np.linspace(-180, 180, 1000)
    P_curve = (E * V / X) * np.sin(np.radians(delta_curve))
    
    fig.add_trace(go.Scatter(
        x=delta_curve,
        y=P_curve,
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='理論曲線'
    ))
    
    # 実データ
    delta = x
    P = (E * V / X) * np.sin(np.radians(delta))
    
    fig.add_trace(go.Scatter(
        x=delta[:curr_idx+1],
        y=P[:curr_idx+1],
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.6),
        name='実測値'
    ))
    
    # 現在の点
    fig.add_trace(go.Scatter(
        x=[delta[curr_idx]],
        y=[P[curr_idx]],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='現在位置'
    ))
    
    fig.update_layout(
        title='Power Angle Curve: P=(EV/X)sin(δ)',
        xaxis_title='相対位相角 δ [deg]（±180°wrap）',
        yaxis_title='電力 P [pu]',
        xaxis=dict(range=[-180, 180]),
        height=300
    )
    
    return fig

def main():
    st.set_page_config(page_title="PMU相対位相アニメUI", layout="wide")
    
    st.title("PMU相対位相アニメーションUI")
    st.markdown("---")
    
    # ファイルアップロード
    st.markdown("### ファイルアップロード")
    col1, col2 = st.columns(2)
    
    with col1:
        freq_file = st.file_uploader(
            "周波数データファイル (f_*.csv)",
            type=['csv'],
            help="1列目: 時刻, 2列目: 周波数データ"
        )
    
    with col2:
        phase_file = st.file_uploader(
            "位相角データファイル (p_*.csv)",
            type=['csv'],
            help="1列目: 時刻, 2列目: 対象位相, 3列目: 基準位相"
        )
    
    # データ読み込み
    data = None
    filenames = (None, None)
    
    if freq_file is not None and phase_file is not None:
        # アップロードされたファイルから読み込み
        data = load_data_from_files(freq_file, phase_file)
        if data is not None:
            filenames = (freq_file.name, phase_file.name)
            st.success(f'ファイル読み込み成功: 周波数 = {freq_file.name}, 位相角 = {phase_file.name}')
    else:
        # ローカルファイルから読み込み（バックアップ）
        if freq_file is None and phase_file is None:
            st.info("ファイルをアップロードしてください。または、アプリと同じディレクトリにf_*.csvとp_*.csvファイルを配置してください。")
            data, filenames = load_data()
            if data is not None:
                st.info(f'ローカルファイル使用: 周波数 = {filenames[0]}, 位相角 = {filenames[1]}')
        elif freq_file is None:
            st.warning("周波数データファイル (f_*.csv) をアップロードしてください")
        elif phase_file is None:
            st.warning("位相角データファイル (p_*.csv) をアップロードしてください")
    
    if data is None:
        st.stop()
    
    st.markdown("---")
    
    # サイドバーのコントロール
    st.sidebar.header("制御パネル")
    
    # 基準角度設定
    theta0 = st.sidebar.selectbox(
        "基準角度 [deg]:",
        options=[0, 5, 10, 15, 20, 25, 30, 35, 40],
        index=6  # デフォルト30
    )
    
    # データ処理
    x, theta_plot, y, phase_diff = process_data(data, theta0)
    
    # スライダーとアニメーション制御
    max_idx = data['N'] - 1
    
    # セッション状態の初期化
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    if 'animation_running' not in st.session_state:
        st.session_state.animation_running = False
    
    # コントロールボタン
    col1, col2, col3, col4 = st.sidebar.columns(4)
    
    with col1:
        if st.button("▶️ Start"):
            st.session_state.animation_running = True
    
    with col2:
        if st.button("⏸️ Stop"):
            st.session_state.animation_running = False
    
    with col3:
        if st.button("⏮️ Reset"):
            st.session_state.current_idx = 0
            st.session_state.animation_running = False
    
    with col4:
        if st.button("⏭️ End"):
            st.session_state.current_idx = max_idx
            st.session_state.animation_running = False
    
    # 再生速度設定
    speed = st.sidebar.selectbox(
        "再生速度:",
        options=[1, 2, 5, 10, 20, 50, 100],
        index=3  # デフォルト10
    )
    
    # スライダー
    curr_idx = st.sidebar.slider(
        "時刻インデックス",
        min_value=0,
        max_value=max_idx,
        value=st.session_state.current_idx,
        key="time_slider"
    )
    
    st.session_state.current_idx = curr_idx
    
    # 現在時刻の表示
    current_time = data['time'].iloc[curr_idx]
    st.sidebar.write(f"現在時刻: {current_time}")
    
    # メインプロット領域
    col1, col2 = st.columns(2)
    
    with col1:
        # 基準系での全ての位相wrap
        polar_fig1 = create_polar_plot(theta_plot, curr_idx, 
                                     f"基準系での全位相wrap, 時刻: {current_time}")
        st.plotly_chart(polar_fig1, use_container_width=True)
        
        # XY散布図
        xy_fig = create_xy_scatter(x, y, curr_idx, 
                                 "基準系での相対位相 × freq [rad/s] 散布図")
        st.plotly_chart(xy_fig, use_container_width=True)
    
    with col2:
        # 極座標プロット（周波数）
        polar_fig2 = go.Figure()
        polar_fig2.add_trace(go.Scatterpolar(
            r=data['freq'][:curr_idx+1],
            theta=np.degrees(theta_plot[:curr_idx+1]),
            mode='markers',
            marker=dict(size=8, color='teal', opacity=0.6),
            name='履歴'
        ))
        polar_fig2.add_trace(go.Scatterpolar(
            r=[data['freq'][curr_idx]],
            theta=[np.degrees(theta_plot[curr_idx])],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='現在位置'
        ))
        polar_fig2.update_layout(
            polar=dict(
                radialaxis=dict(range=[min(data['freq']), max(data['freq'])]),
                angularaxis=dict(direction='clockwise', rotation=90)
            ),
            title='極座標: 周波数(半径) × 基準系位相角(角度)',
            height=300
        )
        st.plotly_chart(polar_fig2, use_container_width=True)
        
        # Power Angle Curve
        power_fig = create_power_angle_curve(x, curr_idx)
        st.plotly_chart(power_fig, use_container_width=True)
    
    # 時系列プロット（全幅）
    st.markdown("### 時系列データ")
    time_fig = create_time_series(data['time'], x, data['freq'], curr_idx)
    st.plotly_chart(time_fig, use_container_width=True)
    
    # アニメーション処理
    if st.session_state.animation_running:
        if st.session_state.current_idx < max_idx:
            time.sleep(0.1)  # アニメーション速度調整
            st.session_state.current_idx = min(max_idx, 
                                             st.session_state.current_idx + speed)
            st.rerun()
        else:
            st.session_state.animation_running = False
    
    # データ情報表示
    st.sidebar.markdown("---")
    st.sidebar.markdown("### データ情報")
    st.sidebar.write(f"データ点数: {data['N']}")
    if filenames[0] and filenames[1]:
        st.sidebar.write(f"周波数ファイル: {filenames[0]}")
        st.sidebar.write(f"位相ファイル: {filenames[1]}")
    
    # ファイルフォーマット説明
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ファイルフォーマット")
    st.sidebar.markdown("""
    **周波数ファイル (f_*.csv)**
    - 1列目: 時刻
    - 2列目: 周波数データ
    
    **位相ファイル (p_*.csv)**
    - 1列目: 時刻  
    - 2列目: 対象位相角
    - 3列目: 基準位相角
    """)

if __name__ == "__main__":
    main()
