import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import time
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_sample_data():
    """サンプルデータを生成（デモ用）"""
    np.random.seed(42)
    n_points = 500
    
    # 時系列データ生成
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_points)]
    
    # 周波数データ（50Hz基準で微小変動）
    base_freq = 50.0
    freq_noise = np.random.normal(0, 0.05, n_points)
    freq_trend = 0.1 * np.sin(np.linspace(0, 4*np.pi, n_points))
    freq1 = base_freq + freq_trend + freq_noise
    freq2 = base_freq + freq_trend * 0.8 + np.random.normal(0, 0.03, n_points)
    
    # 位相データ（度単位、位相差を持つ振動）
    base_phase = 30 * np.sin(np.linspace(0, 6*np.pi, n_points))
    phase_noise = np.random.normal(0, 2, n_points)
    phase1 = base_phase + phase_noise
    phase2 = base_phase + 15 + np.random.normal(0, 1.5, n_points)
    
    # DataFrameを作成
    freq_df = pd.DataFrame({
        'timestamp': timestamps,
        'frequency1': freq1,
        'frequency2': freq2
    })
    
    phase_df = pd.DataFrame({
        'timestamp': timestamps,
        'phase1': phase1,
        'phase2': phase2
    })
    
    return freq_df, phase_df

def auto_find_csv(prefix):
    """CSVファイルを自動検索"""
    files = sorted(glob.glob(f'{prefix}_*.csv'))
    if not files:
        return None
    return files[0]

@st.cache_data
def load_data(freq_file=None, phase_file=None, use_sample=False):
    """データ読み込み - tkinterコードと同じロジック"""
    if use_sample:
        # サンプルデータを使用
        freq_df, phase_df = create_sample_data()
    else:
        if freq_file is None:
            freq_file = auto_find_csv('f')
        if phase_file is None:
            phase_file = auto_find_csv('p')
        
        if freq_file is None or phase_file is None:
            raise FileNotFoundError("CSVファイルが見つかりません")
            
        freq_df = pd.read_csv(freq_file, encoding='utf-8')
        phase_df = pd.read_csv(phase_file, encoding='utf-8')
    
    N = min(len(freq_df), len(phase_df))
    freq_df = freq_df.interpolate(method='linear', limit_direction='both')
    phase_df = phase_df.interpolate(method='linear', limit_direction='both')
    
    return freq_df.iloc[:N], phase_df.iloc[:N]

def process_data(freq_df, phase_df, sampling=10):
    """データ処理 - tkinterコードと完全同期"""
    N = min(len(freq_df), len(phase_df))
    idx_target = 1  # Pythonは0-index
    idx_ref = 2
    idx_freq = 1
    
    # データ処理（tkinterと同じ）
    t = pd.to_datetime(phase_df.iloc[:,0])
    freq = freq_df.iloc[:, idx_freq].values
    phase_target = phase_df.iloc[:, idx_target].values
    phase_ref = phase_df.iloc[:, idx_ref].values
    
    # 位相の連続化（tkinterと同じ処理）
    phase_target_cont = np.unwrap(np.deg2rad(phase_target)) * 180/np.pi
    phase_ref_cont = np.unwrap(np.deg2rad(phase_ref)) * 180/np.pi
    phase_diff = phase_target_cont - phase_ref_cont
    
    # サンプリング
    sample_idx = np.arange(0, N, sampling)
    t_s = t.iloc[sample_idx]
    freq_s = freq[sample_idx]
    phase_diff_s = phase_diff[sample_idx]
    
    return {
        'N': N,
        't': t,
        'freq': freq,
        'phase_diff': phase_diff,
        't_s': t_s,
        'freq_s': freq_s,
        'phase_diff_s': phase_diff_s
    }

class PMUVisualization:
    """tkinterのPMUUIAppクラスを忠実に移植"""
    
    def __init__(self, data):
        self.data = data
        # tkinterと同じパラメータ
        self.E = 1.0
        self.V = 1.0
        self.X = 0.3
    
    def get_x_theta_r_y(self, theta0):
        """tkinterのget_x_theta_r_yと完全同一"""
        x = np.mod(self.data['phase_diff'] + theta0 + 180, 360) - 180
        theta = np.deg2rad(x)
        y = 2*np.pi*self.data['freq']
        r = np.ones(self.data['N'])
        return x, theta, r, y
    
    def get_xs(self, theta0):
        """tkinterのget_xsと完全同一"""
        xs = np.mod(self.data['phase_diff_s'] + theta0 + 180, 360) - 180
        return xs
    
    def get_power_curve(self, theta0):
        """tkinterのget_power_curveと完全同一"""
        delta = np.mod(self.data['phase_diff'] + theta0 + 180, 360) - 180
        delta_rad = np.deg2rad(delta)
        P = (self.E*self.V/self.X)*np.sin(delta_rad)
        return delta, P
    
    def create_plots(self, curr_idx, theta0):
        """tkinterのinit_plot_objects + update_plotsを統合移植"""
        x, theta, r, y = self.get_x_theta_r_y(theta0)
        xs = self.get_xs(theta0)
        idx = curr_idx
        
        # tkinterと同じ図サイズとレイアウト
        fig = plt.figure(figsize=(12, 6), dpi=100)
        gs = fig.add_gridspec(2, 3)
        
        # --- 左上：円グラフ（tkinter完全移植） ---
        ax_polar1 = fig.add_subplot(gs[0,0], polar=True)
        ax_polar1.set_rlim(0, 1.1)
        ax_polar1.plot([theta[idx]], [r[idx]], 'o', markersize=10, color='tab:blue')
        ax_polar1.set_title(f'θ₀基準 相対位相[deg] 円グラフ\n時刻: {self.data["t"].iloc[idx]}')
        
        # --- 中央上：xy散布図（tkinter完全移植） ---
        ax_xy = fig.add_subplot(gs[0,1])
        ax_xy.plot(xs, self.data['freq_s']*2*np.pi, '.', color='tab:blue', alpha=0.2, markersize=4, label='全区間')
        ax_xy.plot(x[:idx+1], y[:idx+1], '.', color='tab:orange', markersize=8, label='履歴')
        ax_xy.plot([x[idx]], [y[idx]], 'r.', markersize=12, label='現在')
        ax_xy.set_xlabel('θ₀基準相対位相[deg]')
        ax_xy.set_ylabel('周波数[rad/s]')
        ax_xy.set_xlim(-180, 180)
        ax_xy.set_title('θ₀基準: 位相×freq 散布図')
        ax_xy.legend(loc='lower right')
        
        # --- 右上：極座標（tkinter完全移植） ---
        ax_polar2 = fig.add_subplot(gs[0,2], polar=True)
        ax_polar2.set_rlim(np.nanmin(self.data['freq']), np.nanmax(self.data['freq'])*1.05)
        ax_polar2.plot([theta[idx]], [self.data['freq'][idx]], 'o', markersize=10, color='tab:red')
        ax_polar2.set_title('極座標: θ₀基準位相 × 周波数')
        
        # --- 下段左：時系列（2軸プロット、tkinter完全移植） ---
        ax_time = fig.add_subplot(gs[1,:2])
        ax_time2 = ax_time.twinx()
        
        # 位相プロット（tkinterと同じ）
        ax_time.plot(self.data['t_s'], xs, color='tab:blue', alpha=0.2, linestyle='-', label='θ₀基準相対位相[deg] 全体(間引)')
        ax_time.plot(self.data['t'][:idx+1], x[:idx+1], color='tab:blue', linestyle='-', marker='.', markersize=3, label='θ₀基準相対位相[deg] 履歴')
        ax_time.plot([self.data['t'][idx]], [x[idx]], 'o', color='tab:blue')
        ax_time.set_ylabel('θ₀基準相対位相[deg]')
        ax_time.set_ylim(-180, 180)
        ax_time.set_xlabel('時刻')
        ax_time.set_xlim(self.data['t'].iloc[0], self.data['t'].iloc[-1])
        
        # 周波数プロット（tkinterと同じ）
        ax_time2.plot(self.data['t_s'], self.data['freq_s'], color='tab:red', alpha=0.2, label='周波数[Hz] 全体(間引)')
        ax_time2.plot(self.data['t'][:idx+1], self.data['freq'][:idx+1], color='tab:red', linestyle='-', marker='.', markersize=3, label='周波数[Hz] 履歴')
        ax_time2.plot([self.data['t'][idx]], [self.data['freq'][idx]], 's', color='tab:red')
        ax_time2.set_ylabel('周波数[Hz]')
        ax_time2.set_ylim(np.nanmin(self.data['freq'])-0.2, np.nanmax(self.data['freq'])+0.2)
        
        ax_time.set_title(f"時系列（θ₀基準相対位相＋周波数）\nカレント: {self.data['t'].iloc[idx]}")
        
        # 凡例統合（tkinterと同じ）
        lines1, labels1 = ax_time.get_legend_handles_labels()
        lines2, labels2 = ax_time2.get_legend_handles_labels()
        ax_time.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # --- 下段右：Power Angle Curve（tkinter完全移植） ---
        ax_power = fig.add_subplot(gs[1,2])
        delta_curve = np.linspace(-180, 180, 1000)
        P_curve = (self.E*self.V/self.X)*np.sin(np.deg2rad(delta_curve))
        ax_power.plot(delta_curve, P_curve, '--', color='gray', linewidth=1)
        
        delta, P = self.get_power_curve(theta0)
        ax_power.plot(delta[:idx+1], P[:idx+1], '.', color='tab:blue', markersize=8)
        ax_power.plot([delta[idx]], [P[idx]], 'r.', markersize=12)
        ax_power.set_xlabel('θ₀基準 相差角 δ [deg]')
        ax_power.set_ylabel('電力P [pu]')
        ax_power.set_xlim(-180, 180)
        ax_power.set_ylim(P_curve.min()-0.2, P_curve.max()+0.2)
        ax_power.set_title('Power Angle Curve: P=(EV/X)sin(δ)')
        
        # tkinterと同じtight_layout
        fig.tight_layout()
        return fig

def main():
    st.set_page_config(page_title="PMU相対位相アニメUI", layout="wide")
    st.title("PMU相対位相アニメUI")
    st.markdown("*CSVデータを読み込んでPMU解析を実行*")
    
    # セッション状態の初期化
    if 'curr_idx' not in st.session_state:
        st.session_state.curr_idx = 0
    if 'auto_play' not in st.session_state:
        st.session_state.auto_play = False
    
    # メインのファイルアップロード機能
    st.header("📂 CSVファイルの読み込み")
    
    # ファイルアップロードUI
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("周波数データ")
        freq_file = st.file_uploader(
            "周波数CSVファイル (f_*.csv)", 
            type="csv",
            help="周波数データを含むCSVファイルをアップロードしてください"
        )
        if freq_file:
            st.success(f"✅ {freq_file.name} を読み込みました")
            
    with col2:
        st.subheader("位相データ")
        phase_file = st.file_uploader(
            "位相CSVファイル (p_*.csv)", 
            type="csv",
            help="位相データを含むCSVファイルをアップロードしてください"
        )
        if phase_file:
            st.success(f"✅ {phase_file.name} を読み込みました")
    
    # サンプルデータオプション
    st.markdown("---")
    col_sample1, col_sample2 = st.columns([3, 1])
    
    with col_sample1:
        st.info("💡 ファイルがない場合は、サンプルデータで動作を確認できます")
    
    with col_sample2:
        use_sample = st.button("🔬 サンプルデータを使用", type="secondary")
    
    # データ処理とアプリケーション実行
    if freq_file and phase_file:
        try:
            # CSVファイルからデータを読み込み
            freq_df = pd.read_csv(freq_file, encoding='utf-8')
            phase_df = pd.read_csv(phase_file, encoding='utf-8')
            
            # データプレビュー
            with st.expander("📊 データプレビュー", expanded=False):
                col_prev1, col_prev2 = st.columns(2)
                with col_prev1:
                    st.write("**周波数データ (最初の5行)**")
                    st.dataframe(freq_df.head())
                with col_prev2:
                    st.write("**位相データ (最初の5行)**")
                    st.dataframe(phase_df.head())
            
            # データ処理
            if 'data' not in st.session_state or st.session_state.get('last_files') != (freq_file.name, phase_file.name):
                st.session_state.last_files = (freq_file.name, phase_file.name)
                st.session_state.data = process_data(freq_df, phase_df)
                st.session_state.visualization = PMUVisualization(st.session_state.data)
                st.session_state.curr_idx = 0
                st.success("✅ データ処理が完了しました")
            
            # メインアプリケーション実行
            run_main_application()
            
        except Exception as e:
            st.error(f"❌ データ読み込みエラー: {e}")
            st.info("ファイル形式を確認してください。各CSVファイルの1列目は時刻データである必要があります。")
            
    elif use_sample:
        # サンプルデータでアプリケーション実行
        try:
            freq_df, phase_df = create_sample_data()
            
            if 'data' not in st.session_state or st.session_state.get('last_files') != ('sample', 'sample'):
                st.session_state.last_files = ('sample', 'sample')
                st.session_state.data = process_data(freq_df, phase_df)
                st.session_state.visualization = PMUVisualization(st.session_state.data)
                st.session_state.curr_idx = 0
                st.info("🔬 サンプルデータを使用しています")
            
            # メインアプリケーション実行
            run_main_application()
            
        except Exception as e:
            st.error(f"❌ サンプルデータエラー: {e}")
    
    else:
        # ファイル未選択時の説明
        st.markdown("---")
        st.subheader("📋 使用方法")
        
        st.markdown("""
        ### CSVファイルの形式
        
        **周波数ファイル (f_*.csv)**
        ```
        timestamp,frequency1,frequency2
        2024-01-01 00:00:00,50.0,50.1
        2024-01-01 00:00:01,50.05,50.08
        ...
        ```
        
        **位相ファイル (p_*.csv)**
        ```
        timestamp,phase1,phase2
        2024-01-01 00:00:00,0,30
        2024-01-01 00:00:01,5,35
        ...
        ```
        
        ### 要件
        - 両方のファイルの1列目は時刻データ（timestamp）
        - 周波数データは2列目以降に数値データ
        - 位相データは2列目以降に数値データ（度単位）
        - エンコーディング: UTF-8
        """)

def run_main_application():
    """メインアプリケーション部分"""
    data = st.session_state.data
    viz = st.session_state.visualization
    
    st.markdown("---")
    st.header("🎛️ PMU解析制御パネル")
    
    # tkinterと同じ制御パネル
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 2, 1])
    
    with col1:
        if st.button("▶️ Start", key="start_btn", help="自動再生開始"):
            st.session_state.auto_play = True
    
    with col2:
        if st.button("⏹️ Stop", key="stop_btn", help="自動再生停止"):
            st.session_state.auto_play = False
    
    with col3:
        if st.button("🔄 Reset", key="reset_btn", help="最初に戻る"):
            st.session_state.curr_idx = 0
            st.session_state.auto_play = False
    
    with col4:
        # tkinterと同じ速度選択肢
        speed_choices = [1, 2, 5, 10, 20, 50, 100]
        speed = st.selectbox("再生速度", options=speed_choices, index=3, key="speed_select")
    
    with col5:
        curr_idx = st.slider("時間位置", 0, data['N']-1, st.session_state.curr_idx, key="time_slider")
        if curr_idx != st.session_state.curr_idx:
            st.session_state.curr_idx = curr_idx
            st.session_state.auto_play = False  # 手動操作時は自動再生停止
    
    with col6:
        # tkinterと同じθ₀選択肢
        theta0_choices = list(range(0, 45, 5))
        theta0 = st.selectbox("基準θ₀[deg]", options=theta0_choices, index=6, key="theta0_select")
    
    # 状態表示
    status_col1, status_col2 = st.columns([3, 1])
    with status_col1:
        if st.session_state.auto_play:
            st.warning("🔄 自動再生中")
        else:
            st.info("⏸️ 停止中")
    
    with status_col2:
        st.metric("進行度", f"{st.session_state.curr_idx}/{data['N']-1}")
    
    # tkinterと同じプロット生成・表示
    st.subheader("📈 PMU解析結果")
    fig = viz.create_plots(st.session_state.curr_idx, theta0)
    st.pyplot(fig)
    plt.close(fig)
    
    # tkinterと同じアニメーションロジック
    if st.session_state.auto_play:
        if st.session_state.curr_idx < data['N'] - 1:
            st.session_state.curr_idx = min(st.session_state.curr_idx + speed, data['N'] - 1)
            time.sleep(0.2)
            st.rerun()
        else:
            st.session_state.auto_play = False
            st.success("✅ アニメーション完了！")
    
    # 詳細情報表示
    st.subheader("📊 現在の測定値")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("現在時刻", str(data['t'].iloc[st.session_state.curr_idx]))
    
    with col2:
        current_freq = data['freq'][st.session_state.curr_idx]
        st.metric("周波数 [Hz]", f"{current_freq:.3f}")
    
    with col3:
        x, _, _, _ = viz.get_x_theta_r_y(theta0)
        current_phase = x[st.session_state.curr_idx]
        st.metric("θ₀基準相対位相 [deg]", f"{current_phase:.1f}")
    
    with col4:
        delta, P = viz.get_power_curve(theta0)
        current_power = P[st.session_state.curr_idx]
        st.metric("電力P [pu]", f"{current_power:.3f}")
    
    # データ統計情報
    with st.expander("📈 データ統計情報", expanded=False):
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.write("**基本情報**")
            st.write(f"- データ点数: {data['N']:,}")
            st.write(f"- 時間範囲: {data['t'].iloc[0]} ～ {data['t'].iloc[-1]}")
        
        with col_stat2:
            st.write("**測定値範囲**")
            st.write(f"- 周波数: {data['freq'].min():.3f} ～ {data['freq'].max():.3f} Hz")
            st.write(f"- 位相差: {data['phase_diff'].min():.1f} ～ {data['phase_diff'].max():.1f} deg")

if __name__ == "__main__":
    main()
