import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import time
from matplotlib.gridspec import GridSpec

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def auto_find_csv(prefix):
    """CSVファイルを自動検索"""
    files = sorted(glob.glob(f'{prefix}_*.csv'))
    if not files:
        raise FileNotFoundError(f'{prefix}_*.csv が見つかりません')
    return files[0]

@st.cache_data
def load_data(freq_file=None, phase_file=None):
    """データ読み込み"""
    if freq_file is None:
        freq_file = auto_find_csv('f')
    if phase_file is None:
        phase_file = auto_find_csv('p')
    
    freq_df = pd.read_csv(freq_file, encoding='utf-8')
    phase_df = pd.read_csv(phase_file, encoding='utf-8')
    
    N = min(len(freq_df), len(phase_df))
    freq_df = freq_df.interpolate(method='linear', limit_direction='both')
    phase_df = phase_df.interpolate(method='linear', limit_direction='both')
    
    return freq_df.iloc[:N], phase_df.iloc[:N]

def process_data(freq_df, phase_df, sampling=10):
    """データ処理 - tkinterコードと同じロジック"""
    N = min(len(freq_df), len(phase_df))
    idx_target = 1
    idx_ref = 2
    idx_freq = 1
    
    # データ処理
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
    """tkinterクラスと同じ構造でStreamlit用に再設計"""
    
    def __init__(self, data):
        self.data = data
        self.E = 1.0
        self.V = 1.0
        self.X = 0.3
    
    def get_x_theta_r_y(self, theta0):
        """tkinterのget_x_theta_r_yと同じ"""
        x = np.mod(self.data['phase_diff'] + theta0 + 180, 360) - 180
        theta = np.deg2rad(x)
        y = 2*np.pi*self.data['freq']
        r = np.ones(self.data['N'])
        return x, theta, r, y
    
    def get_xs(self, theta0):
        """tkinterのget_xsと同じ"""
        xs = np.mod(self.data['phase_diff_s'] + theta0 + 180, 360) - 180
        return xs
    
    def get_power_curve(self, theta0):
        """tkinterのget_power_curveと同じ"""
        delta = np.mod(self.data['phase_diff'] + theta0 + 180, 360) - 180
        delta_rad = np.deg2rad(delta)
        P = (self.E*self.V/self.X)*np.sin(delta_rad)
        return delta, P
    
    def create_plots(self, curr_idx, theta0):
        """tkinterのinit_plot_objects + update_plotsを統合"""
        x, theta, r, y = self.get_x_theta_r_y(theta0)
        xs = self.get_xs(theta0)
        idx = curr_idx
        
        # 新しい図を作成
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 3)
        
        # --- 左上：円グラフ ---
        ax_polar1 = fig.add_subplot(gs[0,0], polar=True)
        ax_polar1.set_rlim(0, 1.1)
        ax_polar1.plot([theta[idx]], [r[idx]], 'o', markersize=10, color='tab:blue')
        ax_polar1.set_title(f'θ₀基準 相対位相[deg] 円グラフ\n時刻: {self.data["t"].iloc[idx]}', fontsize=10)
        
        # --- 中央上：xy散布図 ---
        ax_xy = fig.add_subplot(gs[0,1])
        ax_xy.plot(xs, self.data['freq_s']*2*np.pi, '.', color='tab:blue', alpha=0.2, markersize=4, label='全区間')
        ax_xy.plot(x[:idx+1], y[:idx+1], '.', color='tab:orange', markersize=8, label='履歴')
        ax_xy.plot([x[idx]], [y[idx]], 'r.', markersize=12, label='現在')
        ax_xy.set_xlabel('θ₀基準相対位相[deg]')
        ax_xy.set_ylabel('周波数[rad/s]')
        ax_xy.set_xlim(-180, 180)
        ax_xy.set_title('θ₀基準: 位相×freq 散布図')
        ax_xy.legend(loc='lower right')
        
        # --- 右上：極座標 ---
        ax_polar2 = fig.add_subplot(gs[0,2], polar=True)
        ax_polar2.set_rlim(np.nanmin(self.data['freq']), np.nanmax(self.data['freq'])*1.05)
        ax_polar2.plot([theta[idx]], [self.data['freq'][idx]], 'o', markersize=10, color='tab:red')
        ax_polar2.set_title('極座標: θ₀基準位相 × 周波数')
        
        # --- 下段左：時系列（2軸プロット） ---
        ax_time = fig.add_subplot(gs[1,:2])
        ax_time2 = ax_time.twinx()
        
        # 位相プロット
        ax_time.plot(self.data['t_s'], xs, color='tab:blue', alpha=0.2, linestyle='-', label='θ₀基準相対位相[deg] 全体(間引)')
        ax_time.plot(self.data['t'][:idx+1], x[:idx+1], color='tab:blue', linestyle='-', marker='.', markersize=3, label='θ₀基準相対位相[deg] 履歴')
        ax_time.plot([self.data['t'][idx]], [x[idx]], 'o', color='tab:blue')
        ax_time.set_ylabel('θ₀基準相対位相[deg]')
        ax_time.set_ylim(-180, 180)
        ax_time.set_xlabel('時刻')
        ax_time.set_xlim(self.data['t'].iloc[0], self.data['t'].iloc[-1])
        
        # 周波数プロット
        ax_time2.plot(self.data['t_s'], self.data['freq_s'], color='tab:red', alpha=0.2, label='周波数[Hz] 全体(間引)')
        ax_time2.plot(self.data['t'][:idx+1], self.data['freq'][:idx+1], color='tab:red', linestyle='-', marker='.', markersize=3, label='周波数[Hz] 履歴')
        ax_time2.plot([self.data['t'][idx]], [self.data['freq'][idx]], 's', color='tab:red')
        ax_time2.set_ylabel('周波数[Hz]')
        ax_time2.set_ylim(np.nanmin(self.data['freq'])-0.2, np.nanmax(self.data['freq'])+0.2)
        
        ax_time.set_title(f"時系列（θ₀基準相対位相＋周波数）\nカレント: {self.data['t'].iloc[idx]}", fontsize=10)
        
        # 凡例統合
        lines1, labels1 = ax_time.get_legend_handles_labels()
        lines2, labels2 = ax_time2.get_legend_handles_labels()
        ax_time.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # --- 下段右：Power Angle Curve ---
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
        
        plt.tight_layout()
        return fig

def main():
    st.set_page_config(page_title="PMU相対位相アニメUI", layout="wide")
    st.title("PMU相対位相アニメUI")
    
    # セッション状態の初期化
    if 'curr_idx' not in st.session_state:
        st.session_state.curr_idx = 0
    if 'auto_play' not in st.session_state:
        st.session_state.auto_play = False
    
    try:
        # データ読み込み
        if 'data' not in st.session_state:
            freq_df, phase_df = load_data()
            st.session_state.data = process_data(freq_df, phase_df)
            st.session_state.visualization = PMUVisualization(st.session_state.data)
        
        data = st.session_state.data
        viz = st.session_state.visualization
        
        # 制御パネル
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 2, 1])
        
        with col1:
            if st.button("▶️ Start", key="start_btn"):
                st.session_state.auto_play = True
        
        with col2:
            if st.button("⏹️ Stop", key="stop_btn"):
                st.session_state.auto_play = False
        
        with col3:
            if st.button("🔄 Reset", key="reset_btn"):
                st.session_state.curr_idx = 0
                st.session_state.auto_play = False
        
        with col4:
            speed = st.selectbox("再生速度", options=[1, 2, 5, 10, 20, 50, 100], index=3, key="speed_select")
        
        with col5:
            # 自動再生中でなければスライダーを表示
            if not st.session_state.auto_play:
                curr_idx = st.slider("時間位置", 0, data['N']-1, st.session_state.curr_idx, key="time_slider")
                st.session_state.curr_idx = curr_idx
            else:
                st.slider("時間位置", 0, data['N']-1, st.session_state.curr_idx, disabled=True, key="time_slider_disabled")
        
        with col6:
            theta0 = st.selectbox("基準θ₀[deg]", options=list(range(0, 45, 5)), index=6, key="theta0_select")
        
        # 状態表示
        status_col1, status_col2 = st.columns([3, 1])
        with status_col1:
            if st.session_state.auto_play:
                st.success("🔄 自動再生中...")
            else:
                st.info("⏸️ 停止中")
        
        # プロット作成・表示
        fig = viz.create_plots(st.session_state.curr_idx, theta0)
        st.pyplot(fig)
        plt.close(fig)
        
        # 自動再生の実行
        if st.session_state.auto_play:
            if st.session_state.curr_idx < data['N'] - 1:
                # インデックスを進める
                st.session_state.curr_idx = min(st.session_state.curr_idx + speed, data['N'] - 1)
                time.sleep(0.1)
                st.rerun()
            else:
                # 終了時の処理
                st.session_state.auto_play = False
                st.success("✅ アニメーション完了！")
                st.rerun()
        
        # 情報表示
        st.subheader("現在の情報")
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
            st.metric("インデックス", f"{st.session_state.curr_idx}/{data['N']-1}")
        
    except FileNotFoundError as e:
        st.error(f"CSVファイルが見つかりません: {e}")
        st.info("f_*.csv と p_*.csv ファイルをアップロードしてください")
        
        # ファイルアップロード機能
        st.subheader("ファイルアップロード")
        freq_file = st.file_uploader("周波数ファイル (f_*.csv)", type="csv")
        phase_file = st.file_uploader("位相ファイル (p_*.csv)", type="csv")
        
        if freq_file and phase_file:
            # アップロードされたファイルを一時保存
            with open(f"f_{freq_file.name}", "wb") as f:
                f.write(freq_file.getbuffer())
            with open(f"p_{phase_file.name}", "wb") as f:
                f.write(phase_file.getbuffer())
            
            st.success("ファイルがアップロードされました。ページを再読み込みしてください。")
            if st.button("データを再読み込み"):
                if 'data' in st.session_state:
                    del st.session_state.data
                    del st.session_state.visualization
                st.rerun()

if __name__ == "__main__":
    main()
