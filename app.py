import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.gridspec import GridSpec
import time

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

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
    """データ処理"""
    N = min(len(freq_df), len(phase_df))
    idx_target = 1
    idx_ref = 2
    idx_freq = 1
    
    # データ処理
    t = pd.to_datetime(phase_df.iloc[:,0])
    freq = freq_df.iloc[:, idx_freq].values
    phase_target = phase_df.iloc[:, idx_target].values
    phase_ref = phase_df.iloc[:, idx_ref].values
    
    # 位相の連続化
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

def get_processed_phase(phase_diff, theta0):
    """θ₀基準の位相計算"""
    x = np.mod(phase_diff + theta0 + 180, 360) - 180
    theta = np.deg2rad(x)
    return x, theta

def get_power_curve(phase_diff, theta0, E=1.0, V=1.0, X=0.3):
    """Power Angle Curve計算"""
    delta = np.mod(phase_diff + theta0 + 180, 360) - 180
    delta_rad = np.deg2rad(delta)
    P = (E*V/X)*np.sin(delta_rad)
    return delta, P

def create_plots(data, curr_idx, theta0):
    """プロット作成"""
    x, theta = get_processed_phase(data['phase_diff'], theta0)
    xs, _ = get_processed_phase(data['phase_diff_s'], theta0)
    y = 2*np.pi*data['freq']
    r = np.ones(data['N'])
    
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 3, figure=fig)
    
    # 左上：円グラフ
    ax_polar1 = fig.add_subplot(gs[0,0], polar=True)
    ax_polar1.set_rlim(0, 1.1)
    ax_polar1.plot([theta[curr_idx]], [r[curr_idx]], 'o', markersize=10, color='tab:blue')
    ax_polar1.set_title(f'θ₀基準 相対位相[deg] 円グラフ\\n時刻: {data["t"].iloc[curr_idx]}')
    
    # 中央上：xy散布図
    ax_xy = fig.add_subplot(gs[0,1])
    ax_xy.plot(xs, data['freq_s']*2*np.pi, '.', color='tab:blue', alpha=0.2, markersize=4, label='全区間')
    ax_xy.plot(x[:curr_idx+1], y[:curr_idx+1], '.', color='tab:orange', markersize=8, label='履歴')
    ax_xy.plot([x[curr_idx]], [y[curr_idx]], 'r.', markersize=12, label='現在')
    ax_xy.set_xlabel('θ₀基準相対位相[deg]')
    ax_xy.set_ylabel('周波数[rad/s]')
    ax_xy.set_xlim(-180, 180)
    ax_xy.set_title('θ₀基準: 位相×freq 散布図')
    ax_xy.legend(loc='lower right')
    
    # 右上：極座標
    ax_polar2 = fig.add_subplot(gs[0,2], polar=True)
    ax_polar2.set_rlim(np.nanmin(data['freq']), np.nanmax(data['freq'])*1.05)
    ax_polar2.plot([theta[curr_idx]], [data['freq'][curr_idx]], 'o', markersize=10, color='tab:red')
    ax_polar2.set_title('極座標: θ₀基準位相 × 周波数')
    
    # 下段左：時系列
    ax_time = fig.add_subplot(gs[1,:2])
    ax_time2 = ax_time.twinx()
    
    # 位相プロット
    ax_time.plot(data['t_s'], xs, color='tab:blue', alpha=0.2, linestyle='-', label='θ₀基準相対位相[deg] 全体')
    ax_time.plot(data['t'][:curr_idx+1], x[:curr_idx+1], color='tab:blue', linestyle='-', marker='.', markersize=3, label='θ₀基準相対位相[deg] 履歴')
    ax_time.plot([data['t'][curr_idx]], [x[curr_idx]], 'o', color='tab:blue')
    ax_time.set_ylabel('θ₀基準相対位相[deg]')
    ax_time.set_ylim(-180, 180)
    ax_time.set_xlabel('時刻')
    ax_time.set_xlim(data['t'].iloc[0], data['t'].iloc[-1])
    
    # 周波数プロット
    ax_time2.plot(data['t_s'], data['freq_s'], color='tab:red', alpha=0.2, label='周波数[Hz] 全体')
    ax_time2.plot(data['t'][:curr_idx+1], data['freq'][:curr_idx+1], color='tab:red', linestyle='-', marker='.', markersize=3, label='周波数[Hz] 履歴')
    ax_time2.plot([data['t'][curr_idx]], [data['freq'][curr_idx]], 's', color='tab:red')
    ax_time2.set_ylabel('周波数[Hz]')
    ax_time2.set_ylim(np.nanmin(data['freq'])-0.2, np.nanmax(data['freq'])+0.2)
    
    ax_time.set_title(f"時系列（θ₀基準相対位相＋周波数）\\nカレント: {data['t'].iloc[curr_idx]}")
    
    lines1, labels1 = ax_time.get_legend_handles_labels()
    lines2, labels2 = ax_time2.get_legend_handles_labels()
    ax_time.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 下段右：Power Angle Curve
    ax_power = fig.add_subplot(gs[1,2])
    delta_curve = np.linspace(-180, 180, 1000)
    P_curve = (1.0*1.0/0.3)*np.sin(np.deg2rad(delta_curve))
    ax_power.plot(delta_curve, P_curve, '--', color='gray', linewidth=1)
    
    delta, P = get_power_curve(data['phase_diff'], theta0)
    ax_power.plot(delta[:curr_idx+1], P[:curr_idx+1], '.', color='tab:blue', markersize=8)
    ax_power.plot([delta[curr_idx]], [P[curr_idx]], 'r.', markersize=12)
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
    
    # サイドバー設定
    st.sidebar.header("設定")
    
    # データ読み込み
    try:
        if 'data' not in st.session_state:
            freq_df, phase_df = load_data()
            st.session_state.data = process_data(freq_df, phase_df)
            st.session_state.curr_idx = 0
        
        data = st.session_state.data
        
        # 制御パネル
        col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
        
        with col1:
            theta0 = st.selectbox("基準θ₀[deg]", options=list(range(0, 45, 5)), index=6)
        
        with col2:
            speed = st.selectbox("再生速度", options=[1, 2, 5, 10, 20, 50, 100], index=3)
        
        with col3:
            curr_idx = st.slider("時間位置", 0, data['N']-1, st.session_state.curr_idx)
            st.session_state.curr_idx = curr_idx
        
        with col4:
            auto_play = st.checkbox("自動再生")
        
        # 自動再生機能
        if auto_play and 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()
        
        if auto_play:
            current_time = time.time()
            if current_time - st.session_state.get('last_update', 0) > 0.1:  # 100ms間隔
                if st.session_state.curr_idx < data['N'] - 1:
                    st.session_state.curr_idx = min(st.session_state.curr_idx + speed, data['N'] - 1)
                    st.session_state.last_update = current_time
                    st.rerun()
                else:
                    st.session_state.curr_idx = 0  # リセット
        
        # プロット作成・表示
        fig = create_plots(data, st.session_state.curr_idx, theta0)
        st.pyplot(fig)
        
        # 情報表示
        st.subheader("現在の情報")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("現在時刻", str(data['t'].iloc[st.session_state.curr_idx]))
        
        with col2:
            current_freq = data['freq'][st.session_state.curr_idx]
            st.metric("周波数 [Hz]", f"{current_freq:.3f}")
        
        with col3:
            x, _ = get_processed_phase(data['phase_diff'], theta0)
            current_phase = x[st.session_state.curr_idx]
            st.metric("θ₀基準相対位相 [deg]", f"{current_phase:.1f}")
        
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
                st.rerun()

if __name__ == "__main__":
    main()
