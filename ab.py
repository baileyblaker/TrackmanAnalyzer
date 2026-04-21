import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import plotly.graph_objects as go
from scipy.stats import norm as scipy_norm
import io
import requests

st.set_page_config(page_title="Stuff+ Arsenal Builder", layout="wide", initial_sidebar_state="expanded")

st.markdown("<style>div[data-testid='stDecoration']{display:none}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# TrackMan CSV column mapping (handles common naming variations)
# ------------------------------------------------------------------
TM_COL_MAP = {
    'pitcher':          ['Pitcher', 'PitcherName', 'pitcher', 'pitcher_name'],
    'pitch_type':       ['TaggedPitchType', 'AutoPitchType', 'pitch_type', 'PitchType'],
    'rel_speed':        ['RelSpeed', 'Velocity', 'rel_speed', 'velo', 'Speed'],
    'induced_vert':     ['InducedVertBreak', 'InducedVBreak', 'induced_vert_break', 'iVB', 'InducedVB'],
    'horz_break':       ['HorzBreak', 'HorizontalBreak', 'horz_break', 'HB'],
    'spin_rate':        ['SpinRate', 'spin_rate', 'Spin', 'SpinRPM'],
    'extension':        ['Extension', 'ReleaseExtension', 'extension', 'Ext'],
    'rel_height':       ['RelHeight', 'ReleaseHeight', 'rel_height', 'RelZ'],
    'rel_side':         ['RelSide', 'ReleaseSide', 'rel_side', 'RelX'],
    'pitcher_hand':     ['PitcherThrows', 'ThrowingHand', 'pitcher_hand', 'Throws', 'Hand'],
    'plate_loc_side':   ['PlateLocSide', 'plate_loc_side', 'PlateX', 'px'],
    'plate_loc_height': ['PlateLocHeight', 'plate_loc_height', 'PlateZ', 'pz'],
    # Spin profile — used to infer pronator vs supinator
    'spin_efficiency':  ['SpinEfficiency', 'ActiveSpin', 'ActiveSpinPercent', 'SpinEff',
                         'spin_efficiency', 'SpinEfficiencyPercent', 'SpinAxis3dSpinEfficiency'],
    'spin_axis':        ['SpinAxis', 'Tilt', 'spin_axis', 'SpinAxis3d', 'SpinAxis3D'],
    # Pitch outcome / context columns
    'pitch_call':       ['PitchCall', 'pitch_call', 'Call', 'KorBB', 'korbb', 'PitchResult'],
    'play_result':      ['PlayResult', 'play_result', 'TaggedHitType', 'HitType'],
    'batter':           ['Batter', 'BatterName', 'batter', 'batter_name'],
    'inning':           ['Inning', 'inning', 'Inn'],
    'count_balls':      ['Balls', 'balls', 'Ball'],
    'count_strikes':    ['Strikes', 'strikes', 'Strike'],
}

PITCH_TYPE_NORMALIZE = {
    'Four-Seam': 'Fastball', 'FourSeam': 'Fastball', '4-Seam': 'Fastball', '4S': 'Fastball',
    'FF': 'Fastball', 'FA': 'Fastball', 'Fastball': 'Fastball',
    'Two-Seam': 'Sinker', 'TwoSeam': 'Sinker', '2S': 'Sinker', 'SI': 'Sinker',
    'Sinker': 'Sinker', 'SN': 'Sinker',
    'Slider': 'Slider', 'SL': 'Slider', 'Sweeper': 'Slider',
    'Curveball': 'Curveball', 'CU': 'Curveball', 'CB': 'Curveball', 'KC': 'Curveball',
    'KnuckleCurve': 'Curveball', '12-6': 'Curveball',
    'ChangeUp': 'ChangeUp', 'Changeup': 'ChangeUp', 'CH': 'ChangeUp',
    'Change-Up': 'ChangeUp', 'CHS': 'ChangeUp',
    'Cutter': 'Cutter', 'CT': 'Cutter', 'FC': 'Cutter',
    'Splitter': 'Splitter', 'SP': 'Splitter', 'FS': 'Splitter', 'Split': 'Splitter',
}

VALID_PITCH_TYPES = ["Fastball", "Sinker", "Slider", "Curveball", "ChangeUp", "Cutter", "Splitter"]

# Pitch call display groupings — used to color-code outcomes
CALL_COLORS = {
    # Positive outcomes (pitcher)
    'StrikeSwinging':   '#22C55E',
    'StrikeCalled':     '#4ADE80',
    'Strikeout':        '#16A34A',
    'FoulBall':         '#86EFAC',
    'FoulTip':          '#86EFAC',
    # Neutral
    'BallCalled':       '#94A3B8',
    'Ball':             '#94A3B8',
    'HitByPitch':       '#CBD5E1',
    'IntentionalWalk':  '#CBD5E1',
    # Negative outcomes (pitcher)
    'Single':           '#FCD34D',
    'Double':           '#F97316',
    'Triple':           '#EF4444',
    'HomeRun':          '#DC2626',
    'InPlay':           '#FB923C',
    'FieldersChoice':   '#FDE68A',
    'Sacrifice':        '#FDE68A',
    'Error':            '#F87171',
}

def get_call_color(call):
    if pd.isna(call): return '#64748B'
    for key, color in CALL_COLORS.items():
        if key.lower() in str(call).lower().replace(' ', '').replace('_', ''):
            return color
    return '#64748B'

def get_call_emoji(call):
    if pd.isna(call): return ''
    c = str(call).lower().replace(' ', '').replace('_', '')
    if 'swinging' in c or 'strikeswing' in c: return '🌀'
    if 'called' in c and 'strike' in c:       return '🔴'
    if 'strikeout' in c:                      return '⚡'
    if 'foul' in c:                            return '〽️'
    if 'homerun' in c or 'hr' == c:           return '💥'
    if 'triple' in c:                          return '🔺'
    if 'double' in c:                          return '✌️'
    if 'single' in c:                          return '➡️'
    if 'walk' in c:                            return '🚶'
    if 'ball' in c:                            return '⚪'
    if 'fielderschoice' in c or 'fc' == c:    return '🔀'
    if 'sacrifice' in c or 'sac' in c:        return '🫴'
    if 'error' in c:                           return '❌'
    if 'out' in c:                             return '🟥'
    if 'inplay' in c:                          return '🏃'
    if 'hitbypitch' in c or 'hbp' == c:       return '🩹'
    return '▪️'

def find_col(df, key):
    for name in TM_COL_MAP.get(key, []):
        if name in df.columns:
            return name
    return None

def normalize_pitch_type(pt):
    if pd.isna(pt): return None
    pt_str = str(pt).strip()
    return PITCH_TYPE_NORMALIZE.get(pt_str, pt_str if pt_str in VALID_PITCH_TYPES else None)

def parse_trackman_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Could not read CSV: {e}"

    col_results = {k: find_col(df, k) for k in TM_COL_MAP}
    missing_critical = [k for k in ['pitcher', 'pitch_type', 'rel_speed', 'induced_vert', 'horz_break', 'spin_rate']
                        if col_results[k] is None]
    if missing_critical:
        return None, f"Missing required columns: {missing_critical}. Found columns: {list(df.columns[:20])}"

    out = pd.DataFrame()
    for key, col in col_results.items():
        if col: out[key] = df[col]
        else:   out[key] = np.nan

    out['pitch_type_raw'] = out['pitch_type'].copy()
    out['pitch_type']     = out['pitch_type'].apply(normalize_pitch_type)

    numeric_cols = ['rel_speed', 'induced_vert', 'horz_break', 'spin_rate']
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors='coerce')
    out = out.dropna(subset=numeric_cols + ['pitcher', 'pitch_type'])
    out = out[out['pitch_type'].isin(VALID_PITCH_TYPES)]

    out['extension']  = pd.to_numeric(out['extension'],  errors='coerce').fillna(6.0)
    out['rel_height'] = pd.to_numeric(out['rel_height'], errors='coerce').fillna(5.5)
    out['rel_side']   = pd.to_numeric(out['rel_side'],   errors='coerce').fillna(2.0)

    if col_results['pitcher_hand']:
        out['pitcher_hand'] = df[col_results['pitcher_hand']].map(
            lambda x: 'Right' if str(x).strip() in ['R', 'Right', 'RHP', 'right'] else 'Left')
    else:
        out['pitcher_hand'] = out['rel_side'].apply(lambda x: 'Right' if x > 0 else 'Left')

    # Reset index so pitch numbers are sequential
    out = out.reset_index(drop=True)
    out['pitch_number'] = out.index + 1

    return out, None


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    with open('stuff_event_models.pkl', 'rb') as f:
        event_models = pickle.load(f)
    final_model = lgb.Booster(model_file='stuff_final_model.txt')
    with open('stuff_normalization_params.pkl', 'rb') as f:
        norm_params = pickle.load(f)
    with open('stuff_model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    with open('pitching_event_models.pkl', 'rb') as f:
        pitching_event_models = pickle.load(f)
    pitching_final_model = lgb.Booster(model_file='pitching_final_model.txt')
    with open('location_sd_params.pkl', 'rb') as f:
        location_sd_params = pickle.load(f)
    return event_models, final_model, norm_params, metadata, pitching_event_models, pitching_final_model, location_sd_params

event_models, final_model, norm_params, metadata, pitching_event_models, pitching_final_model, location_sd_params = load_models()
stuff_features    = metadata['stuff_features']
pitching_features = metadata['pitching_features']
pitch_type_map    = metadata['pitch_type_map']
fb_types          = metadata['fb_types']
labels            = metadata['labels']

def fmt(x):
    if pd.isna(x): return x
    try:
        v = float(x)
        return int(v) if v == int(v) else round(v, 1)
    except:
        return x

PITCH_DEFAULTS = {
    "Fastball": {"velo": 94.0, "ivb": 16.0, "hb": 8.0,  "spin": 2300},
    "Sinker":   {"velo": 93.0, "ivb": 8.0,  "hb": 14.0, "spin": 2150},
    "Slider":   {"velo": 85.0, "ivb": 1.0,  "hb": -5.0, "spin": 2450},
    "Curveball":{"velo": 78.0, "ivb": -8.0, "hb": 6.0,  "spin": 2600},
    "ChangeUp": {"velo": 84.0, "ivb": 10.0, "hb": 12.0, "spin": 1700},
    "Cutter":   {"velo": 89.0, "ivb": 10.0, "hb": -2.0, "spin": 2400},
    "Splitter": {"velo": 86.0, "ivb": 4.0,  "hb": 10.0, "spin": 1400},
}

PITCH_COLORS = {
    'Fastball':  '#E53935',
    'Sinker':    '#FF7043',
    'Slider':    '#8E24AA',
    'Curveball': '#1E88E5',
    'ChangeUp':  '#43A047',
    'Cutter':    '#FFB300',
    'Splitter':  '#00ACC1',
}

if 'pitches'       not in st.session_state: st.session_state.pitches = []
if 'active_tab'    not in st.session_state: st.session_state.active_tab = 0
if 'tm_data'       not in st.session_state: st.session_state.tm_data = None
if 'input_mode'    not in st.session_state: st.session_state.input_mode = "Arsenal Builder"

# ------------------------------------------------------------------
# Gemini 2.5 Flash — AI scouting report
# ------------------------------------------------------------------
import time

GEMINI_API_KEY = "AIzaSyDxXZBbww4qr8v2ew0PGEYBRVkZN-EBPIU"
# Free-tier models (as of April 2026). gemini-2.0-flash was retired March 2026.
# Flash-Lite has the highest free-tier quota (15 RPM / 1,000 RPD) and is more
# than capable of a 150-word scouting report. Full Flash is a quality fallback.
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

def _call_gemini(model, prompt, api_key, max_retries=2):
    """Call a single Gemini model with quick retries. Returns (text, error_str)."""
    url = GEMINI_URL_TMPL.format(model=model)
    gen_cfg = {"temperature": 0.4, "maxOutputTokens": 1024}
    # Disable "thinking" on 2.5-family models — massive speed win for short tasks.
    if "2.5" in model:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, params={"key": api_key}, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    cand = data["candidates"][0]
                    text = cand["content"]["parts"][0]["text"].strip()
                    finish = cand.get("finishReason", "")
                    # Truncated output — try once more with more headroom
                    if finish == "MAX_TOKENS" and attempt < max_retries - 1:
                        gen_cfg["maxOutputTokens"] = 2048
                        last_err = "MAX_TOKENS (retrying with higher limit)"
                        continue
                    return text, None
                except (KeyError, IndexError):
                    return None, f"Unexpected response shape: {str(data)[:200]}"
            # 429 = quota / rate limit. All models share the same free-tier quota,
            # so trying fallback models won't help. Signal the caller to stop.
            if resp.status_code == 429:
                return None, "RATE_LIMITED"
            # Retry on other transient errors
            if resp.status_code in (500, 502, 503, 504):
                last_err = f"{resp.status_code} {resp.reason}"
                if attempt < max_retries - 1:
                    time.sleep(0.75)
                continue
            # Non-retryable (400, 401, 403, etc.) — fail fast
            return None, f"{resp.status_code} {resp.reason}: {resp.text[:250]}"
        except requests.Timeout:
            last_err = "Timeout"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    return None, last_err or "Unknown error"


def generate_gemini_summary(pitcher_name, is_righty, scores_df, fb_metrics,
                            max_velo=None, total_pitches=None, api_key=GEMINI_API_KEY):
    """Generate a ~150 word scouting report via Gemini, with retries + model fallback."""
    hand_str = "RHP" if is_righty else "LHP"
    lines = [f"Pitcher: {pitcher_name} ({hand_str})"]
    if total_pitches:
        lines.append(f"Total pitches: {total_pitches}")
    if max_velo is not None:
        lines.append(f"Max velocity: {max_velo:.1f} mph")
    lines.append(
        f"Primary FB avg: {fb_metrics['velo']:.1f} mph, "
        f"{fb_metrics['ivb']:+.1f}\" IVB, {fb_metrics['hb']:+.1f}\" HB, "
        f"{fb_metrics['spin']:.0f} rpm"
    )
    lines.append("\nArsenal (100 = college avg; 110+ elite, <90 below avg):")
    for _, r in scores_df.iterrows():
        line = (
            f"- {r['PitchType']}: {r['RelSpeed']:.1f} mph, "
            f"{r['InducedVertBreak']:+.1f}\" IVB, {r['HorzBreak']:+.1f}\" HB, "
            f"{int(r['SpinRate'])} rpm | Stuff+ {r['StuffPlus']:.0f}"
        )
        if 'PitchingPlus' in scores_df.columns and pd.notna(r.get('PitchingPlus')):
            line += f", Pitching+ {r['PitchingPlus']:.0f}"
        if 'WhiffPlus' in scores_df.columns and pd.notna(r.get('WhiffPlus')):
            line += f", Whiff+ {r['WhiffPlus']:.0f}"
        if 'CalledStrikePlus' in scores_df.columns and pd.notna(r.get('CalledStrikePlus')):
            line += f", CSW+ {r['CalledStrikePlus']:.0f}"
        # Release + spin profile (when available)
        rel_bits = []
        if 'Extension' in scores_df.columns and pd.notna(r.get('Extension')):
            rel_bits.append(f"Ext {r['Extension']:.1f}ft")
        if 'RelHeight' in scores_df.columns and pd.notna(r.get('RelHeight')):
            rel_bits.append(f"RelH {r['RelHeight']:.1f}ft")
        if 'RelSide' in scores_df.columns and pd.notna(r.get('RelSide')):
            rel_bits.append(f"RelS {r['RelSide']:+.1f}ft")
        if rel_bits:
            line += "  [" + " / ".join(rel_bits) + "]"
        spin_bits = []
        if 'SpinEfficiency' in scores_df.columns and pd.notna(r.get('SpinEfficiency')):
            se = float(r['SpinEfficiency'])
            # Normalize 0–1 vs 0–100 representation
            se_pct = se * 100 if se <= 1.0 else se
            spin_bits.append(f"SpinEff {se_pct:.0f}%")
        if 'SpinAxis' in scores_df.columns and pd.notna(r.get('SpinAxis')):
            spin_bits.append(f"Axis {r['SpinAxis']:.0f}°")
        if spin_bits:
            line += "  {" + ", ".join(spin_bits) + "}"
        if 'count' in scores_df.columns and pd.notna(r.get('count')):
            line += f"  ({int(r['count'])} thrown)"
        lines.append(line)
    data_block = "\n".join(lines)

    prompt = f"""You are an experienced college baseball pitching analyst. Write a concise ~180 word scouting report on the pitcher below.

DATA:
{data_block}
"""

    errors = []
    for model in GEMINI_MODELS:
        text, err = _call_gemini(model, prompt, api_key)
        if text:
            return text, model
        # Rate limit applies to the whole key, not just one model —
        # bail immediately instead of hammering siblings with the same quota.
        if err == "RATE_LIMITED":
            msg = (
                "⏱️  **Rate limit hit.** You've exceeded the Gemini free-tier quota.\n\n"
                "Current free-tier limits (April 2026):\n"
                "- `gemini-2.5-flash-lite`: 15 requests/minute, **1,000/day**\n"
                "- `gemini-2.5-flash`: 10 requests/minute, 250/day\n\n"
                "**What to do:**\n"
                "- Wait ~60 seconds and try again (per-minute limits are rolling windows), **or**\n"
                "- If you've hit the daily cap, wait until midnight Pacific (2 AM Central), **or**\n"
                "- Enable billing at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) "
                "to unlock Tier 1 (150–300 RPM, 1,500 RPD) — still free until you actually spend."
            )
            return msg, None
        errors.append(f"`{model}` → {err}")
    msg = (
        "⚠️  All Gemini models are unavailable right now. This is a Google-side issue "
        "(usually temporary high demand). Try again in 30–60 seconds.\n\n"
        + "\n".join(errors)
    )
    return msg, None


# ------------------------------------------------------------------
# Helpers: Score computation
# ------------------------------------------------------------------
def compute_stuff_scores(pitches_list, pitcher_is_righty):
    pitch_counts = {}
    for p in pitches_list:
        if p['pitch_type'] in fb_types:
            pitch_counts[p['pitch_type']] = pitch_counts.get(p['pitch_type'], 0) + 1

    if pitch_counts:
        primary_fb = max(pitch_counts, key=pitch_counts.get)
        fb_pitches = [p for p in pitches_list if p['pitch_type'] == primary_fb]
    else:
        fb_pitches = [pitches_list[0]]

    avg_fb_vel  = np.mean([p['rel_speed']    for p in fb_pitches])
    avg_fb_vert = np.mean([p['induced_vert'] for p in fb_pitches])
    avg_fb_horz = np.mean([p['horz_break']   for p in fb_pitches])
    avg_fb_spin = np.mean([p['spin_rate']     for p in fb_pitches])

    rows = []
    for p in pitches_list:
        rows.append({
            'RelSpeed':          p['rel_speed'],
            'InducedVertBreak':  p['induced_vert'],
            'HorzBreak':         p['horz_break'],
            'SpinRate':          p['spin_rate'],
            'Extension':         p['extension'],
            'RelHeight':         p['rel_height'],
            'RelSide':           p['rel_side'],
            'TaggedPitchTypeNum':pitch_type_map[p['pitch_type']],
            'pitcher_is_righty': pitcher_is_righty,
            'diff_fb_vel':       avg_fb_vel  - p['rel_speed'],
            'diff_fb_vert':      avg_fb_vert - p['induced_vert'],
            'diff_fb_horz':      avg_fb_horz - p['horz_break'],
            'diff_spinrate':     avg_fb_spin - p['spin_rate'],
        })
    df = pd.DataFrame(rows)

    for label in labels:
        df[f'sp_{label}'] = event_models[label].predict(
            df[stuff_features].values, num_iteration=event_models[label].best_iteration)

    spf = [c for c in df.columns if c.startswith('sp_')]
    df['stuff_lw']         = final_model.predict(df[spf].values)
    df['StuffPlus']        = 100 + (norm_params['mu_stuff']       - df['stuff_lw'])                / norm_params['sigma_stuff']          * 10
    df['WhiffPlus']        = 100 + (df['sp_StrikeSwinging']       - norm_params['mu_whiff'])        / norm_params['sigma_whiff']          * 10
    df['CalledStrikePlus'] = 100 + (df['sp_StrikeCalled']         - norm_params['mu_called_strike']) / norm_params['sigma_called_strike']  * 10
    df['wOBACON']          = (
        (df['sp_Single']*0.89 + df['sp_Double']*1.27 + df['sp_Triple']*1.62 + df['sp_HomeRun']*2.10) /
        (df['sp_Single'] + df['sp_Double'] + df['sp_Triple'] + df['sp_HomeRun'] + 0.0001))
    df['wOBACONPlus']      = 100 - (df['wOBACON'] - norm_params['mu_wobacon']) / norm_params['sigma_wobacon'] * 10
    df['PitchType']        = [p['pitch_type'] for p in pitches_list]
    return df, avg_fb_vel, avg_fb_vert, avg_fb_horz, avg_fb_spin


@st.cache_data
def compute_zone_heatmap(pitch_row: dict, avg_fb_vel, avg_fb_vert, avg_fb_horz, avg_fb_spin,
                          pitcher_is_righty, _pitching_event_models, _pitching_final_model,
                          norm_params, pitching_features, pitch_type_map, labels, grid_n=25):
    side_vals   = np.linspace(-3.0, 3.0, grid_n)
    height_vals = np.linspace(-0.5, 5.5, grid_n)
    sg, hg = np.meshgrid(side_vals, height_vals)
    n = sg.size

    data = {
        'RelSpeed':          np.full(n, pitch_row['rel_speed']),
        'InducedVertBreak':  np.full(n, pitch_row['induced_vert']),
        'HorzBreak':         np.full(n, pitch_row['horz_break']),
        'SpinRate':          np.full(n, pitch_row['spin_rate']),
        'Extension':         np.full(n, pitch_row['extension']),
        'RelHeight':         np.full(n, pitch_row['rel_height']),
        'RelSide':           np.full(n, pitch_row['rel_side']),
        'TaggedPitchTypeNum':np.full(n, pitch_type_map[pitch_row['pitch_type']]),
        'pitcher_is_righty': np.full(n, pitcher_is_righty),
        'diff_fb_vel':       np.full(n, avg_fb_vel  - pitch_row['rel_speed']),
        'diff_fb_vert':      np.full(n, avg_fb_vert - pitch_row['induced_vert']),
        'diff_fb_horz':      np.full(n, avg_fb_horz - pitch_row['horz_break']),
        'diff_spinrate':     np.full(n, avg_fb_spin - pitch_row['spin_rate']),
        'PlateLocSide':      sg.flatten(),
        'PlateLocHeight':    hg.flatten(),
    }
    df_g = pd.DataFrame(data)[pitching_features]

    pred_cols = []
    for label in labels:
        col = f'pp_{label}'
        df_g[col] = _pitching_event_models[label].predict(
            df_g[pitching_features].values,
            num_iteration=_pitching_event_models[label].best_iteration)
        pred_cols.append(col)

    df_g['lw'] = _pitching_final_model.predict(df_g[pred_cols].values)
    df_g['PitchingPlus']    = 100 + (norm_params['mu_pitching'] - df_g['lw']) / norm_params['sigma_pitching'] * 10
    df_g['WhiffPlus']       = 100 + (df_g['pp_StrikeSwinging'] - norm_params['mu_whiff_pitching'])         / norm_params['sigma_whiff_pitching']         * 10
    df_g['CalledStrikePlus']= 100 + (df_g['pp_StrikeCalled']   - norm_params['mu_called_strike_pitching']) / norm_params['sigma_called_strike_pitching'] * 10

    return side_vals, height_vals, {
        'Pitching+':     df_g['PitchingPlus'].values.reshape(grid_n, grid_n),
        'Whiff+':        df_g['WhiffPlus'].values.reshape(grid_n, grid_n),
        'CalledStrike+': df_g['CalledStrikePlus'].values.reshape(grid_n, grid_n),
    }


def compute_optimal_zone(grid, side_vals, height_vals, sd_side, sd_height, grid_n=25):
    sg, hg = np.meshgrid(side_vals, height_vals)
    cell_area = (side_vals[1] - side_vals[0]) * (height_vals[1] - height_vals[0])
    expected_plus = np.zeros((grid_n, grid_n))
    for i, aim_h in enumerate(height_vals):
        for j, aim_s in enumerate(side_vals):
            w = scipy_norm.pdf(sg, aim_s, sd_side) * scipy_norm.pdf(hg, aim_h, sd_height) * cell_area
            w /= w.sum()
            expected_plus[i, j] = (w * grid).sum()
    best_idx = np.unravel_index(np.argmax(expected_plus), expected_plus.shape)
    best_aim_s = side_vals[best_idx[1]]
    best_aim_h = height_vals[best_idx[0]]
    best_expected = expected_plus[best_idx]
    w = scipy_norm.pdf(sg, best_aim_s, sd_side) * scipy_norm.pdf(hg, best_aim_h, sd_height) * cell_area
    w /= w.sum()
    prob_favorable = float((w[grid >= 100]).sum())
    return expected_plus, best_aim_s, best_aim_h, best_expected, prob_favorable


# ------------------------------------------------------------------
# Build pitch summary from TrackMan data
# ------------------------------------------------------------------
def tm_to_pitches(tm_df, pitcher_name):
    sub = tm_df[tm_df['pitcher'] == pitcher_name].copy()
    if sub.empty: return [], 'Right'

    hand = sub['pitcher_hand'].mode()[0] if 'pitcher_hand' in sub.columns else 'Right'

    # Coerce optional spin columns to numeric if present
    for opt_col in ('spin_efficiency', 'spin_axis'):
        if opt_col in sub.columns:
            sub[opt_col] = pd.to_numeric(sub[opt_col], errors='coerce')

    pitches = []
    for pt, grp in sub.groupby('pitch_type'):
        if pt not in VALID_PITCH_TYPES: continue
        rec = {
            'pitch_type':   pt,
            'rel_speed':    float(grp['rel_speed'].mean()),
            'induced_vert': float(grp['induced_vert'].mean()),
            'horz_break':   float(grp['horz_break'].mean()),
            'spin_rate':    float(grp['spin_rate'].mean()),
            'extension':    float(grp['extension'].mean()),
            'rel_height':   float(grp['rel_height'].mean()),
            'rel_side':     float(grp['rel_side'].mean()),
            'count':        len(grp),
        }
        if 'spin_efficiency' in grp.columns and grp['spin_efficiency'].notna().any():
            rec['spin_efficiency'] = float(grp['spin_efficiency'].mean())
        if 'spin_axis' in grp.columns and grp['spin_axis'].notna().any():
            rec['spin_axis'] = float(grp['spin_axis'].mean())
        pitches.append(rec)
    return pitches, hand


def compute_stuff_scores_from_tm(tm_df, pitcher_name, pitcher_is_righty):
    sub = tm_df[tm_df['pitcher'] == pitcher_name].copy()
    sub = sub[sub['pitch_type'].isin(VALID_PITCH_TYPES)].reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame(), 0, 0, 0, 0

    fb_sub = sub[sub['pitch_type'].isin(fb_types)]
    if fb_sub.empty: fb_sub = sub.head(1)
    avg_fb_vel  = float(fb_sub['rel_speed'].mean())
    avg_fb_vert = float(fb_sub['induced_vert'].mean())
    avg_fb_horz = float(fb_sub['horz_break'].mean())
    avg_fb_spin = float(fb_sub['spin_rate'].mean())

    feat = pd.DataFrame({
        'RelSpeed':          sub['rel_speed'],
        'InducedVertBreak':  sub['induced_vert'],
        'HorzBreak':         sub['horz_break'],
        'SpinRate':          sub['spin_rate'],
        'Extension':         sub['extension'],
        'RelHeight':         sub['rel_height'],
        'RelSide':           sub['rel_side'],
        'TaggedPitchTypeNum':sub['pitch_type'].map(pitch_type_map),
        'pitcher_is_righty': pitcher_is_righty,
        'diff_fb_vel':       avg_fb_vel  - sub['rel_speed'],
        'diff_fb_vert':      avg_fb_vert - sub['induced_vert'],
        'diff_fb_horz':      avg_fb_horz - sub['horz_break'],
        'diff_spinrate':     avg_fb_spin - sub['spin_rate'],
    })

    for label in labels:
        feat[f'sp_{label}'] = event_models[label].predict(
            feat[stuff_features].values, num_iteration=event_models[label].best_iteration)

    spf = [c for c in feat.columns if c.startswith('sp_')]
    # Store raw linear weight outputs — do NOT convert to +scale yet.
    # We average the lw values first, then apply mu/sigma once per pitch type.
    feat['stuff_lw'] = final_model.predict(feat[spf].values)
    feat['pitch_type'] = sub['pitch_type'].values

    has_loc = (
        'plate_loc_side' in sub.columns and
        'plate_loc_height' in sub.columns and
        sub['plate_loc_side'].notna().any()
    )
    if has_loc:
        loc_mask = sub['plate_loc_side'].notna() & sub['plate_loc_height'].notna()
        pfeat = pd.DataFrame({
            'RelSpeed':          sub.loc[loc_mask, 'rel_speed'],
            'InducedVertBreak':  sub.loc[loc_mask, 'induced_vert'],
            'HorzBreak':         sub.loc[loc_mask, 'horz_break'],
            'SpinRate':          sub.loc[loc_mask, 'spin_rate'],
            'Extension':         sub.loc[loc_mask, 'extension'],
            'RelHeight':         sub.loc[loc_mask, 'rel_height'],
            'RelSide':           sub.loc[loc_mask, 'rel_side'],
            'TaggedPitchTypeNum':sub.loc[loc_mask, 'pitch_type'].map(pitch_type_map),
            'pitcher_is_righty': pitcher_is_righty,
            'diff_fb_vel':       avg_fb_vel  - sub.loc[loc_mask, 'rel_speed'],
            'diff_fb_vert':      avg_fb_vert - sub.loc[loc_mask, 'induced_vert'],
            'diff_fb_horz':      avg_fb_horz - sub.loc[loc_mask, 'horz_break'],
            'diff_spinrate':     avg_fb_spin - sub.loc[loc_mask, 'spin_rate'],
            'PlateLocSide':      sub.loc[loc_mask, 'plate_loc_side'],
            'PlateLocHeight':    sub.loc[loc_mask, 'plate_loc_height'],
        }).reset_index(drop=True)
        pred_cols = []
        for label in labels:
            col = f'pp_{label}'
            pfeat[col] = pitching_event_models[label].predict(
                pfeat[pitching_features].values,
                num_iteration=pitching_event_models[label].best_iteration)
            pred_cols.append(col)
        pfeat['pitching_lw'] = pitching_final_model.predict(pfeat[pred_cols].values)
        pfeat['pitch_type']  = sub.loc[loc_mask, 'pitch_type'].values
        feat.loc[loc_mask, 'pitching_lw']        = pfeat['pitching_lw'].values
        feat.loc[loc_mask, 'pp_StrikeSwinging']  = pfeat['pp_StrikeSwinging'].values
        feat.loc[loc_mask, 'pp_StrikeCalled']    = pfeat['pp_StrikeCalled'].values

    # --- Aggregate raw lw + event probs per pitch type, THEN convert to +scale ---
    agg_dict = dict(
        stuff_lw_mean    =('stuff_lw',         'mean'),
        WhiffProb        =('sp_StrikeSwinging', 'mean'),
        CalledStrikeProb =('sp_StrikeCalled',   'mean'),
        wOBACON_single   =('sp_Single',         'mean'),
        wOBACON_double   =('sp_Double',         'mean'),
        wOBACON_triple   =('sp_Triple',         'mean'),
        wOBACON_hr       =('sp_HomeRun',        'mean'),
        sp_StrikeSwinging=('sp_StrikeSwinging', 'mean'),
        sp_StrikeCalled  =('sp_StrikeCalled',   'mean'),
        sp_HomeRun       =('sp_HomeRun',        'mean'),
        count            =('stuff_lw',          'count'),
    )
    if has_loc:
        agg_dict['pitching_lw_mean'] = ('pitching_lw', 'mean')
    agg = feat.groupby('pitch_type').agg(**agg_dict).reset_index()

    # Apply mu/sigma to the AVERAGED lw — this is the correct approach
    agg['StuffPlus']        = 100 + (norm_params['mu_stuff']        - agg['stuff_lw_mean'])    / norm_params['sigma_stuff']          * 10
    agg['WhiffPlus']        = 100 + (agg['WhiffProb']               - norm_params['mu_whiff']) / norm_params['sigma_whiff']          * 10
    agg['CalledStrikePlus'] = 100 + (agg['CalledStrikeProb']        - norm_params['mu_called_strike']) / norm_params['sigma_called_strike'] * 10
    woba_denom              = agg['wOBACON_single'] + agg['wOBACON_double'] + agg['wOBACON_triple'] + agg['wOBACON_hr'] + 0.0001
    agg['wOBACON']          = (agg['wOBACON_single']*0.89 + agg['wOBACON_double']*1.27 + agg['wOBACON_triple']*1.62 + agg['wOBACON_hr']*2.10) / woba_denom
    agg['wOBACONPlus']      = 100 - (agg['wOBACON'] - norm_params['mu_wobacon']) / norm_params['sigma_wobacon'] * 10
    if has_loc:
        agg['PitchingPlus'] = 100 + (norm_params['mu_pitching'] - agg['pitching_lw_mean']) / norm_params['sigma_pitching'] * 10

    # Drop intermediate columns not needed downstream
    drop_cols = ['stuff_lw_mean', 'WhiffProb', 'CalledStrikeProb',
                 'wOBACON_single', 'wOBACON_double', 'wOBACON_triple', 'wOBACON_hr', 'wOBACON']
    if has_loc: drop_cols.append('pitching_lw_mean')
    agg.drop(columns=[c for c in drop_cols if c in agg.columns], inplace=True)
    agg.rename(columns={'pitch_type': 'PitchType'}, inplace=True)

    return agg, avg_fb_vel, avg_fb_vert, avg_fb_horz, avg_fb_spin


# ------------------------------------------------------------------
# Pitch-by-pitch scorer (returns one row per pitch with grades)
# ------------------------------------------------------------------
def compute_pitch_by_pitch(tm_df, pitcher_name, pitcher_is_righty):
    sub = tm_df[tm_df['pitcher'] == pitcher_name].copy()
    sub = sub[sub['pitch_type'].isin(VALID_PITCH_TYPES)].reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame()

    fb_sub = sub[sub['pitch_type'].isin(fb_types)]
    if fb_sub.empty: fb_sub = sub.head(1)
    avg_fb_vel  = float(fb_sub['rel_speed'].mean())
    avg_fb_vert = float(fb_sub['induced_vert'].mean())
    avg_fb_horz = float(fb_sub['horz_break'].mean())
    avg_fb_spin = float(fb_sub['spin_rate'].mean())

    feat = pd.DataFrame({
        'RelSpeed':          sub['rel_speed'],
        'InducedVertBreak':  sub['induced_vert'],
        'HorzBreak':         sub['horz_break'],
        'SpinRate':          sub['spin_rate'],
        'Extension':         sub['extension'],
        'RelHeight':         sub['rel_height'],
        'RelSide':           sub['rel_side'],
        'TaggedPitchTypeNum':sub['pitch_type'].map(pitch_type_map),
        'pitcher_is_righty': pitcher_is_righty,
        'diff_fb_vel':       avg_fb_vel  - sub['rel_speed'],
        'diff_fb_vert':      avg_fb_vert - sub['induced_vert'],
        'diff_fb_horz':      avg_fb_horz - sub['horz_break'],
        'diff_spinrate':     avg_fb_spin - sub['spin_rate'],
    })

    for label in labels:
        feat[f'sp_{label}'] = event_models[label].predict(
            feat[stuff_features].values, num_iteration=event_models[label].best_iteration)

    spf = [c for c in feat.columns if c.startswith('sp_')]
    feat['stuff_lw']  = final_model.predict(feat[spf].values)
    feat['StuffPlus'] = 100 + (norm_params['mu_stuff'] - feat['stuff_lw']) / norm_params['sigma_stuff'] * 10

    # PitchingPlus per pitch (requires location)
    feat['PitchingPlus'] = np.nan
    has_loc = (
        'plate_loc_side' in sub.columns and
        'plate_loc_height' in sub.columns and
        sub['plate_loc_side'].notna().any()
    )
    if has_loc:
        loc_mask = sub['plate_loc_side'].notna() & sub['plate_loc_height'].notna()
        if loc_mask.any():
            pfeat = pd.DataFrame({
                'RelSpeed':          sub.loc[loc_mask, 'rel_speed'],
                'InducedVertBreak':  sub.loc[loc_mask, 'induced_vert'],
                'HorzBreak':         sub.loc[loc_mask, 'horz_break'],
                'SpinRate':          sub.loc[loc_mask, 'spin_rate'],
                'Extension':         sub.loc[loc_mask, 'extension'],
                'RelHeight':         sub.loc[loc_mask, 'rel_height'],
                'RelSide':           sub.loc[loc_mask, 'rel_side'],
                'TaggedPitchTypeNum':sub.loc[loc_mask, 'pitch_type'].map(pitch_type_map),
                'pitcher_is_righty': pitcher_is_righty,
                'diff_fb_vel':       avg_fb_vel  - sub.loc[loc_mask, 'rel_speed'],
                'diff_fb_vert':      avg_fb_vert - sub.loc[loc_mask, 'induced_vert'],
                'diff_fb_horz':      avg_fb_horz - sub.loc[loc_mask, 'horz_break'],
                'diff_spinrate':     avg_fb_spin - sub.loc[loc_mask, 'spin_rate'],
                'PlateLocSide':      sub.loc[loc_mask, 'plate_loc_side'],
                'PlateLocHeight':    sub.loc[loc_mask, 'plate_loc_height'],
            }).reset_index(drop=True)
            pc = []
            for label in labels:
                c = f'pp_{label}'
                pfeat[c] = pitching_event_models[label].predict(
                    pfeat[pitching_features].values,
                    num_iteration=pitching_event_models[label].best_iteration)
                pc.append(c)
            pfeat['lw'] = pitching_final_model.predict(pfeat[pc].values)
            pfeat['PitchingPlus'] = 100 + (norm_params['mu_pitching'] - pfeat['lw']) / norm_params['sigma_pitching'] * 10
            feat.loc[loc_mask, 'PitchingPlus'] = pfeat['PitchingPlus'].values

    # Assemble output frame — preserve original row order
    # '#' = pitch count for THIS pitcher only (1, 2, 3 … n)
    out = pd.DataFrame()
    out['#']           = range(1, len(sub) + 1)
    out['Pitch Type']  = sub['pitch_type'].values
    out['Velo']        = sub['rel_speed'].round(1).values
    out['IVB']         = sub['induced_vert'].round(1).values
    out['HB']          = sub['horz_break'].round(1).values
    out['Spin']        = sub['spin_rate'].round(0).values
    out['Stuff+']      = feat['StuffPlus'].round(1).values
    out['Pitching+']   = feat['PitchingPlus'].round(1).values

    # Count context
    if 'count_balls' in sub.columns and sub['count_balls'].notna().any():
        out['Count'] = (
            sub['count_balls'].fillna(0).astype(int).astype(str) + '-' +
            sub['count_strikes'].fillna(0).astype(int).astype(str)
        ).values
    else:
        out['Count'] = ''

    # Inning
    if 'inning' in sub.columns and sub['inning'].notna().any():
        out['Inn'] = sub['inning'].values
    else:
        out['Inn'] = ''

    # Batter
    if 'batter' in sub.columns and sub['batter'].notna().any():
        out['Batter'] = sub['batter'].values
    else:
        out['Batter'] = ''

    # Pitch call / result resolution:
    # PlayResult is more specific than PitchCall (it distinguishes Single/Double/Out/etc.),
    # so prefer it when it has a meaningful value. Fall back to PitchCall otherwise.
    # Also handle common TrackMan quirks: "Undefined" for non-InPlay pitches, NaN PitchCall
    # on rows that still have a valid PlayResult (happens on some exports).
    has_play_result = ('play_result' in sub.columns and sub['play_result'].notna().any())
    has_pitch_call  = ('pitch_call'  in sub.columns and sub['pitch_call'].notna().any())

    if has_pitch_call or has_play_result:
        NON_MEANINGFUL = {'undefined', 'nan', 'none', ''}

        def resolve_result(row):
            # Step 1: check PlayResult first (most specific)
            if has_play_result:
                pr = row.get('play_result', '')
                if pd.notna(pr):
                    pr_str = str(pr).strip()
                    if pr_str and pr_str.lower() not in NON_MEANINGFUL:
                        return pr_str
            # Step 2: fall back to PitchCall
            if has_pitch_call:
                call = row.get('pitch_call', '')
                if pd.notna(call):
                    call_str = str(call).strip()
                    if call_str and call_str.lower() not in NON_MEANINGFUL:
                        return call_str
            return ''
        out['Result'] = sub.apply(resolve_result, axis=1).values
    else:
        out['Result'] = ''

    return out.reset_index(drop=True)


def build_leaderboard(tm_df):
    has_loc = (
        'plate_loc_side' in tm_df.columns and
        'plate_loc_height' in tm_df.columns and
        tm_df['plate_loc_side'].notna().any()
    )

    rows = []
    for pitcher in tm_df['pitcher'].dropna().unique():
        sub = tm_df[tm_df['pitcher'] == pitcher]
        sub = sub[sub['pitch_type'].isin(VALID_PITCH_TYPES)].reset_index(drop=True)
        if sub.empty: continue

        hand = sub['pitcher_hand'].mode()[0] if 'pitcher_hand' in sub.columns else 'Right'
        is_righty = 1 if hand == 'Right' else 0

        fb_sub = sub[sub['pitch_type'].isin(fb_types)]
        if fb_sub.empty: fb_sub = sub.head(1)
        afv  = float(fb_sub['rel_speed'].mean())
        afvt = float(fb_sub['induced_vert'].mean())
        afh  = float(fb_sub['horz_break'].mean())
        afs  = float(fb_sub['spin_rate'].mean())

        feat = pd.DataFrame({
            'RelSpeed':          sub['rel_speed'],
            'InducedVertBreak':  sub['induced_vert'],
            'HorzBreak':         sub['horz_break'],
            'SpinRate':          sub['spin_rate'],
            'Extension':         sub['extension'],
            'RelHeight':         sub['rel_height'],
            'RelSide':           sub['rel_side'],
            'TaggedPitchTypeNum':sub['pitch_type'].map(pitch_type_map),
            'pitcher_is_righty': is_righty,
            'diff_fb_vel':       afv  - sub['rel_speed'],
            'diff_fb_vert':      afvt - sub['induced_vert'],
            'diff_fb_horz':      afh  - sub['horz_break'],
            'diff_spinrate':     afs  - sub['spin_rate'],
        }).reset_index(drop=True)

        for label in labels:
            feat[f'sp_{label}'] = event_models[label].predict(
                feat[stuff_features].values, num_iteration=event_models[label].best_iteration)
        spf = [c for c in feat.columns if c.startswith('sp_')]
        # Average raw lw across all pitches first, then apply mu/sigma once
        feat['stuff_lw'] = final_model.predict(feat[spf].values)
        avg_stuff_lw = float(feat['stuff_lw'].mean())
        stuff_plus_val = 100 + (norm_params['mu_stuff'] - avg_stuff_lw) / norm_params['sigma_stuff'] * 10

        pp_val = np.nan
        if has_loc:
            sub_loc = sub.dropna(subset=['plate_loc_side', 'plate_loc_height']).reset_index(drop=True)
            if len(sub_loc) >= 3:
                pf = pd.DataFrame({
                    'RelSpeed':          sub_loc['rel_speed'],
                    'InducedVertBreak':  sub_loc['induced_vert'],
                    'HorzBreak':         sub_loc['horz_break'],
                    'SpinRate':          sub_loc['spin_rate'],
                    'Extension':         sub_loc['extension'],
                    'RelHeight':         sub_loc['rel_height'],
                    'RelSide':           sub_loc['rel_side'],
                    'TaggedPitchTypeNum':sub_loc['pitch_type'].map(pitch_type_map),
                    'pitcher_is_righty': is_righty,
                    'diff_fb_vel':       afv  - sub_loc['rel_speed'],
                    'diff_fb_vert':      afvt - sub_loc['induced_vert'],
                    'diff_fb_horz':      afh  - sub_loc['horz_break'],
                    'diff_spinrate':     afs  - sub_loc['spin_rate'],
                    'PlateLocSide':      sub_loc['plate_loc_side'],
                    'PlateLocHeight':    sub_loc['plate_loc_height'],
                }).reset_index(drop=True)
                pc = []
                for label in labels:
                    c = f'pp_{label}'
                    pf[c] = pitching_event_models[label].predict(
                        pf[pitching_features].values,
                        num_iteration=pitching_event_models[label].best_iteration)
                    pc.append(c)
                pf['lw'] = pitching_final_model.predict(pf[pc].values)
                # Average raw lw first, then convert to +scale once
                avg_pitching_lw = float(pf['lw'].mean())
                pp_val = 100 + (norm_params['mu_pitching'] - avg_pitching_lw) / norm_params['sigma_pitching'] * 10

        row = {
            'Pitcher':  pitcher,
            'Hand':     'R' if is_righty else 'L',
            'Pitches':  len(sub),
            'Max Velo': fmt(float(sub['rel_speed'].max())),
            'Stuff+':   fmt(stuff_plus_val),
        }
        if has_loc:
            row['Pitching+'] = fmt(pp_val) if not np.isnan(pp_val) else np.nan
        rows.append(row)

    sort_col = 'Pitching+' if has_loc and any(r.get('Pitching+') is not None for r in rows) else 'Stuff+'
    lb = pd.DataFrame(rows).sort_values(sort_col, ascending=False).reset_index(drop=True)
    lb.insert(0, 'Rank', range(1, len(lb)+1))
    return lb, sort_col


@st.cache_data
def load_season_summary():
    try:
        return pd.read_csv("pitcher_overall_summary_2025.csv")
    except FileNotFoundError:
        return None


# ------------------------------------------------------------------
# Grade color helpers
# ------------------------------------------------------------------
def color_grade(val):
    try:
        v = float(val)
        if v >= 110: return 'background-color: #1B5E20; color: white; font-weight: bold'
        if v >= 105: return 'background-color: #388E3C; color: white'
        if v >= 100: return 'background-color: #81C784; color: black'
        if v >= 95:  return 'background-color: #FDD835; color: black'
        if v >= 90:  return 'background-color: #FFB74D; color: black'
        return              'background-color: #E53935; color: white'
    except:
        return ''


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚾ Stuff+ Builder")
    input_mode = st.radio(
        "Mode",
        ["Arsenal Builder", "TrackMan Analyzer", "2025 Season"],
        index=["Arsenal Builder", "TrackMan Analyzer", "2025 Season"].index(
            st.session_state.input_mode
            if st.session_state.input_mode in ["Arsenal Builder", "TrackMan Analyzer", "2025 Season"]
            else "Arsenal Builder"
        ),
    )
    st.session_state.input_mode = input_mode
    st.divider()

    if input_mode == "2025 Season":
        st.caption("Full 2025 season pitcher leaderboard.")

    elif input_mode == "Arsenal Builder":
        pitcher_name      = st.text_input("Name", value="My Arsenal", label_visibility="collapsed", placeholder="Pitcher name")
        pitcher_hand      = st.radio("Throws", ["Right", "Left"], horizontal=True)
        pitcher_is_righty = 1 if pitcher_hand == "Right" else 0
        if st.session_state.pitches:
            st.caption(f"{len(st.session_state.pitches)} pitch(es) in arsenal")

    elif input_mode == "TrackMan Analyzer":
        st.markdown("#### Upload TrackMan CSV")
        uploaded_file = st.file_uploader("", type=["csv"], key="tm_upload", label_visibility="collapsed")
        if uploaded_file:
            with st.spinner("Reading file..."):
                tm_data, err = parse_trackman_csv(uploaded_file)
            if err:
                st.error(f"Could not parse file: {err}")
                st.session_state.tm_data = None
            else:
                st.session_state.tm_data = tm_data
                pitchers = sorted(tm_data["pitcher"].unique())
                st.success(f"{len(tm_data):,} pitches · {len(pitchers)} pitcher(s) found")

        if st.session_state.tm_data is not None:
            tm_data = st.session_state.tm_data
            pitchers = sorted(tm_data["pitcher"].unique())
            selected_pitcher = st.selectbox("Select Pitcher", pitchers)
        else:
            selected_pitcher = None

    st.divider()
    st.caption("100 = college baseball average  |  110+ = elite  |  <90 = below avg")
    st.caption("Horizontal break: + = arm side  |  − = glove side")
    st.caption("All charts shown from pitcher's view")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
st.title("Stuff+ Arsenal Builder")

# =========================================================
# 2025 SEASON MODE
# =========================================================
if st.session_state.input_mode == "2025 Season":
    season_df = load_season_summary()
    if season_df is None:
        st.warning("⚠️  `pitcher_overall_summary_2025.csv` not found.")
    else:
        has_pitching = (
            "Overall_PitchingPlus" in season_df.columns and
            season_df["Overall_PitchingPlus"].notna().any()
        )
        disp_cols = ["Pitcher", "PitcherTeam", "Total_Pitches", "Overall_StuffPlus"]
        if has_pitching: disp_cols.append("Overall_PitchingPlus")
        disp_s = season_df[disp_cols].copy()
        rename_map = {"Pitcher": "Pitcher", "PitcherTeam": "Team",
                      "Total_Pitches": "Pitches", "Overall_StuffPlus": "Stuff+"}
        if has_pitching: rename_map["Overall_PitchingPlus"] = "Pitching+"
        disp_s = disp_s.rename(columns=rename_map)
        if "PitcherThrows" in season_df.columns:
            disp_s.insert(1, "Hand", season_df["PitcherThrows"].map(
                lambda x: "R" if str(x).strip() in ["Right", "R", "RHP"] else "L"))

        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])
        with fc1: sort_by = st.selectbox("Sort by", ["Stuff+"] + (["Pitching+"] if has_pitching else []), key="season_sort")
        with fc2: selected_team = st.selectbox("Team", ["All Teams"] + sorted(disp_s["Team"].dropna().unique().tolist()), key="season_team")
        with fc3: name_search = st.text_input("Search pitcher name", value="", placeholder="e.g. Smith", key="season_name")
        with fc4: min_pitches = st.number_input("Min pitches", min_value=1, value=50, step=10, key="season_min_p")

        disp_s = disp_s[disp_s["Pitches"] >= min_pitches]
        if selected_team != "All Teams": disp_s = disp_s[disp_s["Team"] == selected_team]
        if name_search.strip(): disp_s = disp_s[disp_s["Pitcher"].str.contains(name_search.strip(), case=False, na=False)]
        disp_s = disp_s.sort_values(sort_by, ascending=False).reset_index(drop=True)
        disp_s.insert(0, "Rank", range(1, len(disp_s)+1))
        for col in ["Stuff+", "Pitching+"]:
            if col in disp_s.columns: disp_s[col] = disp_s[col].round(1)

        if not disp_s.empty:
            w = disp_s["Pitches"].values
            wa_s = float(np.average(disp_s["Stuff+"].values, weights=w))
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Avg Stuff+ (weighted)", f"{wa_s:.1f}", f"{wa_s-100:+.1f}")
            if "Pitching+" in disp_s.columns and disp_s["Pitching+"].notna().any():
                pm = disp_s["Pitching+"].notna()
                wa_p = float(np.average(disp_s.loc[pm, "Pitching+"].values, weights=disp_s.loc[pm, "Pitches"].values))
                sm2.metric("Avg Pitching+ (weighted)", f"{wa_p:.1f}", f"{wa_p-100:+.1f}")
            sm3.metric("Total Pitches", f"{int(disp_s['Pitches'].sum()):,}")

        grade_cols_s = [c for c in ["Stuff+", "Pitching+"] if c in disp_s.columns]
        st.markdown(f"**{len(disp_s)} pitchers** · min {min_pitches} pitches · sorted by {sort_by}")
        st.dataframe(disp_s.style.applymap(color_grade, subset=grade_cols_s), use_container_width=True, hide_index=True)
        st.caption("100 = college average  ·  110+ = elite  ·  <90 = below average")
        st.download_button("⬇️  Download CSV", disp_s.to_csv(index=False).encode(), "pitcher_summary_2025_export.csv", "text/csv")
    st.stop()

# =========================================================
# ARSENAL BUILDER MODE
# =========================================================
if st.session_state.input_mode == "Arsenal Builder":
    pitch_type = st.selectbox("Pitch Type", VALID_PITCH_TYPES)
    defaults   = PITCH_DEFAULTS[pitch_type]
    default_hb = defaults["hb"] if pitcher_is_righty else -defaults["hb"]

    col1, col2, col3, col4 = st.columns(4)
    with col1: rel_speed    = st.slider("Velocity (mph)",    65.0, 105.0, float(defaults["velo"]), 0.5)
    with col2: induced_vert = st.slider("Induced Vert (in)", -20.0, 25.0, float(defaults["ivb"]),  0.5)
    with col3: horz_break   = st.slider("Horz Break (in)",  -25.0, 25.0, float(default_hb),        0.5)
    with col4: spin_rate    = st.slider("Spin Rate (rpm)",   1000,  3500, int(defaults["spin"]),    50)

    with st.expander("Release Settings"):
        c1, c2, c3 = st.columns(3)
        with c1: extension  = st.slider("Extension (ft)",      4.0, 7.5, 6.0, 0.1)
        with c2: rel_height = st.slider("Release Height (ft)", 4.0, 7.0, 5.5, 0.1)
        with c3: rel_side   = st.slider("Release Side (ft)", -3.0, 3.0, 2.0 if pitcher_is_righty else -2.0, 0.1)

    if st.button("➕ Add Pitch", use_container_width=True, type="primary"):
        st.session_state.pitches.append({
            'pitch_type': pitch_type, 'rel_speed': rel_speed, 'induced_vert': induced_vert,
            'horz_break': horz_break, 'spin_rate': spin_rate, 'extension': extension,
            'rel_height': rel_height, 'rel_side': rel_side,
        })
        st.toast(f"✅ {pitch_type} added to arsenal")

    pitches_to_show           = st.session_state.pitches
    pitcher_is_righty_display = pitcher_is_righty

# =========================================================
# TRACKMAN MODE
# =========================================================
else:
    if st.session_state.tm_data is None or selected_pitcher is None:
        st.info("👈  Upload a TrackMan CSV using the sidebar to get started.")
        st.stop()

    tm_data = st.session_state.tm_data
    pitches_list, pitcher_hand = tm_to_pitches(tm_data, selected_pitcher)
    pitcher_is_righty_display  = 1 if pitcher_hand == 'Right' else 0

    if not pitches_list:
        st.warning(f"No valid pitches found for {selected_pitcher}")
        st.stop()

    pitcher_name    = selected_pitcher
    pitches_to_show = pitches_list

    sub = tm_data[tm_data['pitcher'] == selected_pitcher]
    with st.expander(f"📋  Raw Pitch Data — {selected_pitcher} ({len(sub)} pitches)", expanded=False):
        pt_counts = sub['pitch_type'].value_counts().reset_index()
        pt_counts.columns = ['Pitch Type', 'Count']
        pt_counts['%'] = (pt_counts['Count'] / pt_counts['Count'].sum() * 100).round(1)
        st.dataframe(pt_counts, use_container_width=True, hide_index=True)

        has_loc = ('plate_loc_side' in sub.columns) and sub['plate_loc_side'].notna().any()
        if has_loc:
            st.markdown("**Pitch Locations**")
            fig_raw = go.Figure()
            for pt in sub['pitch_type'].dropna().unique():
                grp = sub[sub['pitch_type'] == pt].dropna(subset=['plate_loc_side', 'plate_loc_height'])
                if grp.empty: continue
                fig_raw.add_trace(go.Scatter(
                    x=grp['plate_loc_side'], y=grp['plate_loc_height'],
                    mode='markers', name=pt,
                    marker=dict(size=6, color=PITCH_COLORS.get(pt, 'white'), opacity=0.7),
                    hovertemplate=f"{pt}<br>Side: %{{x:.2f}}<br>Height: %{{y:.2f}}<extra></extra>"
                ))
            fig_raw.add_trace(go.Scatter(
                x=[-0.708, 0.708, 0.708, -0.708, -0.708], y=[1.5, 1.5, 3.5, 3.5, 1.5],
                mode='lines', line=dict(color='white', width=2), hoverinfo='skip', showlegend=False))
            fig_raw.update_layout(
                xaxis=dict(title="Plate Side (ft)", range=[-3.5, 3.5]),
                yaxis=dict(title="Height (ft)", range=[-0.5, 6.0]),
                height=450, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                font=dict(color='white'), legend=dict(font=dict(color='white')))
            st.plotly_chart(fig_raw, use_container_width=True)


# ------------------------------------------------------------------
# Leaderboard (TrackMan mode only)
# ------------------------------------------------------------------
if st.session_state.input_mode == "TrackMan Analyzer" and st.session_state.tm_data is not None:
    with st.expander("🏆  Pitcher Leaderboard", expanded=True):
        with st.spinner("Scoring all pitchers..."):
            lb, sort_col = build_leaderboard(st.session_state.tm_data)
        if lb.empty:
            st.info("No scorable pitchers found.")
        else:
            for col in ['Max Velo', 'Stuff+', 'Pitching+']:
                if col in lb.columns:
                    lb[col] = lb[col].apply(lambda x: str(fmt(x)).rstrip('0').rstrip('.') if pd.notna(x) else x)
            grade_cols_lb = [c for c in ['Stuff+', 'Pitching+'] if c in lb.columns]
            st.dataframe(lb.style.applymap(color_grade, subset=grade_cols_lb), use_container_width=True, hide_index=True)
            has_pp_lb = 'Pitching+' in lb.columns
            st.caption(f"Ranked by {sort_col}  ·  100 = college baseball average"
                       + ("" if has_pp_lb else "  ·  Upload file with PlateLocSide/Height for Pitching+"))


# ------------------------------------------------------------------
# Arsenal display (shared between modes)
# ------------------------------------------------------------------
if not pitches_to_show:
    if st.session_state.input_mode == "Arsenal Builder":
        st.info("👆  Select a pitch type and adjust the sliders above, then click **Add Pitch** to begin.")
    st.stop()

st.divider()
st.subheader(f"{pitcher_name}  —  {'RHP' if pitcher_is_righty_display else 'LHP'}")

if st.session_state.input_mode == "TrackMan Analyzer":
    tm_scores, avg_fb_vel, avg_fb_vert, avg_fb_horz, avg_fb_spin = compute_stuff_scores_from_tm(
        st.session_state.tm_data, pitcher_name, pitcher_is_righty_display)
    chars = pd.DataFrame([{
        'PitchType': p['pitch_type'], 'RelSpeed': p['rel_speed'],
        'InducedVertBreak': p['induced_vert'], 'HorzBreak': p['horz_break'],
        'SpinRate': p['spin_rate'],
        'Extension': p.get('extension'), 'RelHeight': p.get('rel_height'), 'RelSide': p.get('rel_side'),
        'SpinEfficiency': p.get('spin_efficiency'), 'SpinAxis': p.get('spin_axis'),
    } for p in pitches_to_show])
    df = pd.merge(chars, tm_scores, on='PitchType', how='left')
    df = df.set_index('PitchType').loc[[p['pitch_type'] for p in pitches_to_show]].reset_index()
else:
    df, avg_fb_vel, avg_fb_vert, avg_fb_horz, avg_fb_spin = compute_stuff_scores(
        pitches_to_show, pitcher_is_righty_display)

# Summary metrics
if st.session_state.input_mode == "TrackMan Analyzer" and 'count' in df.columns:
    w = df['count'].fillna(1).values
    def wavg(col): return float(np.average(df[col].fillna(0), weights=w))
    m1, m2, m3 = st.columns(3)
    sv = wavg('StuffPlus')
    m1.metric("Stuff+", f"{sv:.0f}", f"{sv-100:+.0f}")
    if 'PitchingPlus' in df.columns and df['PitchingPlus'].notna().any():
        pv = wavg('PitchingPlus')
        m2.metric("Pitching+", f"{pv:.0f}", f"{pv-100:+.0f}")
    else:
        m2.metric("Pitching+", "N/A", help="Requires plate location data")
    max_velo = float(st.session_state.tm_data[st.session_state.tm_data['pitcher'] == pitcher_name]['rel_speed'].max())
    m3.metric("Max Velo", f"{max_velo:.1f} mph")
else:
    def wavg(col): return float(df[col].mean())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stuff+",        f"{wavg('StuffPlus'):.0f}",        f"{wavg('StuffPlus')-100:+.0f}")
    m2.metric("Whiff+",        f"{wavg('WhiffPlus'):.0f}",        f"{wavg('WhiffPlus')-100:+.0f}")
    m3.metric("CalledStrike+", f"{wavg('CalledStrikePlus'):.0f}", f"{wavg('CalledStrikePlus')-100:+.0f}")
    m4.metric("wOBACON+",      f"{wavg('wOBACONPlus'):.0f}",      f"{wavg('wOBACONPlus')-100:+.0f}")

# Tab nav
tab_options = ["📊 Grades", "🎯 Movement", "🟩 Zone"]
if st.session_state.input_mode == "TrackMan Analyzer":
    tab_options += ["⚾ Pitch Plot", "📋 Pitch by Pitch"]

if st.session_state.active_tab >= len(tab_options):
    st.session_state.active_tab = 0

selected_tab = st.radio(
    "View", tab_options,
    index=st.session_state.active_tab,
    horizontal=True, key="tab_radio", label_visibility="collapsed"
)
st.session_state.active_tab = tab_options.index(selected_tab)


# =====================================================================
# GRADES TAB
# =====================================================================
if selected_tab == "📊 Grades":
    has_pp_col = (
        st.session_state.input_mode == "TrackMan Analyzer" and
        'PitchingPlus' in df.columns and df['PitchingPlus'].notna().any()
    )
    disp_cols = ['PitchType', 'RelSpeed', 'InducedVertBreak', 'HorzBreak',
                 'StuffPlus', 'WhiffPlus', 'CalledStrikePlus', 'wOBACONPlus']
    if has_pp_col: disp_cols.insert(disp_cols.index('WhiffPlus'), 'PitchingPlus')
    disp = df[disp_cols].copy()
    base_cols = ['Pitch', 'Velo', 'IVB', 'HB', 'Stuff+', 'Whiff+', 'CalledStrike+', 'wOBACON+']
    if has_pp_col: base_cols.insert(base_cols.index('Whiff+'), 'Pitching+')
    disp.columns = base_cols

    if st.session_state.input_mode == "TrackMan Analyzer":
        disp['N'] = [p.get('count', 0) for p in pitches_to_show]
        disp = disp.sort_values('N', ascending=False).reset_index(drop=True)
        disp.insert(1, 'N', disp.pop('N'))

    for col in disp.columns:
        if col not in ('Pitch', 'N'):
            disp[col] = disp[col].apply(fmt).apply(str).str.replace(r'\.0$', '', regex=True)
    disp.insert(0, '#', range(1, len(disp)+1))

    grade_cols = [c for c in ['Stuff+', 'Pitching+', 'Whiff+', 'CalledStrike+', 'wOBACON+'] if c in disp.columns]
    st.dataframe(disp.style.applymap(color_grade, subset=grade_cols), use_container_width=True, hide_index=True)
    st.caption("100 = college average  ·  Higher is better  ·  Color scale: green = elite, red = below average")

    for idx, p in enumerate(pitches_to_show):
        label = f"#{idx+1} {p['pitch_type']} — {p['rel_speed']:.1f} mph"
        if 'count' in p: label += f" ({p['count']} pitches)"
        # Pull grades from df (original pitches_to_show order) — NOT disp, which
        # is sorted by count in TrackMan mode and would mismatch the pitch.
        row = df.iloc[idx]
        with st.expander(label):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Pitch Characteristics**")
                st.write(f"Velo: {p['rel_speed']:.1f} mph")
                st.write(f"IVB: {p['induced_vert']:+.1f}\"")
                st.write(f"HB: {p['horz_break']:+.1f}\"")
                st.write(f"Spin: {p['spin_rate']:.0f} rpm")
            with c2:
                st.markdown("**Grades**")
                st.write(f"Stuff+: {row['StuffPlus']:.0f}")
                if 'PitchingPlus' in df.columns and pd.notna(row.get('PitchingPlus')):
                    st.write(f"Pitching+: {row['PitchingPlus']:.0f}")
                st.write(f"Whiff+: {row['WhiffPlus']:.0f}")
                st.write(f"CalledStrike+: {row['CalledStrikePlus']:.0f}")
                st.write(f"wOBACON+: {row['wOBACONPlus']:.0f}")
            with c3:
                st.markdown("**Estimated Probabilities**")
                st.write(f"Whiff: {row['sp_StrikeSwinging']*100:.1f}%")
                st.write(f"Called Strike: {row['sp_StrikeCalled']*100:.1f}%")
                st.write(f"Home Run: {row['sp_HomeRun']*100:.1f}%")
            if st.session_state.input_mode == "Arsenal Builder":
                if st.button(f"Remove #{idx+1}", key=f"del_{idx}"):
                    st.session_state.pitches.pop(idx)
                    st.rerun()


# =====================================================================
# MOVEMENT TAB
# =====================================================================
elif selected_tab == "🎯 Movement":
    st.caption("Movement plot — pitcher's view  ·  Dot color shows Stuff+ grade")
    fig = go.Figure()

    if st.session_state.input_mode == "TrackMan Analyzer":
        show_scatter = st.checkbox("Show individual pitch locations", value=False)
    else:
        show_scatter = False

    if show_scatter and st.session_state.tm_data is not None:
        sub_mv = tm_data[tm_data['pitcher'] == selected_pitcher]
        for pt in sub_mv['pitch_type'].dropna().unique():
            grp = sub_mv[sub_mv['pitch_type'] == pt].dropna(subset=['horz_break', 'induced_vert'])
            if grp.empty: continue
            pc = PITCH_COLORS.get(pt, '#ffffff')
            r, g, b = int(pc[1:3], 16), int(pc[3:5], 16), int(pc[5:7], 16)
            fig.add_trace(go.Scatter(
                x=grp['horz_break'], y=grp['induced_vert'],
                mode='markers', name=f"{pt} (raw)",
                marker=dict(size=5, color=f'rgba({r},{g},{b},0.15)'),
                hoverinfo='skip', showlegend=False))

    for idx, p in enumerate(pitches_to_show):
        sv    = float(df.iloc[idx]['StuffPlus'])
        col   = '#1B5E20' if sv >= 105 else '#81C784' if sv >= 95 else '#FFB74D'
        label = p['pitch_type'] + (f" ({p['count']})" if 'count' in p else "")
        fig.add_trace(go.Scatter(
            x=[p['horz_break']], y=[p['induced_vert']],
            mode='markers+text',
            text=[f"{label}<br>{sv:.0f}"],
            textposition="top center",
            marker=dict(size=15, color=col, line=dict(width=2, color='white')),
            hovertemplate=f"<b>{p['pitch_type']}</b><br>{p['rel_speed']:.1f} mph<br>Stuff+: {sv:.0f}<extra></extra>"
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
    fig.update_layout(
        xaxis_title="Horizontal Break — inches (pitcher's view)",
        yaxis_title="Induced Vertical Break (inches)",
        xaxis=dict(range=[-25, 25]), yaxis=dict(range=[-25, 25]),
        height=580, showlegend=False,
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# ZONE HEATMAP TAB
# =====================================================================
elif selected_tab == "🟩 Zone":
    st.caption("Where each pitch plays best in the zone — pitcher's view  ·  Green = better location, Red = worse")

    pitch_labels = [
        f"#{i+1} {p['pitch_type']} ({p['rel_speed']:.1f} mph)" + (f" [{p['count']}]" if 'count' in p else "")
        for i, p in enumerate(pitches_to_show)
    ]
    zc1, zc2 = st.columns([2, 1])
    with zc1: selected_label = st.selectbox("Pitch", pitch_labels, key="zone_pitch_select")
    with zc2: zone_metric    = st.selectbox("Metric", ["Pitching+", "Whiff+", "CalledStrike+"], key="zone_metric_select")

    sel_idx        = pitch_labels.index(selected_label)
    selected_pitch = pitches_to_show[sel_idx]

    with st.spinner("Computing..."):
        side_vals, height_vals, grids = compute_zone_heatmap(
            pitch_row=selected_pitch, avg_fb_vel=avg_fb_vel, avg_fb_vert=avg_fb_vert,
            avg_fb_horz=avg_fb_horz, avg_fb_spin=avg_fb_spin,
            pitcher_is_righty=pitcher_is_righty_display,
            _pitching_event_models=pitching_event_models, _pitching_final_model=pitching_final_model,
            norm_params=norm_params, pitching_features=pitching_features,
            pitch_type_map=pitch_type_map, labels=labels, grid_n=25)

    grid     = grids[zone_metric]
    peak_idx = np.unravel_index(np.argmax(grid), grid.shape)
    peak_s, peak_h, peak_v = side_vals[peak_idx[1]], height_vals[peak_idx[0]], grid[peak_idx]

    hand_str  = "Right" if pitcher_is_righty_display else "Left"
    pitch_str = selected_pitch['pitch_type']
    sd_row    = location_sd_params[(location_sd_params['PitcherThrows'] == hand_str) &
                                    (location_sd_params['TaggedPitchType'] == pitch_str)]
    if len(sd_row) == 0: sd_row = location_sd_params[location_sd_params['PitcherThrows'] == hand_str]
    sd_side   = float(sd_row['avg_SD_PlateLocSide'].mean())   if len(sd_row) > 0 else 0.25
    sd_height = float(sd_row['avg_SD_PlateLocHeight'].mean()) if len(sd_row) > 0 else 0.25

    if st.session_state.input_mode == "TrackMan Analyzer" and st.session_state.tm_data is not None:
        sub_pt = tm_data[(tm_data['pitcher'] == selected_pitcher) & (tm_data['pitch_type'] == pitch_str)].dropna(subset=['plate_loc_side', 'plate_loc_height'])
        if len(sub_pt) >= 5:
            sd_sa, sd_ha = float(sub_pt['plate_loc_side'].std()), float(sub_pt['plate_loc_height'].std())
            if st.checkbox(f"Use actual command SDs (±{sd_sa:.3f} / ±{sd_ha:.3f} ft)", value=False, key="use_actual_sd"):
                sd_side, sd_height = sd_sa, sd_ha

    exp_grid, best_aim_s, best_aim_h, best_exp, prob_fav = compute_optimal_zone(
        grid, side_vals, height_vals, sd_side, sd_height, grid_n=25)

    st.caption(f"Command scatter: ±{sd_side:.2f} ft horizontal  ·  ±{sd_height:.2f} ft vertical  ({hand_str}HP {pitch_str})")
    view_mode    = st.radio("Map view", ["Raw heatmap", "Expected value (accounts for command scatter)"], horizontal=True, key="zone_view_mode")
    display_grid = grid if view_mode == "Raw heatmap" else exp_grid
    cb_title     = zone_metric if view_mode == "Raw heatmap" else f"Expected {zone_metric}"

    fig_z = go.Figure()
    fig_z.add_trace(go.Contour(x=side_vals, y=height_vals, z=display_grid, colorscale='RdYlGn',
        contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
        colorbar=dict(title=cb_title, thickness=15),
        hovertemplate=f"Side: %{{x:.2f}} ft<br>Height: %{{y:.2f}} ft<br>{cb_title}: %{{z:.1f}}<extra></extra>",
        line_smoothing=0.85))
    fig_z.add_trace(go.Scatter(x=[-0.708, 0.708, 0.708, -0.708, -0.708], y=[1.5, 1.5, 3.5, 3.5, 1.5],
        mode='lines', line=dict(color='white', width=2.5), hoverinfo='skip', showlegend=False))
    fig_z.add_trace(go.Scatter(x=[-0.708, 0.708, 0.708, 0.0, -0.708, -0.708], y=[0, 0, 0.15, 0.22, 0.15, 0],
        mode='lines', line=dict(color='lightgray', width=1.5), hoverinfo='skip', showlegend=False))
    if st.session_state.input_mode == "TrackMan Analyzer" and st.session_state.tm_data is not None:
        sub_pt2 = tm_data[(tm_data['pitcher'] == selected_pitcher) & (tm_data['pitch_type'] == pitch_str)].dropna(subset=['plate_loc_side', 'plate_loc_height'])
        if not sub_pt2.empty:
            fig_z.add_trace(go.Scatter(x=sub_pt2['plate_loc_side'], y=sub_pt2['plate_loc_height'],
                mode='markers', name='Actual', marker=dict(size=5, color='white', opacity=0.4, line=dict(color='gray', width=0.5)),
                hovertemplate="Actual<br>Side: %{x:.2f}<br>Height: %{y:.2f}<extra></extra>"))
    fig_z.add_shape(type='line', x0=best_aim_s, x1=best_aim_s, y0=best_aim_h-.35, y1=best_aim_h+.35, line=dict(color='cyan', width=2))
    fig_z.add_shape(type='line', x0=best_aim_s-.25, x1=best_aim_s+.25, y0=best_aim_h, y1=best_aim_h, line=dict(color='cyan', width=2))
    theta = np.linspace(0, 2*np.pi, 100)
    fig_z.add_trace(go.Scatter(x=best_aim_s+sd_side*np.cos(theta), y=best_aim_h+sd_height*np.sin(theta),
        mode='lines', line=dict(color='cyan', width=1.5, dash='dot'), hoverinfo='skip', name='1 SD'))
    fig_z.add_trace(go.Scatter(x=best_aim_s+2*sd_side*np.cos(theta), y=best_aim_h+2*sd_height*np.sin(theta),
        mode='lines', line=dict(color='cyan', width=1, dash='dash'), hoverinfo='skip', name='2 SD'))
    if view_mode == "Raw heatmap":
        fig_z.add_trace(go.Scatter(x=[peak_s], y=[peak_h], mode='markers+text',
            marker=dict(symbol='star', size=18, color='gold', line=dict(color='black', width=1)),
            text=[f"  {peak_v:.1f}"], textfont=dict(color='white', size=11), textposition='middle right',
            hovertemplate=f"Peak {zone_metric}: {peak_v:.1f}<extra></extra>", name='Peak'))
    fig_z.add_trace(go.Scatter(x=[best_aim_s], y=[best_aim_h], mode='markers',
        marker=dict(symbol='circle', size=10, color='cyan', line=dict(color='black', width=1.5)),
        hovertemplate=f"Optimal aim<br>Expected {zone_metric}: {best_exp:.1f}<br>P(Favorable): {prob_fav*100:.1f}%<extra></extra>",
        name='Optimal aim'))
    fig_z.update_layout(
        xaxis=dict(title="Horizontal (ft, pitcher's view)", range=[-3.5, 3.5], zeroline=False, tickformat='.1f'),
        yaxis=dict(title="Height (ft)", range=[-0.7, 6.0], zeroline=False, scaleanchor='x', scaleratio=1),
        height=600, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='white', size=10)),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'),
        title=dict(text=f"{selected_pitch['pitch_type']} — {cb_title} by Location", font=dict(size=14)))
    st.plotly_chart(fig_z, use_container_width=True)

    with st.expander("ℹ️  How to read this chart"):
        st.markdown(f"""
**Crosshair (cyan)** — optimal aim point that maximizes *expected* {zone_metric} accounting for command scatter.  
**Dotted ellipse** — 1 SD spread (~68% of pitches land inside).  **Dashed ellipse** — 2 SD spread (~95%).  
**Expected {zone_metric}: {best_exp:.1f}**  ·  **P(Favorable): {prob_fav*100:.1f}%** — probability of landing in a zone ≥ 100.
""")


# =====================================================================
# PITCH PLOT TAB
# =====================================================================
elif selected_tab == "⚾ Pitch Plot":
    if st.session_state.tm_data is None:
        st.info("No TrackMan data loaded.")
    else:
        tm_sub = st.session_state.tm_data[st.session_state.tm_data['pitcher'] == pitcher_name].copy()
        tm_sub = tm_sub[tm_sub['pitch_type'].isin(VALID_PITCH_TYPES)]
        has_loc_plot = ('plate_loc_side' in tm_sub.columns and 'plate_loc_height' in tm_sub.columns and tm_sub['plate_loc_side'].notna().any())
        if not has_loc_plot:
            st.warning("No plate location data found in this TrackMan file.")
        else:
            fb_sp = tm_sub[tm_sub['pitch_type'].isin(fb_types)]
            if fb_sp.empty: fb_sp = tm_sub.head(1)
            pfb_v  = float(fb_sp['rel_speed'].mean());  pfb_vt = float(fb_sp['induced_vert'].mean())
            pfb_h  = float(fb_sp['horz_break'].mean()); pfb_s  = float(fb_sp['spin_rate'].mean())

            pfeat = pd.DataFrame({
                'RelSpeed': tm_sub['rel_speed'], 'InducedVertBreak': tm_sub['induced_vert'],
                'HorzBreak': tm_sub['horz_break'], 'SpinRate': tm_sub['spin_rate'],
                'Extension': tm_sub['extension'], 'RelHeight': tm_sub['rel_height'],
                'RelSide': tm_sub['rel_side'],
                'TaggedPitchTypeNum': tm_sub['pitch_type'].map(pitch_type_map),
                'pitcher_is_righty': pitcher_is_righty_display,
                'diff_fb_vel':  pfb_v  - tm_sub['rel_speed'],  'diff_fb_vert': pfb_vt - tm_sub['induced_vert'],
                'diff_fb_horz': pfb_h  - tm_sub['horz_break'], 'diff_spinrate': pfb_s  - tm_sub['spin_rate'],
            }).reset_index(drop=True)

            for label in labels:
                pfeat[f'sp_{label}'] = event_models[label].predict(pfeat[stuff_features].values, num_iteration=event_models[label].best_iteration)
            spf2 = [c for c in pfeat.columns if c.startswith('sp_')]
            pfeat['stuff_lw']  = final_model.predict(pfeat[spf2].values)
            pfeat['StuffPlus'] = 100 + (norm_params['mu_stuff'] - pfeat['stuff_lw']) / norm_params['sigma_stuff'] * 10

            lm2 = tm_sub['plate_loc_side'].notna() & tm_sub['plate_loc_height'].notna()
            pfeat['PitchingPlus'] = np.nan
            if lm2.any():
                pp2 = pd.DataFrame({
                    'RelSpeed': tm_sub.loc[lm2, 'rel_speed'], 'InducedVertBreak': tm_sub.loc[lm2, 'induced_vert'],
                    'HorzBreak': tm_sub.loc[lm2, 'horz_break'], 'SpinRate': tm_sub.loc[lm2, 'spin_rate'],
                    'Extension': tm_sub.loc[lm2, 'extension'], 'RelHeight': tm_sub.loc[lm2, 'rel_height'],
                    'RelSide': tm_sub.loc[lm2, 'rel_side'],
                    'TaggedPitchTypeNum': tm_sub.loc[lm2, 'pitch_type'].map(pitch_type_map),
                    'pitcher_is_righty': pitcher_is_righty_display,
                    'diff_fb_vel':  pfb_v  - tm_sub.loc[lm2, 'rel_speed'],  'diff_fb_vert': pfb_vt - tm_sub.loc[lm2, 'induced_vert'],
                    'diff_fb_horz': pfb_h  - tm_sub.loc[lm2, 'horz_break'], 'diff_spinrate': pfb_s  - tm_sub.loc[lm2, 'spin_rate'],
                    'PlateLocSide': tm_sub.loc[lm2, 'plate_loc_side'], 'PlateLocHeight': tm_sub.loc[lm2, 'plate_loc_height'],
                }).reset_index(drop=True)
                pc2 = []
                for label in labels:
                    c = f'pp_{label}'
                    pp2[c] = pitching_event_models[label].predict(pp2[pitching_features].values, num_iteration=pitching_event_models[label].best_iteration)
                    pc2.append(c)
                pp2['lw'] = pitching_final_model.predict(pp2[pc2].values)
                pp2['PitchingPlus'] = 100 + (norm_params['mu_pitching'] - pp2['lw']) / norm_params['sigma_pitching'] * 10
                pfeat.loc[lm2.values[:len(pfeat)], 'PitchingPlus'] = pp2['PitchingPlus'].values

            pfeat['pitch_type']       = tm_sub['pitch_type'].values
            pfeat['plate_loc_side']   = tm_sub['plate_loc_side'].values
            pfeat['plate_loc_height'] = tm_sub['plate_loc_height'].values
            pfeat['rel_speed']        = tm_sub['rel_speed'].values
            pfeat['induced_vert']     = tm_sub['induced_vert'].values
            pfeat['horz_break']       = tm_sub['horz_break'].values
            pfeat['spin_rate']        = tm_sub['spin_rate'].values
            plot_df = pfeat.dropna(subset=['plate_loc_side', 'plate_loc_height']).copy()

            fc1, fc2 = st.columns([2, 1])
            with fc1:
                pt_options   = sorted(plot_df['pitch_type'].unique())
                selected_pts = st.multiselect("Show pitch types", pt_options, default=pt_options, key="pp_filter")
            with fc2:
                color_by = st.selectbox("Color by", ["Pitch Type", "Stuff+", "Pitching+", "Velo"], key="pp_color_by")

            plot_df = plot_df[plot_df['pitch_type'].isin(selected_pts)]
            fig_pp  = go.Figure()

            if color_by == "Pitch Type":
                for pt in selected_pts:
                    grp = plot_df[plot_df['pitch_type'] == pt]
                    if grp.empty: continue
                    fig_pp.add_trace(go.Scatter(
                        x=grp['plate_loc_side'], y=grp['plate_loc_height'],
                        mode='markers', name=pt,
                        marker=dict(size=10, color=PITCH_COLORS.get(pt, 'white'),
                                    line=dict(color='rgba(255,255,255,0.3)', width=0.5), opacity=0.85),
                        customdata=np.stack([grp['rel_speed'].round(1), grp['induced_vert'].round(1),
                            grp['horz_break'].round(1), grp['spin_rate'].round(0),
                            grp['StuffPlus'].round(1),
                            grp['PitchingPlus'].round(1) if grp['PitchingPlus'].notna().any() else np.full(len(grp), np.nan)], axis=-1),
                        hovertemplate=(f"<b>{pt}</b><br>Velo: %{{customdata[0]}} mph<br>"
                            "IVB: %{customdata[1]}in / HB: %{customdata[2]}in<br>Spin: %{customdata[3]} rpm<br>"
                            "Stuff+: %{customdata[4]}<br>Pitching+: %{customdata[5]}<br>"
                            "Side: %{x:.2f} ft · Height: %{y:.2f} ft<extra></extra>")))
            else:
                cc = {'Stuff+': 'StuffPlus', 'Pitching+': 'PitchingPlus', 'Velo': 'rel_speed'}[color_by]
                cs = 'RdYlGn' if color_by != 'Velo' else 'Plasma'
                fig_pp.add_trace(go.Scatter(
                    x=plot_df['plate_loc_side'], y=plot_df['plate_loc_height'],
                    mode='markers',
                    marker=dict(size=10, color=plot_df[cc], colorscale=cs,
                                cmin=plot_df[cc].quantile(0.05), cmax=plot_df[cc].quantile(0.95),
                                showscale=True, colorbar=dict(title=color_by, thickness=15),
                                line=dict(color='rgba(255,255,255,0.3)', width=0.5), opacity=0.9),
                    text=plot_df['pitch_type'],
                    customdata=np.stack([plot_df['rel_speed'].round(1), plot_df['induced_vert'].round(1),
                        plot_df['horz_break'].round(1), plot_df['spin_rate'].round(0),
                        plot_df['StuffPlus'].round(1),
                        plot_df['PitchingPlus'].round(1) if plot_df['PitchingPlus'].notna().any() else np.full(len(plot_df), np.nan)], axis=-1),
                    hovertemplate=("<b>%{text}</b><br>Velo: %{customdata[0]} mph<br>"
                        "IVB: %{customdata[1]}in / HB: %{customdata[2]}in<br>Spin: %{customdata[3]} rpm<br>"
                        "Stuff+: %{customdata[4]}<br>Pitching+: %{customdata[5]}<br>"
                        "Side: %{x:.2f} ft · Height: %{y:.2f} ft<extra></extra>"), showlegend=False))

            fig_pp.add_trace(go.Scatter(x=[-0.708, 0.708, 0.708, -0.708, -0.708], y=[1.5, 1.5, 3.5, 3.5, 1.5],
                mode='lines', line=dict(color='white', width=2.5), hoverinfo='skip', showlegend=False))
            for x in [-0.236, 0.236]:
                fig_pp.add_shape(type='line', x0=x, x1=x, y0=1.5, y1=3.5, line=dict(color='rgba(255,255,255,0.25)', width=1))
            for y in [2.167, 2.833]:
                fig_pp.add_shape(type='line', x0=-0.708, x1=0.708, y0=y, y1=y, line=dict(color='rgba(255,255,255,0.25)', width=1))
            fig_pp.add_trace(go.Scatter(x=[-0.708, 0.708, 0.708, 0.0, -0.708, -0.708], y=[0, 0, 0.15, 0.22, 0.15, 0],
                mode='lines', line=dict(color='lightgray', width=1.5), hoverinfo='skip', showlegend=False))
            fig_pp.update_layout(
                xaxis=dict(title="Horizontal (ft, pitcher's view)", range=[-3.0, 3.0], zeroline=False, tickformat='.1f'),
                yaxis=dict(title="Height (ft)", range=[-0.5, 5.5], zeroline=False, scaleanchor='x', scaleratio=1),
                height=620, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='white', size=11)),
                title=dict(text=f"{pitcher_name} — Pitch Locations ({len(plot_df)} pitches)", font=dict(size=14)))
            st.plotly_chart(fig_pp, use_container_width=True)
            st.caption("Pitcher's view  ·  Hover over any pitch for full details  ·  Strike zone divided into thirds")


# =====================================================================
# PITCH BY PITCH TAB  (TrackMan only)
# =====================================================================
elif selected_tab == "📋 Pitch by Pitch":
    if st.session_state.tm_data is None:
        st.info("No TrackMan data loaded.")
    else:
        with st.spinner("Scoring pitch-by-pitch..."):
            pbp = compute_pitch_by_pitch(
                st.session_state.tm_data, pitcher_name, pitcher_is_righty_display)

        if pbp.empty:
            st.warning("No scorable pitches found for this pitcher.")
        else:
            has_result  = 'Result' in pbp.columns and pbp['Result'].notna().any() and (pbp['Result'] != '').any()
            has_pp_pbp  = 'Pitching+' in pbp.columns and pbp['Pitching+'].notna().any()
            has_count   = 'Count' in pbp.columns and (pbp['Count'] != '').any()
            has_inning  = 'Inn' in pbp.columns and (pbp['Inn'] != '').any()
            has_batter  = 'Batter' in pbp.columns and (pbp['Batter'] != '').any()

            # ── Filters ──────────────────────────────────────────────
            f1, f2, f3 = st.columns(3)
            with f1:
                pt_filter = st.multiselect(
                    "Filter by pitch type",
                    options=sorted(pbp['Pitch Type'].unique()),
                    default=sorted(pbp['Pitch Type'].unique()),
                    key="pbp_pt_filter"
                )
            with f2:
                if has_result:
                    result_filter = st.multiselect(
                        "Filter by result",
                        options=sorted(pbp['Result'].dropna().unique()),
                        default=sorted(pbp['Result'].dropna().unique()),
                        key="pbp_result_filter"
                    )
                else:
                    result_filter = []
            with f3:
                sort_pbp = st.selectbox(
                    "Sort by",
                    ["Pitch # (chronological)", "Stuff+ (best first)", "Pitching+ (best first)", "Velo (fastest first)"],
                    key="pbp_sort"
                )

            # Apply filters
            view = pbp[pbp['Pitch Type'].isin(pt_filter)].copy()
            if has_result and result_filter:
                view = view[view['Result'].isin(result_filter)]

            # Apply sort
            if sort_pbp == "Stuff+ (best first)":
                view = view.sort_values('Stuff+', ascending=False)
            elif sort_pbp == "Pitching+ (best first)" and has_pp_pbp:
                view = view.sort_values('Pitching+', ascending=False, na_position='last')
            elif sort_pbp == "Velo (fastest first)":
                view = view.sort_values('Velo', ascending=False)
            else:
                view = view.sort_values('#')

            view = view.reset_index(drop=True)

            # ── Summary strip ─────────────────────────────────────────
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Pitches shown", len(view))
            s2.metric("Avg Stuff+", f"{view['Stuff+'].mean():.1f}")
            if has_pp_pbp and view['Pitching+'].notna().any():
                s3.metric("Avg Pitching+", f"{view['Pitching+'].mean():.1f}")
            s4.metric("Avg Velo", f"{view['Velo'].mean():.1f} mph")

            # ── Build display table ────────────────────────────────────
            # Choose which columns to show
            show_cols = ['#', 'Pitch Type', 'Velo', 'IVB', 'HB', 'Spin', 'Stuff+']
            if has_pp_pbp:  show_cols.append('Pitching+')
            if has_count:   show_cols.append('Count')
            if has_inning:  show_cols.append('Inn')
            if has_batter:  show_cols.append('Batter')
            if has_result:  show_cols.append('Result')

            disp_pbp = view[show_cols].copy()

            # Format numeric columns
            for col in ['Velo', 'IVB', 'HB']:
                disp_pbp[col] = disp_pbp[col].apply(lambda x: f"{x:+.1f}" if col in ('IVB', 'HB') else f"{x:.1f}")
            for col in ['Spin']:
                disp_pbp[col] = disp_pbp[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else '')
            for col in ['Stuff+', 'Pitching+']:
                if col in disp_pbp.columns:
                    disp_pbp[col] = disp_pbp[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '—')

            # Add emoji to result column
            if has_result:
                disp_pbp['Result'] = view['Result'].apply(
                    lambda x: f"{get_call_emoji(x)} {x}" if pd.notna(x) and x != '' else '')

            # Color grade columns
            grade_cols_pbp = [c for c in ['Stuff+', 'Pitching+'] if c in disp_pbp.columns]

            def color_grade_pbp(val):
                try:
                    v = float(str(val).replace('—', 'nan'))
                    if v >= 110: return 'background-color: #1B5E20; color: white; font-weight: bold'
                    if v >= 105: return 'background-color: #388E3C; color: white'
                    if v >= 100: return 'background-color: #81C784; color: black'
                    if v >= 95:  return 'background-color: #FDD835; color: black'
                    if v >= 90:  return 'background-color: #FFB74D; color: black'
                    return              'background-color: #E53935; color: white'
                except:
                    return ''

            def color_pitch_type(val):
                color = PITCH_COLORS.get(str(val), None)
                if color:
                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    lum = 0.299*r + 0.587*g + 0.114*b
                    txt = 'black' if lum > 128 else 'white'
                    return f'background-color: {color}; color: {txt}; font-weight: 600'
                return ''

            styled = disp_pbp.style.applymap(color_grade_pbp, subset=grade_cols_pbp)
            if 'Pitch Type' in disp_pbp.columns:
                styled = styled.applymap(color_pitch_type, subset=['Pitch Type'])

            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.caption(
                f"{len(view)} pitches  ·  "
                + ("Result column shows PitchCall/PlayResult from TrackMan  ·  " if has_result else "No PitchCall column found in this file  ·  ")
                + "IVB/HB in inches  ·  100 = college average"
            )

            # ── Performance Trend Chart ──────────────────────────────
            # Design goals: MA lines are the focal point, individual pitches are
            # subtle background context, damage events are clearly highlighted.
            chart_data = pbp.sort_values('#').copy()
            chart_data['Stuff+ MA5'] = chart_data['Stuff+'].rolling(5, min_periods=1).mean()
            has_pp_chart = 'Pitching+' in chart_data.columns and chart_data['Pitching+'].notna().any()
            if has_pp_chart:
                chart_data['Pitching+ MA5'] = chart_data['Pitching+'].rolling(5, min_periods=1).mean()

            st.markdown("#### Performance Trend")
            st.caption("5-pitch rolling average across the outing. Filled lines show the trend; "
                       "dots are individual pitches. Damage events are marked on the timeline.")

            # ── Session summary strip ─────────────────────────────────
            n_pitches = len(chart_data)
            half = n_pitches // 2
            first_half_stuff  = float(chart_data['Stuff+'].iloc[:half].mean()) if half > 0 else float(chart_data['Stuff+'].mean())
            second_half_stuff = float(chart_data['Stuff+'].iloc[half:].mean()) if half > 0 else float(chart_data['Stuff+'].mean())
            fatigue_delta     = second_half_stuff - first_half_stuff
            peak_stuff        = float(chart_data['Stuff+'].max())
            trough_stuff      = float(chart_data['Stuff+'].min())

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("1st Half Stuff+", f"{first_half_stuff:.1f}")
            sm2.metric("2nd Half Stuff+", f"{second_half_stuff:.1f}", f"{fatigue_delta:+.1f}",
                       delta_color='normal' if abs(fatigue_delta) < 2 else ('inverse' if fatigue_delta < 0 else 'normal'))
            sm3.metric("Session Peak", f"{peak_stuff:.1f}")
            sm4.metric("Session Low", f"{trough_stuff:.1f}")

            fig_ma = go.Figure()

            # Y-axis bounds — compute first for background bands
            y_vals = list(chart_data['Stuff+'].dropna())
            if has_pp_chart:
                y_vals += list(chart_data['Pitching+'].dropna())
            y_lo = max(55, min(y_vals) - 10)
            y_hi = min(165, max(y_vals) + 10)

            # Grade-tier background bands — very subtle, give context without dominating
            fig_ma.add_hrect(y0=110, y1=y_hi,  fillcolor='#10B981', opacity=0.05, line_width=0, layer='below')
            fig_ma.add_hrect(y0=100, y1=110,   fillcolor='#10B981', opacity=0.025, line_width=0, layer='below')
            fig_ma.add_hrect(y0=90,  y1=100,   fillcolor='#F59E0B', opacity=0.025, line_width=0, layer='below')
            fig_ma.add_hrect(y0=y_lo, y1=90,   fillcolor='#EF4444', opacity=0.05, line_width=0, layer='below')

            # 100-average reference line
            fig_ma.add_hline(y=100, line_dash='dot', line_color='rgba(255,255,255,0.25)', line_width=1)

            # ── Individual pitch dots — background layer, small and translucent ──
            for pt in chart_data['Pitch Type'].unique():
                grp = chart_data[chart_data['Pitch Type'] == pt]
                result_col = grp['Result'].fillna('') if has_result else pd.Series([''] * len(grp), index=grp.index)
                fig_ma.add_trace(go.Scatter(
                    x=grp['#'], y=grp['Stuff+'],
                    mode='markers',
                    name=pt,
                    marker=dict(
                        size=6, color=PITCH_COLORS.get(pt, '#94A3B8'),
                        line=dict(color='rgba(255,255,255,0.15)', width=0.5),
                        opacity=0.45,
                    ),
                    customdata=np.stack([grp['Velo'].values, result_col.values], axis=-1),
                    hovertemplate=(
                        f"<b>{pt}</b> · Pitch #%{{x}}<br>"
                        "Velo: %{customdata[0]} mph · Stuff+: %{y:.1f}"
                        + ("<br>Result: %{customdata[1]}" if has_result else "")
                        + "<extra></extra>"
                    ),
                    legendgroup='pitches', legendgrouptitle_text='Pitch Types',
                ))

            # ── Stuff+ MA line — primary focal point ──
            fig_ma.add_trace(go.Scatter(
                x=chart_data['#'], y=chart_data['Stuff+ MA5'],
                mode='lines',
                name='Stuff+ trend',
                line=dict(color='#60A5FA', width=3.5, shape='spline', smoothing=0.8),
                hovertemplate="Pitch #%{x}<br><b>Stuff+ (5-MA):</b> %{y:.1f}<extra></extra>",
                legendgroup='trends', legendgrouptitle_text='Rolling Average',
            ))

            # ── Pitching+ MA line — secondary focal point ──
            if has_pp_chart:
                fig_ma.add_trace(go.Scatter(
                    x=chart_data['#'], y=chart_data['Pitching+ MA5'],
                    mode='lines',
                    name='Pitching+ trend',
                    line=dict(color='#F472B6', width=3.5, shape='spline', smoothing=0.8),
                    hovertemplate="Pitch #%{x}<br><b>Pitching+ (5-MA):</b> %{y:.1f}<extra></extra>",
                    legendgroup='trends',
                ))

            # ── Damage events — large ringed markers at the individual pitch's Stuff+ ──
            # Only the stuff that actually matters to coaches: hits, HRs, Ks.
            if has_result:
                damage_events = [
                    {'match': 'homerun',  'color': '#DC2626', 'size': 18, 'symbol': 'star',             'label': 'HR'},
                    {'match': 'triple',   'color': '#EA580C', 'size': 15, 'symbol': 'diamond',          'label': '3B'},
                    {'match': 'double',   'color': '#F97316', 'size': 14, 'symbol': 'diamond',          'label': '2B'},
                    {'match': 'single',   'color': '#FBBF24', 'size': 12, 'symbol': 'diamond',          'label': '1B'},
                    {'match': 'strikeout','color': '#22C55E', 'size': 13, 'symbol': 'star-triangle-up', 'label': 'K'},
                ]
                for ev in damage_events:
                    hits = chart_data[chart_data['Result'].str.lower().str.contains(
                        ev['match'], case=False, na=False, regex=False)]
                    if hits.empty: continue
                    fig_ma.add_trace(go.Scatter(
                        x=hits['#'], y=hits['Stuff+'],
                        mode='markers',
                        marker=dict(
                            size=ev['size'], color=ev['color'], symbol=ev['symbol'],
                            line=dict(color='white', width=1.5),
                        ),
                        name=ev['label'],
                        hovertemplate=(
                            f"<b>{ev['label']}</b> · Pitch #%{{x}}<br>"
                            "Stuff+: %{y:.1f}"
                            "<extra></extra>"
                        ),
                        legendgroup='events', legendgrouptitle_text='Key Events',
                    ))

            fig_ma.update_layout(
                xaxis=dict(
                    title=dict(text="Pitch Number", font=dict(size=11, color='#94A3B8')),
                    showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False,
                    range=[0.5, n_pitches + 0.5],
                    tickfont=dict(color='#94A3B8', size=10),
                ),
                yaxis=dict(
                    title=dict(text="Grade  (100 = college avg)", font=dict(size=11, color='#94A3B8')),
                    range=[y_lo, y_hi],
                    showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False,
                    tickfont=dict(color='#94A3B8', size=10),
                    tickmode='array',
                    tickvals=[v for v in [60, 70, 80, 90, 100, 110, 120, 130, 140, 150] if y_lo <= v <= y_hi],
                ),
                height=480,
                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                font=dict(color='white', family='Inter, -apple-system, sans-serif'),
                legend=dict(
                    orientation='v',
                    yanchor='top', y=1.0,
                    xanchor='left', x=1.02,
                    font=dict(color='#CBD5E1', size=10),
                    bgcolor='rgba(0,0,0,0)',
                    groupclick='toggleitem',
                ),
                margin=dict(t=20, b=50, l=60, r=160),
                hovermode='closest',
            )
            st.plotly_chart(fig_ma, use_container_width=True)

            # ── Download ──────────────────────────────────────────────
            csv_pbp = view[show_cols].to_csv(index=False).encode()
            st.download_button(
                "⬇️  Download pitch-by-pitch CSV",
                data=csv_pbp,
                file_name=f"{pitcher_name.replace(' ', '_')}_pitch_by_pitch.csv",
                mime="text/csv",
                key="pbp_download"
            )


# ------------------------------------------------------------------
# AI Scouting Report (Gemini) — cached per pitcher, throttled per session
# ------------------------------------------------------------------
if 'ai_reports'   not in st.session_state: st.session_state.ai_reports   = {}
if 'last_ai_call' not in st.session_state: st.session_state.last_ai_call = 0.0

st.divider()
with st.expander("🤖  AI Scouting Report (Gemini)", expanded=False):
    # Show cached report if we already generated one for this pitcher
    cached = st.session_state.ai_reports.get(pitcher_name)

    col_a, col_b = st.columns([3, 1])
    with col_b:
        btn_label = "Regenerate" if cached else "Generate Report"
        gen = st.button(btn_label, key="gen_gemini_summary", use_container_width=True)

    if cached and not gen:
        st.markdown(cached[0])
        if cached[1]:
            st.caption(f"Generated by `{cached[1]}` · Cached — click Regenerate to refresh.")

    if gen:
        # Client-side throttle: at least 5s between calls to protect free-tier quota
        now = time.time()
        since_last = now - st.session_state.last_ai_call
        if since_last < 5:
            st.warning(f"⏱️  Slow down — wait {5 - since_last:.0f}s before generating again. "
                       "This protects your Gemini free-tier quota.")
        else:
            st.session_state.last_ai_call = now
            with st.spinner("Generating scouting report..."):
                fb_metrics = {
                    'velo': avg_fb_vel, 'ivb': avg_fb_vert,
                    'hb':   avg_fb_horz, 'spin': avg_fb_spin,
                }
                max_v = None
                total_p = None
                if st.session_state.input_mode == "TrackMan Analyzer" and st.session_state.tm_data is not None:
                    sub_ai = st.session_state.tm_data[st.session_state.tm_data['pitcher'] == pitcher_name]
                    if not sub_ai.empty:
                        max_v = float(sub_ai['rel_speed'].max())
                        total_p = int(len(sub_ai))
                summary_text, used_model = generate_gemini_summary(
                    pitcher_name=pitcher_name,
                    is_righty=pitcher_is_righty_display,
                    scores_df=df,
                    fb_metrics=fb_metrics,
                    max_velo=max_v,
                    total_pitches=total_p,
                )
            st.markdown(summary_text)
            if used_model:
                st.caption(f"Generated by `{used_model}` · Verify all claims against the data above.")
                st.session_state.ai_reports[pitcher_name] = (summary_text, used_model)


# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.divider()
if st.session_state.input_mode == "Arsenal Builder":
    if st.button("🗑️  Clear All Pitches", type="secondary"):
        st.session_state.pitches = []
        st.rerun()