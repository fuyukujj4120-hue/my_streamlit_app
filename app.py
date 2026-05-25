"""
版本 A：特徵導向流程
流程：看圖 → 標部位特徵（眼睛/耳朵/尾巴/身體姿勢）→ 選最終情緒 → 若選「其他/無法判斷」則選原因 → 填信心
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="家貓情緒標註系統｜版本 A 特徵導向", layout="wide")

st.markdown(
    """
    <style>
    .section-title {
        font-size: 20px;
        font-weight: 800;
        color: #222;
        margin-bottom: 4px;
        padding-bottom: 8px;
        border-bottom: 2.5px solid #2e7d5e;
        display: inline-block;
    }
    .info-box {
        padding: 12px 16px;
        border-radius: 10px;
        background: #e8f5e9;
        border: 1.5px solid #66bb6a;
        margin: 8px 0 12px 0;
        font-size: 14px;
        color: #2e7d32;
    }
    .warn-box {
        padding: 12px 16px;
        border-radius: 10px;
        background: #fff8e1;
        border: 1.5px solid #f0c36d;
        margin: 8px 0 12px 0;
        font-size: 14px;
    }
    .step-badge {
        display: inline-block;
        background: #2e7d5e;
        color: white;
        font-size: 13px;
        font-weight: 700;
        border-radius: 20px;
        padding: 3px 14px;
        margin-bottom: 10px;
    }
    .progress-bar {
        display: flex;
        gap: 0;
        margin: 8px 0 20px 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
    }
    .prog-item {
        flex: 1;
        text-align: center;
        padding: 10px 8px;
        font-size: 13px;
        font-weight: 600;
        border-right: 1px solid #e0e0e0;
    }
    .prog-item:last-child { border-right: none; }
    .prog-done { background: #edf7ed; color: #2e7d32; }
    .prog-active { background: #2e7d5e; color: #fff; }
    .prog-pending { background: #f9f9f9; color: #aaa; }
    div[data-testid="stRadio"] label { cursor: pointer; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 常數定義
# ──────────────────────────────────────────────

FLOW_TYPE = "A"
APP_TITLE = "家貓情緒標註系統｜版本 A（特徵導向）"

# 情緒類別（含操作型定義）
EMOTION_CATEGORIES = {
    "害怕": "耳朵後壓、身體壓低、瞳孔放大、試圖退縮或警戒",
    "生氣": "身體緊繃、耳朵側壓或後壓、尾巴快速擺動、露齒或攻擊姿勢",
    "滿意": "眼睛半閉、姿勢放鬆、耳朵自然、與環境互動平穩",
    "好奇": "注意力明顯集中、耳朵朝向刺激來源、身體前傾或探索姿勢",
    "中性": "無明顯正向或負向情緒線索，整體狀態平穩",
    "其他／無法判斷": "影像品質不足、線索不足、多種情緒並存或超出現有分類",
}

MAIN_EMOTIONS = ["害怕", "生氣", "滿意", "好奇", "中性", "其他／無法判斷"]
UNCERTAIN_EMOTION = "其他／無法判斷"

# 無法判斷原因選項
UNCERTAIN_REASONS = [
    "影像品質不足（模糊、光線不足）",
    "家貓部位被遮擋，無法觀察",
    "多種情緒並存，難以判定主導情緒",
    "超出現有分類，無適合選項",
    "其他原因",
]

# 各部位特徵選項（供版本 A 特徵導向流程使用）
FEATURE_OPTIONS = {
    "眼睛": [
        "雙眼睜大",
        "瞳孔圓形放大",
        "眼睛半閉",
        "瞳孔縮小（橢圓形）",
        "直視前方",
        "避免眼神接觸",
        "無法判讀",
    ],
    "耳朵": [
        "直立朝前",
        "向側面轉動",
        "背面壓平",
        "朝向刺激來源",
        "耳廓不可見",
        "無法判讀",
    ],
    "尾巴": [
        "垂直豎起",
        "水平伸展",
        "夾在身體下方",
        "尾巴快速甩動",
        "放鬆靜止",
        "尾巴看不見",
        "無法判讀",
    ],
    "身體姿勢": [
        "身體放鬆（趴／坐）",
        "身體緊繃",
        "身體壓低",
        "身體前傾探索",
        "毛髮豎立",
        "發抖",
        "無法判讀",
    ],
}

# 圖片列表（目前空白，待照片確定後填入）
# 格式：{"image_id": "cat_001", "path": "images/cat_001.jpg"}
IMAGES: list[dict] = [
    # {"image_id": "cat_001", "path": "images/cat_001.jpg"},
    # {"image_id": "cat_002", "path": "images/cat_002.jpg"},
]

# ──────────────────────────────────────────────
# Session State 初始化
# ──────────────────────────────────────────────

def init_session():
    defaults = {
        "page": "intro",           # intro | annotation | survey | done
        "participant_id": "",
        "current_index": 0,
        "annotation_step": 1,      # 1=特徵, 2=情緒, 3=原因(若需), 4=信心
        # 當前標註暫存
        "selected_features": {},   # {部位: 選項}
        "final_emotion": None,
        "uncertain_reason": None,
        "uncertain_other_text": "",
        "confidence": None,
        "step_start_time": None,
        "annotation_start_time": None,
        # 儲存所有標註結果
        "annotations": [],         # list of record dicts
        # 問卷
        "survey_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_annotation_state():
    st.session_state.annotation_step = 1
    st.session_state.selected_features = {}
    st.session_state.final_emotion = None
    st.session_state.uncertain_reason = None
    st.session_state.uncertain_other_text = ""
    st.session_state.confidence = None
    st.session_state.annotation_start_time = time.time()
    st.session_state.step_start_time = time.time()
    # 清除 widget keys
    idx = st.session_state.current_index
    for part in FEATURE_OPTIONS:
        key = f"feat_{idx}_{part}"
        if key in st.session_state:
            del st.session_state[key]
    for k in [f"emotion_{idx}", f"uncertain_reason_{idx}", f"uncertain_other_{idx}", f"confidence_{idx}"]:
        if k in st.session_state:
            del st.session_state[k]


# ──────────────────────────────────────────────
# 進度條
# ──────────────────────────────────────────────

def render_progress_bar():
    step = st.session_state.annotation_step
    # 步驟：1=部位特徵, 2=選情緒, 3=原因(條件), 4=信心
    labels = ["Step 1：部位特徵", "Step 2：選擇情緒", "Step 3：標註信心"]
    # 若最終情緒是「其他/無法判斷」則加入原因步驟
    final_emo = st.session_state.final_emotion
    if final_emo == UNCERTAIN_EMOTION or st.session_state.annotation_step == 3 and final_emo == UNCERTAIN_EMOTION:
        labels = ["Step 1：部位特徵", "Step 2：選擇情緒", "Step 3：無法判斷原因", "Step 4：標註信心"]

    current_step_idx = step - 1
    items_html = ""
    for i, label in enumerate(labels):
        if i < current_step_idx:
            items_html += f'<div class="prog-item prog-done">✅ {label}</div>'
        elif i == current_step_idx:
            items_html += f'<div class="prog-item prog-active">▶ {label}</div>'
        else:
            items_html += f'<div class="prog-item prog-pending">◻ {label}</div>'

    st.markdown(f'<div class="progress-bar">{items_html}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 頁面：說明頁
# ──────────────────────────────────────────────

def render_intro_page():
    st.markdown(f"## 🐱 {APP_TITLE}")
    st.markdown(
        """
        <div class="info-box">
        <b>版本 A：特徵導向流程</b><br>
        本流程將引導您先觀察家貓照片中的各部位特徵，再根據特徵判斷情緒。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 標註流程說明")
    st.markdown("""
    1. **Step 1：觀察並標註部位特徵**  
       請依序選擇照片中家貓的眼睛、耳朵、尾巴、身體姿勢等部位特徵。若某部位看不清楚，請選「無法判讀」。

    2. **Step 2：根據特徵選擇最終情緒**  
       根據您在 Step 1 觀察到的特徵，選擇最符合的情緒類別。

    3. **Step 3（條件）：若選擇「其他／無法判斷」，請說明原因。**

    4. **Step 4：填寫標註信心**  
       請評估您對本次標註結果的信心程度（1-5）。
    """)

    st.markdown("### 🐾 情緒類別定義")
    for emo, definition in EMOTION_CATEGORIES.items():
        st.markdown(f"- **{emo}**：{definition}")

    st.markdown("---")
    pid = st.text_input("請輸入您的受試者代號（學號）", placeholder="例如：B12345678")
    st.caption("此代號僅用於資料整理，不作為個人身分辨識用途。")

    disabled = pid.strip() == "" or len(IMAGES) == 0
    if len(IMAGES) == 0:
        st.warning("⚠️ 目前尚未載入照片，系統準備就緒後將顯示標註按鈕。")

    if st.button("✅ 我已閱讀完畢，開始標註", type="primary", disabled=disabled):
        st.session_state.participant_id = pid.strip()
        st.session_state.page = "annotation"
        st.session_state.current_index = 0
        reset_annotation_state()
        st.rerun()


# ──────────────────────────────────────────────
# 頁面：標註頁
# ──────────────────────────────────────────────

def render_annotation_page():
    participant_id = st.session_state.participant_id
    idx = st.session_state.current_index
    total = len(IMAGES)

    if idx >= total:
        st.session_state.page = "survey"
        st.rerun()
        return

    image_info = IMAGES[idx]
    image_id = image_info["image_id"]

    # 頂部資訊列
    col_info, col_prog = st.columns([3, 1])
    with col_info:
        st.markdown(f"## 🐱 {APP_TITLE}")
        st.caption(f"受試者：{participant_id}　｜　照片 {idx + 1} / {total}　｜　image_id: {image_id}")
    with col_prog:
        pct = int(idx / total * 100)
        st.markdown(
            f"""
            <div style="text-align:right;padding-top:12px;">
                <div style="font-size:28px;font-weight:800;color:#2e7d5e;">{pct}%</div>
                <div style="font-size:12px;color:#888;">已完成進度</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_progress_bar()

    # 顯示照片
    img_path = Path(image_info.get("path", ""))
    st.markdown("### 📷 請仔細觀察以下照片")
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.markdown(
            """
            <div style="background:#f5f5f5;border:2px dashed #ccc;border-radius:12px;
            height:300px;display:flex;align-items:center;justify-content:center;
            color:#aaa;font-size:16px;">
            📷 照片載入區（圖片尚未設定）
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 根據步驟渲染對應畫面
    step = st.session_state.annotation_step

    if step == 1:
        render_step1_features(idx)
    elif step == 2:
        render_step2_emotion(idx)
    elif step == 3:
        # 此步驟僅在 final_emotion == UNCERTAIN_EMOTION 時出現
        render_step3_uncertain_reason(idx)
    elif step == 4:
        render_step4_confidence(idx, image_id)


def render_step1_features(idx: int):
    """Step 1：觀察並標註各部位特徵"""
    st.markdown('<div class="step-badge">Step 1：部位特徵標註</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">請觀察照片，依序選擇各部位特徵</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">請根據照片中家貓的實際狀態選擇最符合的特徵。若某部位在照片中看不清楚，請選「無法判讀」。</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    selected = {}
    all_answered = True

    for part, options in FEATURE_OPTIONS.items():
        st.markdown(f"**{part}**")
        key = f"feat_{idx}_{part}"
        choice = st.radio(
            f"請選擇{part}特徵",
            options,
            index=None,
            key=key,
            horizontal=False,
            label_visibility="collapsed",
        )
        selected[part] = choice
        if choice is None:
            all_answered = False
        st.markdown("---")

    if not all_answered:
        st.markdown('<div class="warn-box">⚠️ 請為所有部位選擇一個選項（若看不清楚請選「無法判讀」）。</div>', unsafe_allow_html=True)

    if st.button("繼續 Step 2：選擇情緒 →", type="primary", disabled=not all_answered, key=f"step1_next_{idx}"):
        st.session_state.selected_features = selected
        st.session_state.annotation_step = 2
        st.rerun()


def render_step2_emotion(idx: int):
    """Step 2：根據特徵選擇最終情緒"""
    st.markdown('<div class="step-badge">Step 2：選擇最終情緒</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">根據您觀察到的特徵，選擇最符合的情緒</div>', unsafe_allow_html=True)

    # 顯示已選特徵摘要
    feats = st.session_state.selected_features
    if feats:
        summary_parts = [f"**{part}**：{val}" for part, val in feats.items() if val]
        st.markdown(
            '<div style="background:#f0f4f0;border-radius:10px;padding:12px 16px;margin-bottom:12px;font-size:13px;">'
            + "　｜　".join(summary_parts)
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**請根據上方觀察到的特徵，選擇最符合的情緒類別：**")

    key = f"emotion_{idx}"
    choice = st.radio(
        "最終情緒",
        MAIN_EMOTIONS,
        index=None,
        key=key,
        label_visibility="collapsed",
    )

    # 顯示選中情緒的定義
    if choice and choice in EMOTION_CATEGORIES:
        st.markdown(
            f'<div class="info-box"><b>{choice}</b>：{EMOTION_CATEGORIES[choice]}</div>',
            unsafe_allow_html=True,
        )

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← 返回 Step 1", key=f"step2_back_{idx}"):
            st.session_state.annotation_step = 1
            st.rerun()
    with col_next:
        if st.button("繼續 →", type="primary", disabled=(choice is None), key=f"step2_next_{idx}"):
            st.session_state.final_emotion = choice
            if choice == UNCERTAIN_EMOTION:
                st.session_state.annotation_step = 3
            else:
                st.session_state.annotation_step = 4
            st.rerun()


def render_step3_uncertain_reason(idx: int):
    """Step 3（條件）：選擇「其他／無法判斷」的原因"""
    st.markdown('<div class="step-badge">Step 3：無法判斷原因</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">您選擇了「其他／無法判斷」，請說明原因</div>', unsafe_allow_html=True)

    key_reason = f"uncertain_reason_{idx}"
    key_other = f"uncertain_other_{idx}"

    reason = st.radio(
        "無法判斷原因",
        UNCERTAIN_REASONS,
        index=None,
        key=key_reason,
        label_visibility="collapsed",
    )

    other_text = ""
    if reason == "其他原因":
        other_text = st.text_area(
            "請說明其他原因",
            key=key_other,
            placeholder="請輸入您無法判斷的原因…",
            height=80,
        )

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← 返回 Step 2", key=f"step3_back_{idx}"):
            st.session_state.annotation_step = 2
            st.rerun()
    with col_next:
        next_disabled = (reason is None) or (reason == "其他原因" and other_text.strip() == "")
        if st.button("繼續 Step 4 →", type="primary", disabled=next_disabled, key=f"step3_next_{idx}"):
            st.session_state.uncertain_reason = reason
            st.session_state.uncertain_other_text = other_text.strip()
            st.session_state.annotation_step = 4
            st.rerun()


def render_step4_confidence(idx: int, image_id: str):
    """Step 4：填寫標註信心"""
    st.markdown('<div class="step-badge">Step 4：標註信心</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">請評估您對本次標註結果的信心程度</div>', unsafe_allow_html=True)

    # 確認摘要
    feats = st.session_state.selected_features
    final_emotion = st.session_state.final_emotion
    summary_parts = [f"**{part}**：{val}" for part, val in feats.items() if val]
    st.markdown(
        '<div style="background:#f0f4f0;border-radius:10px;padding:12px 16px;margin-bottom:8px;font-size:13px;">'
        + "<br>".join([
            "　".join(summary_parts[:2]),
            "　".join(summary_parts[2:]) if len(summary_parts) > 2 else "",
        ])
        + f"<br><b>最終情緒：{final_emotion}</b>"
        + ("" if final_emotion != UNCERTAIN_EMOTION else f"<br>原因：{st.session_state.uncertain_reason}")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**您對這張照片的情緒標註結果有多少信心？**")
    st.caption("1 = 完全沒有信心，5 = 非常有信心")

    key = f"confidence_{idx}"
    confidence = st.radio(
        "信心程度",
        [1, 2, 3, 4, 5],
        index=None,
        key=key,
        horizontal=True,
        format_func=lambda x: {1: "1\n完全沒信心", 2: "2\n沒把握", 3: "3\n普通", 4: "4\n有把握", 5: "5\n非常有信心"}[x],
        label_visibility="collapsed",
    )

    col_back, col_save = st.columns(2)
    with col_back:
        back_step = 3 if final_emotion == UNCERTAIN_EMOTION else 2
        if st.button(f"← 返回 Step {back_step}", key=f"step4_back_{idx}"):
            st.session_state.annotation_step = back_step
            st.rerun()
    with col_save:
        if st.button("✅ 儲存並繼續下一張", type="primary", disabled=(confidence is None), key=f"step4_save_{idx}"):
            save_annotation(image_id, confidence)
            st.session_state.current_index += 1
            reset_annotation_state()
            st.rerun()


def save_annotation(image_id: str, confidence: int):
    """將當前標註結果存入 session"""
    annotation_time = round(time.time() - st.session_state.annotation_start_time, 1)
    feats = st.session_state.selected_features

    record = {
        "participant_id": st.session_state.participant_id,
        "flow_type": FLOW_TYPE,
        "image_id": image_id,
        "initial_emotion": None,          # 版本 A 無初始情緒
        "selected_features": json.dumps(feats, ensure_ascii=False),
        "final_emotion": st.session_state.final_emotion,
        "uncertain_reason": st.session_state.uncertain_reason,
        "uncertain_other_text": st.session_state.uncertain_other_text,
        "confidence": confidence,
        "annotation_time": annotation_time,
        "emotion_changed": False,          # 版本 A 沒有初始→最終的改判機制
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state.annotations.append(record)


# ──────────────────────────────────────────────
# 頁面：問卷頁
# ──────────────────────────────────────────────

def render_survey_page():
    st.markdown(f"## 🐱 {APP_TITLE}")
    st.markdown(
        '<div class="info-box"><b>標註完成！</b> 請填寫以下問卷，協助我們了解這個標註流程的使用體驗。</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### 📋 使用者問卷（版本 A 特徵導向流程）")
    st.caption("以下所有題項均以 1-5 點 Likert 量表評分（1=非常不同意，5=非常同意）")
    st.markdown("---")

    def likert(label: str, key: str) -> int | None:
        st.markdown(f"**{label}**")
        val = st.radio(
            label,
            [1, 2, 3, 4, 5],
            horizontal=True,
            index=None,
            key=key,
            format_func=lambda x: f"{x}",
            label_visibility="collapsed",
        )
        st.markdown("")
        return val

    st.markdown("#### 一、認知負荷")
    wl1 = likert("1. 我覺得此標註流程需要花費較多心力。", "wl1")
    wl2 = likert("2. 我在使用此流程時需要反覆思考才能完成標註。", "wl2")
    wl3 = likert("3. 我覺得此流程的判斷負擔較高。", "wl3")

    st.markdown("#### 二、標註信心")
    cf1 = likert("4. 我對自己最後選擇的情緒結果有信心。", "cf1")
    cf2 = likert("5. 我認為自己的標註結果有足夠依據。", "cf2")
    cf3 = likert("6. 我能根據照片中的特徵做出合理判斷。", "cf3")

    st.markdown("#### 三、流程清楚度")
    cl1 = likert("7. 我能清楚理解此標註流程的操作順序。", "cl1")

    st.markdown("#### 四、有用性")
    us1 = likert("8. 我認為此流程有助於我判斷家貓情緒。", "us1")

    st.markdown("#### 五、使用意圖")
    in1 = likert("9. 若未來需要標註更多家貓照片，我願意使用此流程。", "in1")

    st.markdown("---")
    st.markdown("#### 六、開放式回饋")
    open_feedback = st.text_area(
        "請描述您在使用版本 A（特徵導向）流程時的感受、遇到的困難，或對流程的改進建議：",
        key="open_feedback",
        height=150,
        placeholder="例如：步驟太多、不知道該看哪個部位、先看特徵再選情緒感覺比較有依據…",
    )

    scores = [wl1, wl2, wl3, cf1, cf2, cf3, cl1, us1, in1]
    all_answered = all(s is not None for s in scores)

    if not all_answered:
        st.markdown('<div class="warn-box">⚠️ 請完成所有量表題項後再提交。</div>', unsafe_allow_html=True)

    if st.button("📤 提交問卷", type="primary", disabled=not all_answered):
        survey_record = {
            "participant_id": st.session_state.participant_id,
            "flow_type": FLOW_TYPE,
            "workload_score": round((wl1 + wl2 + wl3) / 3, 2),
            "workload_1": wl1, "workload_2": wl2, "workload_3": wl3,
            "confidence_score": round((cf1 + cf2 + cf3) / 3, 2),
            "confidence_1": cf1, "confidence_2": cf2, "confidence_3": cf3,
            "clarity_score": cl1,
            "usefulness_score": us1,
            "intention_score": in1,
            "open_feedback": open_feedback,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state["survey_record"] = survey_record
        st.session_state.survey_done = True
        st.session_state.page = "done"
        st.rerun()


# ──────────────────────────────────────────────
# 頁面：完成頁
# ──────────────────────────────────────────────

def render_done_page():
    st.balloons()
    st.markdown(f"## 🎉 感謝您完成版本 A 的標註！")
    st.markdown(
        '<div class="info-box">您的標註資料與問卷回饋已記錄完成，感謝您的參與！</div>',
        unsafe_allow_html=True,
    )

    annotations = st.session_state.annotations
    survey = st.session_state.get("survey_record", {})

    if annotations:
        st.markdown("### 📊 本次標註結果摘要")
        df_ann = pd.DataFrame(annotations)
        st.dataframe(df_ann, use_container_width=True)

        # 合併標註資料與問卷資料
        df_export = df_ann.copy()
        for k, v in survey.items():
            if k not in ("participant_id", "flow_type", "timestamp"):
                df_export[k] = v

        csv_bytes = df_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ 下載標註結果 CSV",
            data=csv_bytes,
            file_name=f"annotation_A_{st.session_state.participant_id}.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("**資料欄位說明（對應 Proposal 3.4）：**")
    col_desc = {
        "participant_id": "受試者學號",
        "flow_type": "使用流程（A=特徵導向）",
        "image_id": "家貓照片編號",
        "initial_emotion": "初始情緒（版本 A 為 None）",
        "selected_features": "選擇的部位特徵（JSON）",
        "final_emotion": "最終情緒",
        "uncertain_reason": "若選其他/無法判斷，記錄原因",
        "uncertain_other_text": "其他原因補充文字",
        "confidence": "標註信心程度（1-5）",
        "annotation_time": "完成標註所需時間（秒）",
        "emotion_changed": "初始情緒與最終情緒是否不同（版本 A 為 False）",
        "workload_score": "認知負荷分數（3題平均）",
        "confidence_score": "標註信心分數（3題平均）",
        "clarity_score": "流程清楚度分數",
        "usefulness_score": "有用性分數",
        "intention_score": "使用意圖分數",
        "open_feedback": "開放式回饋",
    }
    for field, desc in col_desc.items():
        st.markdown(f"- `{field}`：{desc}")


# ──────────────────────────────────────────────
# 主程式入口
# ──────────────────────────────────────────────

st.markdown(
    f'<h1 style="font-size:26px;font-weight:800;color:#2e7d5e;margin-bottom:2px;">🐱 {APP_TITLE}</h1>'
    f'<p style="color:#888;font-size:13px;margin-top:0;">流程 A：部位特徵 → 情緒選擇 → 標註信心</p>',
    unsafe_allow_html=True,
)

init_session()

page = st.session_state.page
if page == "intro":
    render_intro_page()
elif page == "annotation":
    render_annotation_page()
elif page == "survey":
    render_survey_page()
elif page == "done":
    render_done_page()
