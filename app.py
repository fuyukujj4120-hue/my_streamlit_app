
import json
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="貓咪情緒標註系統", layout="wide")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 560px !important;
        min-width: 560px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 560px !important;
        min-width: 560px !important;
    }
    section[data-testid="stSidebar"] video {
        width: 500px !important;
        height: 400px !important;
        object-fit: contain !important;
        background: #000 !important;
        border-radius: 8px !important;
        display: block !important;
        margin: 0 auto !important;
    }

    .warn-box {
        padding: 12px 16px;
        border-radius: 10px;
        background: #fff8e1;
        border: 1.5px solid #f0c36d;
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 15px;
    }
    .ok-box {
        padding: 12px 16px;
        border-radius: 10px;
        background: #edf7ed;
        border: 1.5px solid #81c995;
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 15px;
    }
    .low-box {
        padding: 12px 16px;
        border-radius: 10px;
        background: #fdecea;
        border: 1.5px solid #f28b82;
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 15px;
    }

    div[data-testid="stRadio"] > label p {
        font-size: 22px !important;
        font-weight: 700 !important;
    }

    .feature-radio div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
        gap: 8px !important;
        width: 100% !important;
    }
    .feature-radio div[data-testid="stRadio"] label {
        padding: 8px 12px !important;
        border-radius: 18px !important;
        border: 1.5px solid #d0d0d0 !important;
        background: #fafafa !important;
        cursor: pointer !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #555 !important;
        transition: all 0.15s ease !important;
        line-height: 1.35 !important;
        text-align: center !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-height: 48px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    .feature-radio div[data-testid="stRadio"] label:hover {
        border-color: #534ab7 !important;
        background: #f0effe !important;
        color: #3c3489 !important;
    }
    .feature-radio div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    .feature-radio div[data-testid="stRadio"] > div > label > div:first-child {
        display: none !important;
    }

    .section-title {
        font-size: 22px;
        font-weight: 800;
        color: #222;
        margin-bottom: 4px;
        padding-bottom: 8px;
        border-bottom: 2.5px solid #534ab7;
        display: inline-block;
    }

    .group-divider {
        border: none;
        border-top: 1px dashed #e0e0e0;
        margin: 12px 0;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        border-radius: 10px !important;
        font-size: 13px !important;
        padding: 8px 10px !important;
        transition: all 0.15s !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background: #eeedfe !important;
        border-color: #534ab7 !important;
        color: #3c3489 !important;
    }

    div[data-testid="column"] div[data-testid="stButton"] > button {
        border-radius: 10px !important;
        font-size: 15px !important;
        padding: 10px 18px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_TITLE = "貓咪情緒標註系統"
BASE_URL = "https://storage.googleapis.com/cat-emotion-videos-fuyu/videos/"

video_names = [
    "v11__s000010__e000020.mp4",
    "v45__s000080__e000090.mp4",
    "v59__s000000__e000010.mp4",
    "v62__s000000__e000010.mp4",
    "v79__s000030__e000040.mp4",
    "v81__s000010__e000020.mp4",
    "v87__s000010__e000015.mp4",
    "v99__s000020__e000030.mp4",
    "v108__s000050__e000054.mp4",
    "v111__s000010__e000020.mp4",
    "v112__s000000__e000010.mp4",
    "v122__s000040__e000050.mp4",
    "v122__s000050__e000060.mp4",
    "v130__s000010__e000020.mp4",
    "v180__s000000__e000010.mp4",
]
VIDEOS = [{"name": name, "url": f"{BASE_URL}{name}"} for name in video_names]

MAIN_EMOTIONS = ["害怕", "憤怒/狂怒", "歡樂/玩耍", "滿意", "興趣", "uncertain"]

EMOTION_ICONS = {
    "害怕": "😿",
    "憤怒/狂怒": "😾",
    "歡樂/玩耍": "😺",
    "滿意": "😽",
    "興趣": "🐾",
    "uncertain": "❓",
}

ANNOTATION_RULES = [
    ("請先完整觀看，再進行標註", False),
    ("Step 1 先判斷是否有主要情緒；若無主要情緒則強制為 uncertain", False),
    ("Step 2 依據該情緒選擇各部位最符合的單一特徵（眼睛、耳朵、尾巴、身體）", False),
    ("Step 3 再選該情緒下最符合的單一行為特徵", False),
    ("若某一部位或行為無法觀察，選擇「無法判斷」即可", True),
    ("影片中若有兩種情緒以上請選擇可以做為主導的情緒 若兩者皆很重要請勾選為無主要情緒", True),
]

EMOTION_SCHEMA = {
    "害怕": {
        "definition": "由立即感知到的危險或危險的威脅引起的，表現為警惕和試圖撤退或逃跑",
        "features": {
            "眼睛": ["雙眼睜大", "瞳孔呈圓形且散大", "眨眼", "目光向左", "半眨眼", "緊閉雙眼(行為)", "避免眼神接觸(行為)"],
            "耳朵": ["向側面", "耳廓不可見", "背面壓平"],
            "尾巴": ["夾在身體下方", "或繞在身體旁"],
            "身體": ["毛豎起（炸毛）", "身體緊繃", "發抖", "壓低身體"],
            "行為": ["高度警戒", "受驚反應", "顫抖", "四肢僵硬", "躲藏", "逃避／迴避", "梳理毛髮", "缺乏基本維持行為（進食、飲水、排泄）或不睡眠"],
        },
    },
    "憤怒/狂怒": {
        "definition": "由執行行動/實現目標的願望受挫或資源競爭引起，表現為攻擊性或攻擊威脅。",
        "features": {
            "眼睛": ["瞳孔呈橢圓形且散大", "直視"],
            "耳朵": ["向側面旋轉", "可見內耳廓"],
            "尾巴": ["壓低且僵硬", "呈倒 L 形", "拍打地面", "快速左右或上下甩動"],
            "身體": ["毛髮豎立（沿脊椎與尾巴）", "身體前傾", "臀部抬高", "拱背站立", "露出牙齒"],
            "行為": ["撲向或追逐目標", "用爪或口攻擊", "驅趕其他個體"],
        },
    },
    "歡樂/玩耍": {
        "definition": "表現為非功能性行為，包括運動遊戲、社交遊戲或物件遊戲。",
        "features": {
            "眼睛": ["瞳孔因興奮而放大/變圓", "瞳孔放鬆/變軟"],
            "耳朵": ["直立且面向前方"],
            "尾巴": ["垂直", "可能呈倒 U 形"],
            "身體": ["半張開嘴", "拱背", "身體姿勢變化多樣"],
            "行為": [
                "攀爬", "奔跑", "接近其他貓", "跳躍", "拍打、撥弄",
                "用前肢抓住對方", "咬對方", "翻滾或露出腹部", "扭打",
                "踢擊", "追逐", "側移或逃跑", "站立去抓物體",
                "拍打物體", "用爪抓住物體", "嗅聞、舔舐", "咬或啃",
                "丟擲", "與物體扭打", "潛行", "捕獵追逐", "捕獵跳躍", "撲擊"
            ],
        },
    },
    "滿意": {
        "definition": "由需求和願望得到滿足而產生的正向情緒狀態，表現為休息、平靜和親和。",
        "features": {
            "眼睛": ["瞳孔呈小的縮瞳狀垂直卵圓形", "半睜"],
            "耳朵": ["直立且面向前方"],
            "尾巴": ["放鬆且靜止狀態", "輕微彎曲"],
            "身體": ["坐著", "身體放鬆(趴/躺著)", "蜷縮"],
            "行為": ["伸展", "打哈欠(想睡覺)", "自我或互相梳理毛髮", "踩踏", "親暱行為（碰鼻、頂頭、磨蹭）", "翻滾", "撒嬌", "進食", "抓物體"],
        },
    },
    "興趣": {
        "definition": "由新奇或顯著刺激引起，表現為注意、定向或探索行為。",
        "features": {
            "眼睛": ["瞳孔放大/呈圓形", "目光向右"],
            "耳朵": ["耳朵直立並朝向刺激物", "耳朵輕微抖動"],
            "尾巴": ["水平", "豎起"],
            "身體": ["可能用後腳站立", "前腳靠在物體上", "頭微向前伸", "頭向右轉"],
            "行為": ["觀察個體或物體", "探索環境", "嗅聞", "舔舐", "用爪觸碰", "社交接觸（碰鼻、磨蹭）", "捕獵行為（潛行、追逐、撲擊、抓取、咬）"],
        },
    },
}

DEFINITION_IMAGE_MAP = {
    "害怕": Path("images/fear.png"),
    "憤怒/狂怒": Path("images/anger.png"),
    "歡樂/玩耍": Path("images/joy.png"),
    "滿意": Path("images/contentment.png"),
    "興趣": Path("images/interest.png"),
}

SHEET_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbzA_0AnSkFSeN6GFLr1wDsvx-l28-5a3s605l9CV6QwwTfcJ4GejNepx2yOIjX7M85m/exec"
SHEET_SECRET = "my_cat_annotation_secret"


@st.dialog("情緒定義")
def show_emotion_dialog(emotion_name: str):
    item = EMOTION_SCHEMA[emotion_name]
    st.subheader(emotion_name)
    img_path = DEFINITION_IMAGE_MAP.get(emotion_name)
    if img_path and img_path.exists():
        st.image(str(img_path), use_container_width=True)
    st.write(f"**定義：** {item['definition']}")
    for grp, opts in item["features"].items():
        st.markdown(f"**{grp}**")
        for opt in opts:
            st.markdown(f"- {opt}")


def append_to_google_sheet(record: dict, annotator_name: str):
    payload = {**record, "annotator_name": annotator_name, "secret": SHEET_SECRET}
    resp = requests.post(SHEET_WEBHOOK_URL, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise ValueError(data.get("error", "Unknown Google Sheet error"))


def load_video_files():
    return VIDEOS


def render_small_video(video_item: dict):
    st.video(video_item["url"])


def get_video_name(video_item: dict):
    return video_item["name"]


def compute_store_key(annotator_name: str, video_name: str):
    raw = f"{annotator_name.strip()}::{video_name}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def get_annotations_store():
    if "annotations_store" not in st.session_state:
        st.session_state["annotations_store"] = {}
    return st.session_state["annotations_store"]


def get_saved_record(annotator_name: str, video_name: str):
    if not annotator_name:
        return None
    return get_annotations_store().get(compute_store_key(annotator_name, video_name))


def upsert_annotation(record: dict, annotator_name: str):
    get_annotations_store()[compute_store_key(annotator_name, record["video_file"])] = record


def get_annotations_df(annotator_name: str):
    if not annotator_name:
        return pd.DataFrame()
    rows = [v for v in get_annotations_store().values() if v.get("annotator_name", "") == annotator_name.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df


def render_definition_block(emotion_name: str, item: dict):
    icon = EMOTION_ICONS.get(emotion_name, "")
    with st.expander(f"{icon} {emotion_name}", expanded=False):
        img_path = DEFINITION_IMAGE_MAP.get(emotion_name)
        if img_path and img_path.exists():
            st.image(str(img_path), use_container_width=True)
        st.write(f"**定義：** {item['definition']}")
        for grp, opts in item["features"].items():
            st.markdown(f"- **{grp}**")
            for opt in opts:
                st.markdown(f"  - {opt}")


def get_emotion_feature_groups(emotion_name: str):
    if emotion_name in EMOTION_SCHEMA:
        item = EMOTION_SCHEMA[emotion_name]
        return {k: v for k, v in item["features"].items() if k != "行為"}, item["features"].get("行為", [])
    return {"眼睛": [], "耳朵": [], "尾巴": [], "身體": []}, []


def get_all_step2_groups_union():
    union_groups = {"眼睛": [], "耳朵": [], "尾巴": [], "身體": []}
    for emotion_data in EMOTION_SCHEMA.values():
        for group_name in union_groups:
            for feature in emotion_data["features"].get(group_name, []):
                if feature not in union_groups[group_name]:
                    union_groups[group_name].append(feature)
    return union_groups


def get_all_behavior_union():
    behaviors = []
    for emotion_data in EMOTION_SCHEMA.values():
        for behavior in emotion_data["features"].get("行為", []):
            if behavior not in behaviors:
                behaviors.append(behavior)
    return behaviors


def reset_feature_widget_state(video_index: int):
    prefixes = [
        f"step2_single_{video_index}_",
        f"step3_behavior_value_{video_index}",
        f"step3_behavior_radio_{video_index}",
        f"step3_behavior_unknown_{video_index}",
        f"step1_no_primary_{video_index}",
        f"step1_emotion_{video_index}",
    ]
    keys_to_delete = [k for k in list(st.session_state.keys()) if any(k.startswith(prefix) for prefix in prefixes)]
    for k in keys_to_delete:
        del st.session_state[k]


def clear_step4_state(video_index: int):
    for key in [f"final_emotion_radio_{video_index}", f"note_{video_index}"]:
        if key in st.session_state:
            del st.session_state[key]


def reset_step_flow():
    video_index = st.session_state.current_index
    st.session_state.annotation_step = 1
    st.session_state.selected_emotion = None
    st.session_state.step1_has_primary_emotion = None
    st.session_state.step2_selected_features = []
    st.session_state.step2_unknown_groups = []
    st.session_state.step2_result = None
    st.session_state.step3_selected_behavior = []
    st.session_state.step3_unknown_behavior = False
    st.session_state.step3_result = None
    st.session_state.loaded_saved_record_video = None
    reset_feature_widget_state(video_index)
    clear_step4_state(video_index)


def go_to_instruction():
    st.session_state.page = "instruction"
    reset_step_flow()
    st.rerun()


def go_to_step1():
    st.session_state.annotation_step = 1
    st.session_state.step2_result = None
    st.session_state.step3_selected_behavior = []
    st.session_state.step3_unknown_behavior = False
    st.session_state.step3_result = None
    clear_step4_state(st.session_state.current_index)
    st.rerun()


def go_to_step2():
    st.session_state.annotation_step = 2
    st.session_state.step3_result = None
    clear_step4_state(st.session_state.current_index)
    st.rerun()


def evaluate_feature_support(selected_emotion, selected_features, unknown_groups):
    if not selected_emotion:
        return {"emotion": None, "feature_count": 0, "confidence": "低", "summary": "尚未選擇情緒。"}

    if selected_emotion == "uncertain":
        summary = "情緒為 uncertain，特徵只作為紀錄。"
        if unknown_groups:
            summary += f"（無法判斷部位：{', '.join(unknown_groups)}）"
        return {"emotion": selected_emotion, "feature_count": len(selected_features), "confidence": "uncertain", "summary": summary}

    feature_count = len(selected_features)
    if feature_count >= 2:
        summary = "已選擇 2 個以上 Step 2 特徵。"
    elif feature_count == 1:
        summary = "已選擇 1 個 Step 2 特徵。"
    else:
        summary = "尚未選擇任何 Step 2 特徵。"

    if unknown_groups:
        summary += f"（無法判斷部位：{', '.join(unknown_groups)}）"
    return {"emotion": selected_emotion, "feature_count": feature_count, "confidence": "低", "summary": summary}


def evaluate_behavior_support(selected_emotion, step2_result, selected_behavior, unknown_behavior):
    if not selected_emotion:
        return {"emotion": None, "confidence": "低", "summary": "尚未選擇情緒。", "matched": False}

    has_behavior = (not unknown_behavior) and len(selected_behavior) > 0

    if selected_emotion == "uncertain":
        return {"emotion": selected_emotion, "confidence": "uncertain", "summary": "情緒為 uncertain，特徵與行為只作為紀錄。", "matched": has_behavior}

    feature_count = step2_result.get("feature_count", 0)
    if feature_count >= 2 and has_behavior:
        confidence = "高"
        summary = "✅ 信心度：高（有任意 2 個 Step 2 特徵 + 行為）"
    elif feature_count >= 2 and not has_behavior:
        confidence = "中"
        summary = "✅ 信心度：中（有任意 2 個 Step 2 特徵 + 無行為）"
    elif feature_count == 1 and has_behavior:
        confidence = "中"
        summary = "✅ 信心度：中（有任意 1 個 Step 2 特徵 + 1 行為）"
    elif feature_count == 0 and has_behavior:
        confidence = "低"
        summary = "⚠️ 信心度：低（有 0 個 Step 2 特徵 + 1 行為）"
    else:
        confidence = "低"
        summary = "⚠️ 信心度：低（皆未選擇或資訊不足）"

    return {"emotion": selected_emotion, "confidence": confidence, "summary": summary, "matched": has_behavior}


def init_session(videos):
    defaults = {
        "page": "instruction",
        "current_index": 0,
        "videos": videos,
        "completed": 0,
        "annotation_step": 1,
        "selected_emotion": None,
        "step1_has_primary_emotion": None,
        "step2_selected_features": [],
        "step2_unknown_groups": [],
        "step2_result": None,
        "step3_selected_behavior": [],
        "step3_unknown_behavior": False,
        "step3_result": None,
        "loaded_saved_record_video": None,
        "annotations_store": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_progress_banner():
    step = st.session_state.annotation_step
    labels = ["Step 1：情緒", "Step 2：特徵", "Step 3：行為", "Step 4：最終確認"]
    step_map = {1: 0, 2: 1, 3: 2, 4: 3}
    current_label_idx = step_map.get(step, 0)

    items_html = ""
    for i, label in enumerate(labels):
        if i < current_label_idx:
            items_html += f'<div class="prog-item prog-done">✅ {label}</div>'
        elif i == current_label_idx:
            items_html += f'<div class="prog-item prog-active">▶ {label}</div>'
        else:
            items_html += f'<div class="prog-item prog-pending">◻ {label}</div>'

    st.markdown(
        f"""
        <style>
        .progress-bar {{
            display: flex;
            gap: 0;
            margin: 8px 0 20px 0;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #e0e0e0;
        }}
        .prog-item {{
            flex: 1;
            text-align: center;
            padding: 10px 8px;
            font-size: 13.5px;
            font-weight: 600;
            border-right: 1px solid #e0e0e0;
            transition: all 0.2s;
        }}
        .prog-item:last-child {{ border-right: none; }}
        .prog-done {{ background: #edf7ed; color: #2e7d32; }}
        .prog-active {{ background: #534ab7; color: #fff; }}
        .prog-pending {{ background: #f9f9f9; color: #aaa; }}
        </style>
        <div class="progress-bar">{items_html}</div>
        """,
        unsafe_allow_html=True,
    )


def load_saved_step2_group_choices(saved_record):
    if not saved_record:
        return {}
    mapping = {}
    for group_name in ["眼睛", "耳朵", "尾巴", "身體"]:
        selected_raw = saved_record.get(f"step2_{group_name}", "null")
        try:
            selected_value = json.loads(selected_raw) if selected_raw not in ["", None] else None
        except Exception:
            selected_value = None
        unknown_flag = str(saved_record.get(f"step2_{group_name}_無法判斷", "False")) == "True"
        mapping[group_name] = "無法判斷" if unknown_flag else selected_value
    return mapping


def load_saved_step3_choice(saved_record):
    if not saved_record:
        return None, False
    unknown_flag = str(saved_record.get("step3_unknown_behavior", "False")) == "True"
    selected_raw = saved_record.get("step3_selected_behavior", "null")
    try:
        selected_value = json.loads(selected_raw) if selected_raw not in ["", None] else None
    except Exception:
        selected_value = None
    return selected_value, unknown_flag


def render_single_choice_feature_selector(group_dict, page_index, prefix, saved_choices):
    selected_features = []
    unknown_groups = []
    groups = list(group_dict.keys())

    for idx_group, group_name in enumerate(groups):
        options = group_dict.get(group_name, []) + ["無法判斷"]
        widget_key = f"{prefix}_{page_index}_{group_name}"

        if widget_key not in st.session_state:
            st.session_state[widget_key] = saved_choices.get(group_name)

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
            f'<span style="font-size:15px;font-weight:700;color:#333;">{group_name}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        current_value = st.session_state[widget_key]
        st.markdown('<div class="feature-radio">', unsafe_allow_html=True)
        choice = st.radio(
            f"{group_name}選項",
            options,
            index=options.index(current_value) if current_value in options else None,
            key=widget_key,
            horizontal=False,
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if choice == "無法判斷":
            unknown_groups.append(group_name)
        elif choice is not None:
            selected_features.append(choice)

        if idx_group != len(groups) - 1:
            st.markdown('<hr class="group-divider">', unsafe_allow_html=True)

    return selected_features, unknown_groups


def render_single_choice_behavior_selector(options, page_index, saved_behavior, saved_unknown):
    behavior_value_key = f"step3_behavior_value_{page_index}"
    behavior_radio_key = f"step3_behavior_radio_{page_index}"
    unknown_key = f"step3_behavior_unknown_{page_index}"

    if behavior_value_key not in st.session_state:
        st.session_state[behavior_value_key] = saved_behavior
    if unknown_key not in st.session_state:
        st.session_state[unknown_key] = saved_unknown
    if behavior_radio_key not in st.session_state:
        st.session_state[behavior_radio_key] = "無法判斷" if saved_unknown else saved_behavior

    choices = options + ["無法判斷"]

    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
        '<span style="font-size:15px;font-weight:700;color:#333;">行為</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="feature-radio">', unsafe_allow_html=True)
    choice = st.radio(
        "行為選項",
        choices,
        index=choices.index(st.session_state[behavior_radio_key]) if st.session_state[behavior_radio_key] in choices else None,
        key=behavior_radio_key,
        horizontal=False,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if choice == "無法判斷":
        st.session_state[unknown_key] = True
        st.session_state[behavior_value_key] = None
        return [], True

    st.session_state[unknown_key] = False
    st.session_state[behavior_value_key] = choice
    return [choice] if choice is not None else [], False


def split_step2_features_by_group(selected_emotion_local, selected_features):
    eye_value = None
    ear_value = None
    tail_value = None
    body_value = None

    if selected_emotion_local == "uncertain":
        groups = get_all_step2_groups_union()
    elif selected_emotion_local in EMOTION_SCHEMA:
        groups = {k: v for k, v in EMOTION_SCHEMA[selected_emotion_local]["features"].items() if k != "行為"}
    else:
        groups = {"眼睛": [], "耳朵": [], "尾巴": [], "身體": []}

    eye_pool = set(groups.get("眼睛", []))
    ear_pool = set(groups.get("耳朵", []))
    tail_pool = set(groups.get("尾巴", []))
    body_pool = set(groups.get("身體", []))

    for f in selected_features:
        if f in eye_pool:
            eye_value = f
        elif f in ear_pool:
            ear_value = f
        elif f in tail_pool:
            tail_value = f
        elif f in body_pool:
            body_value = f

    return eye_value, ear_value, tail_value, body_value


st.markdown(
    f'<h1 style="font-size:28px;font-weight:800;color:#222;margin-bottom:2px;">'
    f'🐱 {APP_TITLE}</h1>'
    f'<p style="color:#888;font-size:14px;margin-top:0;margin-bottom:20px;">'
    f'流程：Step 1 選情緒 → Step 2 選特徵 → Step 3 選行為 → Step 4 最終確認</p>',
    unsafe_allow_html=True,
)

videos = load_video_files()
init_session(videos)

with st.sidebar:
    st.markdown('<div style="font-size:17px;font-weight:800;color:#222;margin-bottom:12px;">📊 標註進度</div>', unsafe_allow_html=True)

    total = len(st.session_state.videos)
    completed = st.session_state.completed
    pct = int(completed / total * 100) if total else 0

    st.markdown(
        f"""
        <div style="background:#f5f4ff;border-radius:10px;padding:12px 14px;border:1px solid #d4d0f5;margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;font-size:13px;color:#555;margin-bottom:6px;">
                <span>已完成 {completed} / {total} 支影片</span>
                <span style="font-weight:700;color:#534ab7;">{pct}%</span>
            </div>
            <div style="background:#e0e0e0;border-radius:6px;height:8px;overflow:hidden;">
                <div style="background:#534ab7;height:8px;width:{pct}%;border-radius:6px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    annotator_name = st.text_input("👤 標註者姓名 / 編號", value=st.session_state.get("annotator_name", ""), placeholder="請輸入姓名或編號…")
    st.session_state["annotator_name"] = annotator_name

    if st.button("🏠 回到說明頁", use_container_width=True):
        go_to_instruction()

    if st.session_state.page == "annotation" and len(st.session_state.videos) > 0 and st.session_state.current_index < len(st.session_state.videos):
        sidebar_video = st.session_state.videos[st.session_state.current_index]
        st.markdown("---")
        st.markdown(
            f'<div style="font-size:12px;color:#888;margin-bottom:4px;">🎬 {get_video_name(sidebar_video)}</div>'
            f'<div style="font-size:12px;color:#aaa;margin-bottom:6px;">索引：{st.session_state.current_index + 1} / {total}</div>',
            unsafe_allow_html=True,
        )
        render_small_video(sidebar_video)

    st.markdown("---")
    st.markdown('<div style="font-size:13px;font-weight:700;color:#555;margin-bottom:8px;">📖 情緒定義快速查看</div>', unsafe_allow_html=True)
    button_labels = {
        "害怕": "😿 害怕",
        "憤怒/狂怒": "😾 憤怒/狂怒",
        "歡樂/玩耍": "😺 歡樂/玩耍",
        "滿意": "😽 滿意",
        "興趣": "🐾 興趣",
    }
    col1, col2 = st.columns(2)
    for i, emotion_name in enumerate(EMOTION_SCHEMA.keys()):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            if st.button(button_labels.get(emotion_name, emotion_name), key=f"sidebar_emotion_{emotion_name}", use_container_width=True):
                show_emotion_dialog(emotion_name)

if st.session_state.page == "instruction":
    st.subheader("一、標註規則")
    for i, (rule, is_bold) in enumerate(ANNOTATION_RULES, start=1):
        st.markdown(f"{i}. **{rule}**" if is_bold else f"{i}. {rule}")

    st.subheader("二、情緒定義與判斷參考")
    for emo_name, emo_item in EMOTION_SCHEMA.items():
        render_definition_block(emo_name, emo_item)

    st.info("請先完整閱讀以上規則與定義，再開始標註。")
    start_disabled = (annotator_name.strip() == "") or (len(st.session_state.videos) == 0)
    if len(st.session_state.videos) == 0:
        st.warning("目前找不到影片。")

    if st.button("✅ 我已閱讀完畢，開始標註", disabled=start_disabled, type="primary"):
        st.session_state.page = "annotation"
        reset_step_flow()
        st.rerun()

else:
    if len(st.session_state.videos) == 0:
        st.error("沒有可標註的影片。")
        st.stop()

    if st.session_state.current_index >= len(st.session_state.videos):
        st.success("🎉 所有影片都標註完成了！")
        df_all = get_annotations_df(annotator_name)
        if not df_all.empty:
            st.dataframe(df_all, use_container_width=True)
            csv_bytes = df_all.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ 下載標註結果 CSV", data=csv_bytes, file_name=f"annotations_{annotator_name.strip()}.csv", mime="text/csv")
        st.stop()

    current_video = st.session_state.videos[st.session_state.current_index]
    current_video_name = get_video_name(current_video)
    saved_record = get_saved_record(annotator_name, current_video_name) if annotator_name else None

    st.markdown(
        f'<div style="background:#f5f4ff;border:1.5px solid #d4d0f5;border-radius:10px;'
        f'padding:10px 16px;font-size:14px;color:#534ab7;font-weight:600;margin-bottom:12px;">'
        f'🎬 {current_video_name}</div>',
        unsafe_allow_html=True,
    )

    if saved_record and st.session_state.annotation_step == 1:
        st.info("📝 這支影片你已經標過。你可以修改後重新儲存。")

    if saved_record and st.session_state.get("loaded_saved_record_video") != current_video_name:
        reset_feature_widget_state(st.session_state.current_index)
        st.session_state["loaded_saved_record_video"] = current_video_name

    render_progress_banner()

    if st.session_state.annotation_step == 1:
        st.markdown('<div class="section-title">Step 1：選擇主導情緒</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        no_primary_key = f"step1_no_primary_{st.session_state.current_index}"
        selected_emotion_key = f"step1_emotion_{st.session_state.current_index}"

        if no_primary_key not in st.session_state:
            if saved_record:
                saved_has_primary = str(saved_record.get("has_primary_emotion", "True")) == "True"
                st.session_state[no_primary_key] = not saved_has_primary
            else:
                st.session_state[no_primary_key] = False

        if selected_emotion_key not in st.session_state:
            if saved_record and saved_record.get("final_emotion") in MAIN_EMOTIONS:
                st.session_state[selected_emotion_key] = saved_record.get("final_emotion")
            else:
                st.session_state[selected_emotion_key] = None

        no_primary = st.checkbox("是否為無主要情緒", key=no_primary_key)

        if no_primary:
            selected_emotion = "uncertain"
            st.session_state[selected_emotion_key] = "uncertain"
            st.info("已勾選「無主要情緒」，系統將主導情緒強制設為 uncertain。")
        else:
            emotion_options = [e for e in MAIN_EMOTIONS if e != "uncertain"]
            current_value = st.session_state.get(selected_emotion_key)
            radio_index = emotion_options.index(current_value) if current_value in emotion_options else None

            selected_emotion = st.radio(
                "請先選擇這段影片的主導情緒",
                emotion_options,
                index=radio_index,
                key=selected_emotion_key,
            )
          

        st.session_state.step1_has_primary_emotion = (selected_emotion != "uncertain") if selected_emotion else None

        if selected_emotion and selected_emotion in EMOTION_SCHEMA:
            item = EMOTION_SCHEMA[selected_emotion]
            st.markdown(
                f"""
                <div style="background:#f5f4ff;border:1.5px solid #d4d0f5;border-radius:14px;
                padding:16px 20px;margin-top:16px;">
                <div style="font-size:20px;font-weight:800;color:#3c3489;margin-bottom:4px;">
                    {EMOTION_ICONS.get(selected_emotion, "")} {selected_emotion}
                </div>
                <div style="font-size:13px;color:#7366c8;margin-bottom:12px;">定義：{item['definition']}</div>
                """,
                unsafe_allow_html=True,
            )
            if selected_emotion != "uncertain":
                with st.expander("查看此情緒的特徵參考", expanded=True):
                    for grp, opts in item["features"].items():
                        st.markdown(f"- **{grp}**")
                        for opt in opts:
                            st.markdown(f"  - {opt}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("← 返回說明頁", key=f"step1_back_{st.session_state.current_index}"):
                go_to_instruction()
        with c2:
            if st.button("繼續 Step 2 →", key=f"step1_next_{st.session_state.current_index}", disabled=(selected_emotion is None), type="primary"):
                st.session_state.selected_emotion = selected_emotion
                st.session_state.annotation_step = 2
                st.rerun()

    elif st.session_state.annotation_step == 2:
        selected_emotion = st.session_state.selected_emotion
        st.markdown('<div class="section-title">Step 2：選擇各部位特徵</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if not selected_emotion:
            st.warning("請先完成 Step 1 選擇情緒。")
            if st.button("← 返回 Step 1"):
                go_to_step1()
            st.stop()

        all_groups = get_all_step2_groups_union() if selected_emotion == "uncertain" else get_emotion_feature_groups(selected_emotion)[0]
        saved_choices = load_saved_step2_group_choices(saved_record)

        selected_features, unknown_groups = render_single_choice_feature_selector(all_groups, st.session_state.current_index, "step2_single", saved_choices)

        auto_result = evaluate_feature_support(selected_emotion, selected_features, unknown_groups)
        st.session_state.step2_selected_features = selected_features
        st.session_state.step2_unknown_groups = unknown_groups
        st.session_state.step2_result = auto_result

        st.divider()
        if auto_result["confidence"] == "uncertain":
            st.markdown(f'<div class="ok-box"><b>{auto_result["summary"]}</b></div>', unsafe_allow_html=True)
        elif auto_result["feature_count"] >= 1:
            st.markdown(f'<div class="ok-box"><b>{auto_result["summary"]}</b></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warn-box"><b>{auto_result["summary"]}</b></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("← 返回 Step 1", key=f"step2_back_{st.session_state.current_index}"):
                go_to_step1()
        with c2:
            if st.button("繼續 Step 3 →", key=f"step2_next_{st.session_state.current_index}", type="primary"):
                st.session_state.annotation_step = 3
                st.rerun()

    elif st.session_state.annotation_step == 3:
        selected_emotion = st.session_state.selected_emotion
        step2_result = st.session_state.step2_result or {}
        st.markdown('<div class="section-title">Step 3：選擇行為特徵</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if not selected_emotion:
            st.warning("請先完成前面步驟。")
            if st.button("← 返回 Step 1"):
                go_to_step1()
            st.stop()

        behavior_options = get_all_behavior_union() if selected_emotion == "uncertain" else get_emotion_feature_groups(selected_emotion)[1]
        saved_behavior, saved_unknown = load_saved_step3_choice(saved_record)

        selected_behavior, behavior_unknown = render_single_choice_behavior_selector(behavior_options, st.session_state.current_index, saved_behavior, saved_unknown)

        auto_result = evaluate_behavior_support(selected_emotion, step2_result, selected_behavior, behavior_unknown)
        st.session_state.step3_selected_behavior = selected_behavior
        st.session_state.step3_unknown_behavior = behavior_unknown
        st.session_state.step3_result = auto_result

        st.divider()
        if auto_result["confidence"] in ["uncertain", "高"]:
            box_class = "ok-box"
        elif auto_result["confidence"] == "中":
            box_class = "warn-box"
        else:
            box_class = "low-box"
        st.markdown(f'<div class="{box_class}"><b>{auto_result["summary"]}</b></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("← 返回 Step 2", key=f"step3_back_{st.session_state.current_index}"):
                go_to_step2()
        with c2:
            if st.button("繼續 Step 4 →", key=f"step3_next_{st.session_state.current_index}", type="primary"):
                st.session_state.annotation_step = 4
                st.rerun()

    else:
        selected_emotion = st.session_state.selected_emotion
        step2_result = st.session_state.step2_result or {}
        step3_result = st.session_state.step3_result or {}
        final_confidence = step3_result.get("confidence") or "低"

        st.markdown('<div class="section-title">Step 4：最終確認</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        conf_color = {"高": "#edf7ed", "中": "#fff8e1", "低": "#fdecea", "uncertain": "#f5f4ff"}
        conf_border = {"高": "#81c995", "中": "#f0c36d", "低": "#f28b82", "uncertain": "#d4d0f5"}
        conf_icon = {"高": "✅", "中": "⚠️", "低": "❗", "uncertain": "❓"}
        bg = conf_color.get(final_confidence, "#fdecea")
        bd = conf_border.get(final_confidence, "#f28b82")
        ic = conf_icon.get(final_confidence, "❗")

        has_primary_display = "是" if st.session_state.step1_has_primary_emotion else "否"

        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:20px;">
                <div style="background:#f5f4ff;border:1px solid #d4d0f5;border-radius:12px;padding:14px;">
                    <div style="font-size:11px;font-weight:700;color:#9189d0;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">是否有主要情緒</div>
                    <div style="font-size:17px;font-weight:800;color:#3c3489;">{has_primary_display}</div>
                </div>
                <div style="background:#f5f4ff;border:1px solid #d4d0f5;border-radius:12px;padding:14px;">
                    <div style="font-size:11px;font-weight:700;color:#9189d0;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Step 1 情緒</div>
                    <div style="font-size:17px;font-weight:800;color:#3c3489;">{EMOTION_ICONS.get(selected_emotion or '', '')} {selected_emotion or '—'}</div>
                </div>
                <div style="background:#f5f5f5;border:1px solid #e0e0e0;border-radius:12px;padding:14px;">
                    <div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Step 2 特徵</div>
                    <div style="font-size:13px;color:#444;">{step2_result.get('summary', '—')}</div>
                </div>
                <div style="background:{bg};border:1px solid {bd};border-radius:12px;padding:14px;">
                    <div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">信心程度</div>
                    <div style="font-size:17px;font-weight:800;">{ic} {final_confidence}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        final_options = MAIN_EMOTIONS
        default_index = final_options.index(selected_emotion) if selected_emotion in final_options else 0
        selected_final = st.radio("請再次確認最終主導情緒：", final_options, index=default_index, key=f"final_emotion_radio_{st.session_state.current_index}", horizontal=True)

        if selected_final != selected_emotion:
            st.markdown(
                f'<div class="warn-box">⚠️ 和先前情緒不一致：先前為「{selected_emotion}」，目前最終情緒為「{selected_final}」。</div>',
                unsafe_allow_html=True,
            )

        default_note = saved_record.get("note", "") if saved_record else ""
        note = st.text_area("📝 備註（選填）", value=default_note, height=100, key=f"note_{st.session_state.current_index}", placeholder="可在此輸入補充說明…")

        st.divider()

        def build_record(final_emotion: str):
            if not annotator_name:
                st.error("請先在左側輸入標註者姓名或編號。")
                return None
            if not final_emotion:
                st.error("請先選擇最終主導情緒。")
                return None

            selected_features = st.session_state.step2_selected_features or []
            unknown_groups = st.session_state.step2_unknown_groups or []
            selected_emotion_local = st.session_state.selected_emotion

            eye_value, ear_value, tail_value, body_value = split_step2_features_by_group(selected_emotion_local, selected_features)
            behavior_value = st.session_state.step3_selected_behavior[0] if st.session_state.step3_selected_behavior else None

            record = {
                "annotator_name": annotator_name.strip(),
                "video_file": current_video_name,
                "has_primary_emotion": str(final_emotion != "uncertain"),
                "step1_selected_emotion": selected_emotion_local or "",
                "step2_眼睛": json.dumps(eye_value, ensure_ascii=False),
                "step2_耳朵": json.dumps(ear_value, ensure_ascii=False),
                "step2_尾巴": json.dumps(tail_value, ensure_ascii=False),
                "step2_身體": json.dumps(body_value, ensure_ascii=False),
                "step2_眼睛_無法判斷": str("眼睛" in unknown_groups),
                "step2_耳朵_無法判斷": str("耳朵" in unknown_groups),
                "step2_尾巴_無法判斷": str("尾巴" in unknown_groups),
                "step2_身體_無法判斷": str("身體" in unknown_groups),
                "step3_selected_behavior": json.dumps(behavior_value, ensure_ascii=False),
                "step3_unknown_behavior": str(st.session_state.step3_unknown_behavior),
                "final_emotion": final_emotion,
                "confidence": final_confidence,
                "note": note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            return record

        c_back, c_download, c_save = st.columns([2, 3, 3])

        with c_back:
            if st.button("← 返回 Step 2", key=f"step4_back_{st.session_state.current_index}"):
                go_to_step2()

        df_mine = get_annotations_df(annotator_name) if annotator_name else pd.DataFrame()
        current_record = build_record(selected_final)
        if current_record is not None:
            if not df_mine.empty and "video_file" in df_mine.columns:
                preview_df = pd.concat([df_mine[df_mine["video_file"] != current_video_name], pd.DataFrame([current_record])], ignore_index=True)
            else:
                preview_df = pd.DataFrame([current_record])
            csv_bytes = preview_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        else:
            csv_bytes = None

        with c_download:
            st.download_button(
                label="⬇️ 下載我的標註 CSV",
                data=csv_bytes if csv_bytes is not None else b"",
                file_name=f"annotations_{annotator_name.strip() if annotator_name else 'annotations'}.csv",
                mime="text/csv",
                help="按下後會下載到你自己的電腦",
                key=f"download_csv_{st.session_state.current_index}",
                disabled=(csv_bytes is None),
            )

        with c_save:
            if st.button("☁️ 儲存並同步 Google Sheet", key=f"save_sync_{st.session_state.current_index}", type="primary"):
                record = current_record
                if record:
                    upsert_annotation(record, annotator_name)
                    st.session_state.completed = len(get_annotations_df(annotator_name))
                    try:
                        append_to_google_sheet(record, annotator_name)
                        st.success("✅ 已同步到 Google Sheet，並可下載到你的電腦。")
                    except Exception as e:
                        st.warning(f"已保留本次標註資料，下載仍可使用；但同步 Google Sheet 失敗：{e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ 上一段", disabled=st.session_state.current_index == 0, use_container_width=True):
            st.session_state.current_index -= 1
            reset_step_flow()
            st.rerun()
    with col2:
        if st.button("下一段 ▶", disabled=st.session_state.current_index >= len(st.session_state.videos) - 1, use_container_width=True, type="primary"):
            st.session_state.current_index += 1
            reset_step_flow()
            st.rerun()
