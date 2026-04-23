import json
import hashlib
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
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
    .result-box {
        padding: 12px 14px;
        border-radius: 10px;
        background: #eef6ff;
        border: 1px solid #b6d4fe;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .warn-box {
        padding: 12px 14px;
        border-radius: 10px;
        background: #fff8e1;
        border: 1px solid #f0c36d;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .ok-box {
        padding: 12px 14px;
        border-radius: 10px;
        background: #edf7ed;
        border: 1px solid #81c995;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .low-box {
        padding: 12px 14px;
        border-radius: 10px;
        background: #fdecea;
        border: 1px solid #f28b82;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .conflict-box {
        padding: 12px 14px;
        border-radius: 10px;
        background: #f3e5f5;
        border: 1px solid #ce93d8;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    div[data-testid="stRadio"] > label p {
        font-size: 24px !important;
        font-weight: 700 !important;
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

OUTPUT_DIR = Path("annotations")
OUTPUT_DIR.mkdir(exist_ok=True)

MAIN_EMOTIONS = ["害怕", "憤怒/狂怒", "歡樂/玩耍", "滿意", "興趣", "uncertain"]
FEATURE_GROUPS = ["眼睛", "耳朵", "尾巴", "身體", "行為"]

ANNOTATION_RULES = [
    "請先完整觀看，再進行標註",
    "Step 1 先選 1 個主導情緒",
    "Step 2 依據該情緒勾選可觀察到的特徵（眼睛、耳朵、尾巴、身體）",
    "Step 3 再選該情緒下最符合的行為特徵",
    "若某一部位或行為無法觀察，勾選「無法判斷」即可",
    "每段影片僅能標註 1 種主情緒（主導情緒定義為在該片段中出現時間最長且最具代表性的情緒狀態）",
]

EMOTION_SCHEMA = {
    "害怕": {
        "definition": "fear",
        "core_features": {
            "眼睛": [
                "雙眼睜大",
                "瞳孔呈圓形且散大",
                "眨眼",
                "目光向左",
                "半眨眼",
                "緊閉雙眼(行為)",
                "避免眼神接觸(行為)",
            ],
            "耳朵": [
                "向側面",
                "耳廓不可見",
                "背面壓平",
            ],
        },
        "aux_features": {
            "尾巴": [
                "夾在身體下方",
                "或繞在身體旁",
            ],
            "身體": [
                "毛豎起（炸毛）",
                "身體緊繃",
                "發抖",
                "壓低身體",
            ],
            "行為": [
                "高度警戒",
                "受驚反應",
                "顫抖",
                "四肢僵硬",
                "躲藏",
                "逃避／迴避",
                "梳理毛髮",
                "缺乏基本維持行為（進食、飲水、排泄）或不睡眠",
            ],
        },
    },
    "憤怒/狂怒": {
        "definition": "憤怒/狂怒",
        "core_features": {
            "眼睛": [
                "瞳孔呈橢圓形且散大",
                "直視",
            ],
            "耳朵": [
                "向側面旋轉",
                "可見內耳廓",
            ],
        },
        "aux_features": {
            "尾巴": [
                "壓低且僵硬",
                "呈倒 L 形",
                "拍打地面",
                "快速左右或上下甩動",
            ],
            "身體": [
                "毛髮豎立（沿脊椎與尾巴）",
                "身體前傾",
                "臀部抬高",
                "拱背站立",
                "露出牙齒",
            ],
            "行為": [
                "撲向或追逐目標",
                "用爪或口攻擊",
                "驅趕其他個體",
            ],
        },
    },
    "歡樂/玩耍": {
        "definition": "歡樂/玩耍",
        "core_features": {
            "眼睛": [
                "瞳孔因興奮而放大/變圓",
                "瞳孔放鬆/變軟",
            ],
            "耳朵": [
                "直立且面向前方",
            ],
        },
        "aux_features": {
            "尾巴": [
                "垂直",
                "可能呈倒 U 形",
            ],
            "身體": [
                "半張開嘴",
                "拱背",
                "身體姿勢變化多樣",
            ],
            "行為": [
                "運動型遊戲：攀爬",
                "運動型遊戲：奔跑",
                "社交遊戲：接近其他貓",
                "社交遊戲：跳躍",
                "社交遊戲：拍打、撥弄",
                "社交遊戲：用前肢抓住對方",
                "社交遊戲：咬對方",
                "社交遊戲：翻滾或露出腹部",
                "社交遊戲：扭打",
                "社交遊戲：踢擊",
                "社交遊戲：追逐",
                "社交遊戲：側移或逃跑",
                "物件遊戲：站立去抓物體",
                "物件遊戲：拍打物體",
                "物件遊戲：用爪抓住物體",
                "物件遊戲：嗅聞、舔舐",
                "物件遊戲：咬或啃",
                "物件遊戲：丟擲",
                "物件遊戲：與物體扭打",
                "捕獵行為：潛行",
                "捕獵行為：追逐",
                "捕獵行為：跳躍",
                "捕獵行為：撲擊",
            ],
        },
    },
    "滿意": {
        "definition": "滿意",
        "core_features": {
            "眼睛": [
                "瞳孔呈小的縮瞳狀垂直卵圓形",
                "半睜",
            ],
            "耳朵": [
                "直立且面向前方",
            ],
        },
        "aux_features": {
            "尾巴": [
                "放鬆且靜止狀態",
                "輕微彎曲",
            ],
            "身體": [
                "坐著",
                "身體放鬆(趴/躺著)",
                "蜷縮",
            ],
            "行為": [
                "伸展",
                "打哈欠(想睡覺)",
                "自我或互相梳理毛髮",
                "踩踏",
                "親暱行為（碰鼻、頂頭、磨蹭）",
                "翻滾",
                "撒嬌",
                "進食",
                "抓物體",
            ],
        },
    },
    "興趣": {
        "definition": "興趣",
        "core_features": {
            "眼睛": [
                "瞳孔放大/呈圓形",
                "目光向右",
            ],
            "耳朵": [
                "耳朵直立並朝向刺激物",
                "耳朵輕微抖動",
            ],
        },
        "aux_features": {
            "尾巴": [
                "水平",
                "豎起",
            ],
            "身體": [
                "可能用後腳站立",
                "前腳靠在物體上",
                "頭微向前伸",
                "頭向右轉",
            ],
            "行為": [
                "觀察個體或物體",
                "探索環境",
                "嗅聞",
                "舔舐",
                "用爪觸碰",
                "社交接觸（碰鼻、磨蹭）",
                "捕獵行為（潛行、追逐、撲擊、抓取、咬）",
            ],
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
    st.markdown("### 核心特徵")
    for grp, opts in item["core_features"].items():
        st.markdown(f"**{grp}**")
        for opt in opts:
            st.markdown(f"- {opt}")
    st.markdown("### 次要 / 輔助特徵")
    for grp, opts in item["aux_features"].items():
        st.markdown(f"**{grp}**")
        for opt in opts:
            st.markdown(f"- {opt}")


def append_to_google_sheet(record: dict, annotator_name: str):
    payload = {
        **record,
        "annotator_name": annotator_name,
        "secret": SHEET_SECRET,
    }
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


def get_annotation_file(annotator_name: str):
    safe_name = annotator_name.strip() if annotator_name.strip() else "anonymous"
    return OUTPUT_DIR / f"annotations_{safe_name}.csv"


def compute_record_id(annotator_name: str, video_name: str):
    raw = f"{annotator_name}::{video_name}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_existing_annotations(annotator_name: str):
    output_path = get_annotation_file(annotator_name)
    if output_path.exists():
        return pd.read_csv(output_path)
    return pd.DataFrame()


def upsert_annotation(record: dict, annotator_name: str):
    output_path = get_annotation_file(annotator_name)
    df_old = load_existing_annotations(annotator_name)
    df_new = pd.DataFrame([record])

    if not df_old.empty and "record_id" in df_old.columns:
        df_old = df_old[df_old["record_id"] != record["record_id"]]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new if df_old.empty else pd.concat([df_old, df_new], ignore_index=True)

    df_all.to_csv(output_path, index=False, encoding="utf-8-sig")


def get_saved_record(annotator_name: str, video_name: str):
    df = load_existing_annotations(annotator_name)
    if df.empty or "video_file" not in df.columns:
        return None
    matched = df[df["video_file"] == video_name]
    if matched.empty:
        return None
    return matched.iloc[-1].to_dict()


def render_definition_block(emotion_name: str, item: dict):
    with st.expander(f"{emotion_name}", expanded=False):
        img_path = DEFINITION_IMAGE_MAP.get(emotion_name)
        if img_path and img_path.exists():
            st.image(str(img_path), use_container_width=True)
        st.write(f"**定義：** {item['definition']}")
        st.write("**核心特徵**")
        for grp, opts in item["core_features"].items():
            st.markdown(f"- **{grp}**")
            for i, opt in enumerate(opts, start=1):
                st.markdown(f"  - {chr(64 + i)}. {opt}")
        st.write("**次要特徵 / 輔助特徵**")
        for grp, opts in item["aux_features"].items():
            st.markdown(f"- **{grp}**")
            for i, opt in enumerate(opts, start=1):
                st.markdown(f"  - {chr(64 + i)}. {opt}")


def get_emotion_feature_groups(emotion_name: str):
    if emotion_name not in EMOTION_SCHEMA:
        return {}, {}, []
    item = EMOTION_SCHEMA[emotion_name]
    core = item["core_features"]
    aux = item["aux_features"]
    behavior = aux.get("行為", [])
    aux_no_behavior = {k: v for k, v in aux.items() if k != "行為"}
    return core, aux_no_behavior, behavior


def build_saved_selection_map(selected_values, unknown_groups=None):
    selected_map = {feature: True for feature in (selected_values or [])}
    unknown_map = {group: True for group in (unknown_groups or [])}
    return selected_map, unknown_map


def render_grouped_feature_selector(
    title: str,
    group_dict: dict,
    page_index: int,
    prefix: str,
    saved_values: dict,
    saved_unknown_groups: dict,
):
    st.markdown(title)
    selected = []
    unknown_groups = []

    groups = list(group_dict.keys())
    for idx_group, group_name in enumerate(groups):
        features = group_dict.get(group_name, [])
        if not features:
            continue

        unknown_key = f"{prefix}_{page_index}_{group_name}_unknown"
        if unknown_key not in st.session_state:
            st.session_state[unknown_key] = saved_unknown_groups.get(group_name, False)

        feature_keys = []
        for idx, feature in enumerate(features):
            feature_key = f"{prefix}_{page_index}_{group_name}_{idx}_{feature}"
            feature_keys.append((feature, feature_key))
            if feature_key not in st.session_state:
                st.session_state[feature_key] = saved_values.get(feature, False)

        selected_in_group = [
            feature for feature, feature_key in feature_keys
            if st.session_state.get(feature_key, False)
        ]
        unknown_disabled = len(selected_in_group) > 0 and not st.session_state.get(unknown_key, False)

        h1, h2 = st.columns([8, 2])
        with h1:
            st.markdown(f"### {group_name}")
        with h2:
            st.checkbox("無法判斷", key=unknown_key, disabled=unknown_disabled)

        is_unknown = st.session_state.get(unknown_key, False)
        if is_unknown:
            unknown_groups.append(group_name)
            for _, feature_key in feature_keys:
                st.session_state[feature_key] = False
            st.caption(f"{group_name} 已標為無法判斷，因此不能再勾選其他特徵。")

        cols = st.columns(3)
        for idx, (feature, feature_key) in enumerate(feature_keys):
            with cols[idx % 3]:
                checked = st.checkbox(feature, key=feature_key, disabled=is_unknown)
                if checked and not is_unknown:
                    selected.append(feature)

        if idx_group != len(groups) - 1:
            st.divider()

    return selected, unknown_groups


def reset_feature_widget_state(video_index: int):
    prefixes = [
        f"step2_core_{video_index}_",
        f"step2_aux_{video_index}_",
        f"step3_behavior_{video_index}_",
    ]
    keys_to_delete = [
        k for k in list(st.session_state.keys())
        if any(k.startswith(prefix) for prefix in prefixes)
    ]
    for k in keys_to_delete:
        del st.session_state[k]


def clear_step4_state(video_index: int):
    for key in [
        f"final_emotion_radio_{video_index}",
        f"inconsistency_confirm_{video_index}",
        f"note_{video_index}",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def clear_step3_state(video_index: int):
    for key in [
        f"behavior_unknown_{video_index}",
        f"behavior_single_{video_index}",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def reset_step_flow():
    video_index = st.session_state.current_index
    st.session_state.annotation_step = 1
    st.session_state.selected_emotion = None
    st.session_state.step2_selected_features = []
    st.session_state.step2_unknown_groups = []
    st.session_state.step2_result = None
    st.session_state.step3_selected_behavior = []
    st.session_state.step3_unknown_behavior = False
    st.session_state.step3_result = None
    st.session_state.loaded_saved_record_video = None
    st.session_state.inconsistency_confirm = None
    reset_feature_widget_state(video_index)
    clear_step3_state(video_index)
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
    clear_step3_state(st.session_state.current_index)
    clear_step4_state(st.session_state.current_index)
    st.rerun()


def go_to_step2():
    st.session_state.annotation_step = 2
    st.session_state.step3_result = None
    clear_step4_state(st.session_state.current_index)
    st.rerun()


def evaluate_feature_support(selected_emotion, selected_features, unknown_groups):
    if selected_emotion == "uncertain":
        return {
            "emotion": "uncertain",
            "core_count": 0,
            "aux_count": 0,
            "selected_count": 0,
            "confidence": "低等",
            "summary": "已選擇 uncertain，特徵步驟僅作記錄。",
        }

    if not selected_emotion or selected_emotion not in EMOTION_SCHEMA:
        return {
            "emotion": None,
            "core_count": 0,
            "aux_count": 0,
            "selected_count": 0,
            "confidence": "低等",
            "summary": "尚未選擇情緒。",
        }

    item = EMOTION_SCHEMA[selected_emotion]
    core_pool = set()
    aux_pool = set()
    for group, feats in item["core_features"].items():
        core_pool.update(feats)
    for group, feats in item["aux_features"].items():
        if group != "行為":
            aux_pool.update(feats)

    core_count = sum(1 for f in selected_features if f in core_pool)
    aux_count = sum(1 for f in selected_features if f in aux_pool)
    selected_count = len(selected_features)

    if core_count >= 2:
        confidence = "中等"
        summary = "✅ 已有 2 個以上核心特徵支持此情緒。"
    elif core_count >= 1 and aux_count >= 1:
        confidence = "中等"
        summary = "✅ 已有 1 個核心特徵 + 1 個次要特徵支持此情緒。"
    elif selected_count >= 1:
        confidence = "低等"
        summary = "⚠️ 已選情緒，但目前特徵支持仍偏弱。"
    else:
        confidence = "低等"
        summary = "⚠️ 尚未勾選任何可支持此情緒的特徵。"

    if unknown_groups:
        summary += f"（無法判斷部位：{', '.join(unknown_groups)}）"

    return {
        "emotion": selected_emotion,
        "core_count": core_count,
        "aux_count": aux_count,
        "selected_count": selected_count,
        "confidence": confidence,
        "summary": summary,
    }


def evaluate_behavior_support(selected_emotion, step2_result, selected_behavior, unknown_behavior):
    if selected_emotion == "uncertain":
        return {
            "emotion": "uncertain",
            "confidence": "低等",
            "summary": "最終情緒為 uncertain，行為步驟僅作記錄。",
            "matched": False,
        }

    if unknown_behavior:
        return {
            "emotion": selected_emotion,
            "confidence": step2_result.get("confidence", "低等"),
            "summary": "行為無法判斷，維持前一步信心。",
            "matched": False,
        }

    if not selected_behavior:
        return {
            "emotion": selected_emotion,
            "confidence": step2_result.get("confidence", "低等"),
            "summary": "尚未選擇行為，維持前一步信心。",
            "matched": False,
        }

    behavior = selected_behavior[0]
    behavior_pool = set(EMOTION_SCHEMA[selected_emotion]["aux_features"].get("行為", []))
    matched = behavior in behavior_pool

    if matched:
        if step2_result.get("confidence") == "中等":
            confidence = "高等"
            summary = "✅ 行為與已選情緒一致，信心提升為高等。"
        else:
            confidence = "中等"
            summary = "✅ 行為與已選情緒一致。"
    else:
        confidence = step2_result.get("confidence", "低等")
        summary = "⚠️ 所選行為與此情緒定義不一致，請再確認。"

    return {
        "emotion": selected_emotion,
        "confidence": confidence,
        "summary": summary,
        "matched": matched,
    }


def init_session(videos):
    defaults = {
        "page": "instruction",
        "current_index": 0,
        "videos": videos,
        "completed": 0,
        "annotation_step": 1,
        "selected_emotion": None,
        "step2_selected_features": [],
        "step2_unknown_groups": [],
        "step2_result": None,
        "step3_selected_behavior": [],
        "step3_unknown_behavior": False,
        "step3_result": None,
        "loaded_saved_record_video": None,
        "inconsistency_confirm": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_progress_banner():
    step = st.session_state.annotation_step
    labels = ["Step 1：情緒", "Step 2：特徵", "Step 3：行為", "Step 4：最終確認"]
    step_map = {1: 0, 2: 1, 3: 2, 4: 3}
    current_label_idx = step_map.get(step, 0)

    cols = st.columns(4)
    for i, label in enumerate(labels):
        with cols[i]:
            if i < current_label_idx:
                st.markdown(f"✅ **{label}**")
            elif i == current_label_idx:
                st.markdown(f"▶️ **{label}**")
            else:
                st.markdown(f"⬜ {label}")

    st.divider()


st.title(APP_TITLE)
st.caption("流程：Step 1 選情緒 → Step 2 選特徵 → Step 3 選行為 → Step 4 最終確認")

videos = load_video_files()
init_session(videos)

with st.sidebar:
    st.header("標註進度")
    st.write(f"目前影片數：{len(st.session_state.videos)}")
    st.write(f"目前索引：{st.session_state.current_index + 1 if st.session_state.videos else 0}")
    st.write(f"已完成：{st.session_state.completed}")

    annotator_name = st.text_input(
        "標註者姓名 / 編號",
        value=st.session_state.get("annotator_name", "")
    )
    st.session_state["annotator_name"] = annotator_name

    if st.button("回到說明頁"):
        go_to_instruction()

    if (
        st.session_state.page == "annotation"
        and len(st.session_state.videos) > 0
        and st.session_state.current_index < len(st.session_state.videos)
    ):
        sidebar_video = st.session_state.videos[st.session_state.current_index]
        st.markdown("---")
        st.caption(f"影片：{get_video_name(sidebar_video)}")
        st.caption(f"影片索引：{st.session_state.current_index + 1}")
        render_small_video(sidebar_video)

    st.markdown("---")
    st.markdown("**情緒定義快速查看**")
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
            if st.button(
                button_labels.get(emotion_name, emotion_name),
                key=f"sidebar_emotion_{emotion_name}",
                use_container_width=True,
            ):
                show_emotion_dialog(emotion_name)


if st.session_state.page == "instruction":
    st.subheader("一、標註規則")
    for i, rule in enumerate(ANNOTATION_RULES, start=1):
        st.markdown(f"{i}. {rule}")

    st.subheader("二、情緒定義與判斷參考")
    for emo_name, emo_item in EMOTION_SCHEMA.items():
        render_definition_block(emo_name, emo_item)

    st.info("請先完整閱讀以上規則與定義，再開始標註。")
    start_disabled = (not annotator_name) or (len(st.session_state.videos) == 0)
    if len(st.session_state.videos) == 0:
        st.warning("目前找不到影片。請先把影片放進專案根目錄下的 videos/ 資料夾。")

    if st.button("我已閱讀完畢，開始標註", disabled=start_disabled):
        st.session_state.page = "annotation"
        reset_step_flow()
        st.rerun()

else:
    if len(st.session_state.videos) == 0:
        st.error("沒有可標註的影片，請先把影片檔放到 videos/ 資料夾。")
        st.stop()

    if st.session_state.current_index >= len(st.session_state.videos):
        st.success("所有影片都標註完成了。")
        annotator_name = st.session_state.get("annotator_name", "anonymous")
        output_path = get_annotation_file(annotator_name)
        if output_path.exists():
            df = pd.read_csv(output_path)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "下載標註結果 CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name=output_path.name,
                mime="text/csv",
            )
        st.stop()

    current_video = st.session_state.videos[st.session_state.current_index]
    current_video_name = get_video_name(current_video)
    saved_record = get_saved_record(annotator_name, current_video_name) if annotator_name else None

    st.subheader(f"目前影片：{current_video_name}")

    if saved_record and st.session_state.annotation_step == 1:
        st.info("這支影片你已經標過。你可以修改後重新儲存，系統會覆蓋舊資料。")

    if saved_record and st.session_state.get("loaded_saved_record_video") != current_video_name:
        reset_feature_widget_state(st.session_state.current_index)
        st.session_state["loaded_saved_record_video"] = current_video_name

    render_progress_banner()

    # ----------------------------
    # Step 1：先選情緒
    # ----------------------------
    if st.session_state.annotation_step == 1:
        st.markdown("## Step 1：先選主導情緒")

        selected_emotion_key = f"step1_emotion_{st.session_state.current_index}"
        if selected_emotion_key not in st.session_state:
            if saved_record and saved_record.get("final_emotion") in MAIN_EMOTIONS:
                st.session_state[selected_emotion_key] = saved_record.get("final_emotion")
            else:
                st.session_state[selected_emotion_key] = st.session_state.selected_emotion

        selected_emotion = st.radio(
            "請先選擇這段影片的主導情緒",
            MAIN_EMOTIONS,
            index=MAIN_EMOTIONS.index(st.session_state[selected_emotion_key])
            if st.session_state[selected_emotion_key] in MAIN_EMOTIONS else None,
            key=selected_emotion_key,
        )

        if selected_emotion and selected_emotion in EMOTION_SCHEMA:
            st.markdown("---")
            st.markdown(f"### 你選擇的情緒：{selected_emotion}")
            item = EMOTION_SCHEMA[selected_emotion]
            st.markdown(f"**定義：** {item['definition']}")

            with st.expander("查看此情緒的特徵參考", expanded=True):
                st.markdown("**核心特徵**")
                for grp, opts in item["core_features"].items():
                    st.markdown(f"- **{grp}**")
                    for opt in opts:
                        st.markdown(f"  - {opt}")

                st.markdown("**次要 / 輔助特徵**")
                for grp, opts in item["aux_features"].items():
                    st.markdown(f"- **{grp}**")
                    for opt in opts:
                        st.markdown(f"  - {opt}")

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            if st.button("返回說明頁", key=f"step1_back_{st.session_state.current_index}"):
                go_to_instruction()

        with c2:
            if st.button("繼續 → Step 2", key=f"step1_next_{st.session_state.current_index}", disabled=not selected_emotion):
                st.session_state.selected_emotion = selected_emotion
                st.session_state.annotation_step = 2
                st.rerun()

    # ----------------------------
    # Step 2：再選特徵
    # ----------------------------
    elif st.session_state.annotation_step == 2:
        selected_emotion = st.session_state.selected_emotion
        st.markdown(f"## Step 2：選擇「{selected_emotion}」對應的特徵")

        if not selected_emotion:
            st.warning("請先完成 Step 1 選擇情緒。")
            if st.button("返回 Step 1"):
                go_to_step1()
            st.stop()

        if selected_emotion == "uncertain":
            st.info("你已選擇 uncertain，因此特徵步驟可略過；若仍想補充，也可以保留空白直接進下一步。")
            st.session_state.step2_result = evaluate_feature_support("uncertain", [], [])
            c1, c2 = st.columns(2)
            with c1:
                if st.button("返回上一步", key=f"step2_back_uncertain_{st.session_state.current_index}"):
                    go_to_step1()
            with c2:
                if st.button("繼續 → Step 3", key=f"step2_next_uncertain_{st.session_state.current_index}"):
                    st.session_state.annotation_step = 3
                    st.rerun()

        else:
            core_groups, aux_groups, _ = get_emotion_feature_groups(selected_emotion)

            default_values, default_unknown = build_saved_selection_map(
                st.session_state.step2_selected_features,
                st.session_state.step2_unknown_groups,
            )

            selected_core, unknown_core = render_grouped_feature_selector(
                "### 核心特徵",
                core_groups,
                st.session_state.current_index,
                "step2_core",
                default_values,
                default_unknown,
            )

            st.divider()

            selected_aux, unknown_aux = render_grouped_feature_selector(
                "### 次要特徵",
                aux_groups,
                st.session_state.current_index,
                "step2_aux",
                default_values,
                default_unknown,
            )

            selected_features = selected_core + selected_aux
            unknown_groups = unknown_core + unknown_aux

            st.divider()
            c1, c2, c3 = st.columns(3)

            with c1:
                if st.button("返回上一步", key=f"step2_back_{st.session_state.current_index}"):
                    go_to_step1()

            with c2:
                if st.button("檢查 Step 2", key=f"step2_check_{st.session_state.current_index}"):
                    st.session_state.step2_selected_features = selected_features
                    st.session_state.step2_unknown_groups = unknown_groups
                    st.session_state.step2_result = evaluate_feature_support(
                        selected_emotion,
                        selected_features,
                        unknown_groups,
                    )
                    st.rerun()

            result2 = st.session_state.step2_result
            can_continue = False
            if result2:
                if result2["confidence"] == "中等":
                    st.markdown(
                        f'<div class="ok-box"><b>{result2["summary"]}</b></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="warn-box"><b>{result2["summary"]}</b></div>',
                        unsafe_allow_html=True,
                    )
                can_continue = True

            with c3:
                if st.button(
                    "繼續 → Step 3",
                    key=f"step2_next_{st.session_state.current_index}",
                    disabled=not can_continue,
                ):
                    st.session_state.annotation_step = 3
                    st.rerun()

    # ----------------------------
    # Step 3：再選行為
    # ----------------------------
    elif st.session_state.annotation_step == 3:
        selected_emotion = st.session_state.selected_emotion
        step2_result = st.session_state.step2_result or {}

        st.markdown(f"## Step 3：選擇「{selected_emotion}」對應的行為")

        if not selected_emotion:
            st.warning("請先完成前面步驟。")
            if st.button("返回 Step 1"):
                go_to_step1()
            st.stop()

        _, _, behavior_options = get_emotion_feature_groups(selected_emotion) if selected_emotion in EMOTION_SCHEMA else ({}, {}, [])

        behavior_key = f"behavior_single_{st.session_state.current_index}"
        behavior_unknown_key = f"behavior_unknown_{st.session_state.current_index}"

        if behavior_key not in st.session_state:
            if st.session_state.step3_selected_behavior:
                st.session_state[behavior_key] = st.session_state.step3_selected_behavior[0]
            else:
                st.session_state[behavior_key] = None

        if behavior_unknown_key not in st.session_state:
            st.session_state[behavior_unknown_key] = st.session_state.step3_unknown_behavior

        if selected_emotion == "uncertain":
            st.info("你已選擇 uncertain，行為步驟僅作補充記錄，可直接進入最終確認。")
            st.session_state.step3_result = evaluate_behavior_support("uncertain", step2_result, [], False)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("返回上一步", key=f"step3_back_uncertain_{st.session_state.current_index}"):
                    go_to_step2()
            with c2:
                if st.button("繼續 → Step 4", key=f"step3_next_uncertain_{st.session_state.current_index}"):
                    st.session_state.annotation_step = 4
                    st.rerun()

        else:
            col_u1, col_u2 = st.columns([8, 2])
            with col_u1:
                st.markdown("### 行為")
            with col_u2:
                behavior_unknown = st.checkbox("無法判斷", key=behavior_unknown_key)

            if behavior_unknown:
                st.session_state[behavior_key] = None
                selected_behavior = []
            else:
                selected_behavior_value = st.radio(
                    "請選擇一個最主要的行為特徵",
                    behavior_options,
                    index=behavior_options.index(st.session_state[behavior_key])
                    if st.session_state[behavior_key] in behavior_options else None,
                    key=behavior_key,
                )
                selected_behavior = [selected_behavior_value] if selected_behavior_value else []

            st.divider()
            c1, c2, c3 = st.columns(3)

            with c1:
                if st.button("返回上一步", key=f"step3_back_{st.session_state.current_index}"):
                    go_to_step2()

            with c2:
                if st.button("檢查 Step 3", key=f"step3_check_{st.session_state.current_index}"):
                    st.session_state.step3_selected_behavior = selected_behavior
                    st.session_state.step3_unknown_behavior = behavior_unknown
                    st.session_state.step3_result = evaluate_behavior_support(
                        selected_emotion,
                        step2_result,
                        selected_behavior,
                        behavior_unknown,
                    )
                    st.rerun()

            result3 = st.session_state.step3_result
            can_continue = False
            if result3:
                if result3["confidence"] == "高等":
                    st.markdown(
                        f'<div class="ok-box"><b>{result3["summary"]}</b></div>',
                        unsafe_allow_html=True,
                    )
                elif result3["confidence"] == "中等":
                    st.markdown(
                        f'<div class="warn-box"><b>{result3["summary"]}</b></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="low-box"><b>{result3["summary"]}</b></div>',
                        unsafe_allow_html=True,
                    )
                can_continue = True

            with c3:
                if st.button(
                    "繼續 → Step 4",
                    key=f"step3_next_{st.session_state.current_index}",
                    disabled=not can_continue,
                ):
                    st.session_state.annotation_step = 4
                    st.rerun()

    # ----------------------------
    # Step 4：最終確認
    # ----------------------------
    else:
        selected_emotion = st.session_state.selected_emotion
        step2_result = st.session_state.step2_result or {}
        step3_result = st.session_state.step3_result or {}

        final_confidence = step3_result.get("confidence") or step2_result.get("confidence") or "低等"

        st.markdown("## Step 4：最終情緒確認")
        st.markdown(f"- **Step 1 選擇情緒：** {selected_emotion or '—'}")
        st.markdown(f"- **Step 2 結果：** {step2_result.get('summary', '—')}")
        st.markdown(f"- **Step 3 結果：** {step3_result.get('summary', '—')}")

        if final_confidence == "高等":
            st.markdown(
                '<div class="ok-box"><b>信心程度：高等</b></div>',
                unsafe_allow_html=True,
            )
        elif final_confidence == "中等":
            st.markdown(
                '<div class="warn-box"><b>信心程度：中等</b></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="low-box"><b>信心程度：低等</b></div>',
                unsafe_allow_html=True,
            )

        final_options = ["uncertain"] if selected_emotion == "uncertain" else MAIN_EMOTIONS
        default_index = final_options.index(selected_emotion) if selected_emotion in final_options else 0

        selected_final = st.radio(
            "請選擇最終主導情緒",
            final_options,
            index=default_index,
            key=f"final_emotion_radio_{st.session_state.current_index}",
        )

        inconsistency_msg = None
        if selected_final and selected_emotion and selected_final != selected_emotion:
            inconsistency_msg = f"你最終選擇的情緒「{selected_final}」與 Step 1 選擇的情緒「{selected_emotion}」不一致。"

        if inconsistency_msg:
            st.markdown(
                f'<div class="warn-box">⚠️ {inconsistency_msg}</div>',
                unsafe_allow_html=True,
            )

        default_note = saved_record.get("note", "") if saved_record else ""
        note = st.text_area(
            "備註",
            value=default_note,
            height=120,
            key=f"note_{st.session_state.current_index}",
        )

        st.divider()

        c_back, _ = st.columns([1, 3])
        with c_back:
            if st.button("返回上一步", key=f"step4_back_{st.session_state.current_index}"):
                go_to_step2()

        def build_record(final_emotion: str):
            if not annotator_name:
                st.error("請先在左側輸入標註者姓名或編號。")
                return None

            if not final_emotion:
                st.error("請先選擇最終主導情緒。")
                return None

            selected_emotion_local = st.session_state.selected_emotion
            if selected_emotion_local in EMOTION_SCHEMA:
                core_groups, aux_groups, _ = get_emotion_feature_groups(selected_emotion_local)
                core_selected = [
                    f for f in st.session_state.step2_selected_features
                    if any(f in feats for feats in core_groups.values())
                ]
                aux_selected = [
                    f for f in st.session_state.step2_selected_features
                    if any(f in feats for feats in aux_groups.values())
                ]
            else:
                core_selected = []
                aux_selected = []

            record = {
                "record_id": compute_record_id(annotator_name.strip(), current_video_name),
                "video_file": current_video_name,
                "step1_selected_emotion": selected_emotion_local or "",
                "step2_selected_features": json.dumps(st.session_state.step2_selected_features, ensure_ascii=False),
                "step2_unknown_groups": json.dumps(st.session_state.step2_unknown_groups, ensure_ascii=False),
                "step2_core_selected": json.dumps(core_selected, ensure_ascii=False),
                "step2_aux_selected": json.dumps(aux_selected, ensure_ascii=False),
                "step3_selected_behavior": json.dumps(st.session_state.step3_selected_behavior, ensure_ascii=False),
                "step3_unknown_behavior": str(st.session_state.step3_unknown_behavior),
                "final_emotion": final_emotion,
                "final_matches_step1": str(final_emotion == (selected_emotion_local or "")),
                "confidence": final_confidence,
                "step2_summary": step2_result.get("summary", ""),
                "step3_summary": step3_result.get("summary", ""),
                "note": note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            return record

        confirm_key = f"inconsistency_confirm_{st.session_state.current_index}"
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = None

        if st.session_state.get(confirm_key) == "pending":
            st.markdown("---")
            st.markdown(
                f'<div class="warn-box">'
                f'<b>⚠️ 標註不一致確認</b><br>'
                f'{inconsistency_msg}<br><br>'
                f'請確認你的標註是否正確？'
                f'</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ 是，確認標註無誤", use_container_width=True, key=f"confirm_yes_{st.session_state.current_index}"):
                    st.session_state[confirm_key] = "confirmed"
                    st.rerun()
            with c2:
                if st.button("❌ 否，重新標註", use_container_width=True, key=f"confirm_no_{st.session_state.current_index}"):
                    st.session_state[confirm_key] = None
                    go_to_step1()

        else:
            c1, c2 = st.columns(2)

            with c1:
                if annotator_name:
                    df_mine = load_existing_annotations(annotator_name)
                    if not df_mine.empty:
                        csv_bytes = df_mine.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                        st.download_button(
                            label="⬇️ 下載我的標註 CSV",
                            data=csv_bytes,
                            file_name=f"annotations_{annotator_name.strip()}.csv",
                            mime="text/csv",
                            help="只包含你自己的標註資料",
                        )

            with c2:
                if st.button("☁️ 儲存並同步 Google Sheet", key=f"save_sync_{st.session_state.current_index}"):
                    if inconsistency_msg and st.session_state[confirm_key] != "confirmed":
                        st.session_state[confirm_key] = "pending"
                        st.rerun()
                    else:
                        record = build_record(selected_final)
                        if record:
                            upsert_annotation(record, annotator_name)
                            st.session_state.completed = len(load_existing_annotations(annotator_name))
                            try:
                                append_to_google_sheet(record, annotator_name)
                                st.success("✅ 已儲存到本地，並同步到 Google Sheet。")
                            except Exception as e:
                                st.warning(f"本地已儲存，但同步 Google Sheet 失敗：{e}")
                            st.session_state[confirm_key] = None

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("上一段", disabled=st.session_state.current_index == 0):
            st.session_state.current_index -= 1
            reset_step_flow()
            st.rerun()

    with col2:
        if st.button("下一段", disabled=st.session_state.current_index >= len(st.session_state.videos) - 1):
            st.session_state.current_index += 1
            reset_step_flow()
            st.rerun()
