import json
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import requests
import io

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
VIDEOS = [
    {
        "name": "v1__s000000__e000010.mp4",
        "url": "https://storage.googleapis.com/cat-emotion-videos-fuyu/videos/v1__s000000__e000010.mp4",
    },
    {
        "name": "v2__s000000__e000009.mp4",
        "url": "https://storage.googleapis.com/cat-emotion-videos-fuyu/videos/v2__s000000__e000009.mp4",
    },
]
OUTPUT_DIR = Path("annotations")
OUTPUT_DIR.mkdir(exist_ok=True)

MAIN_EMOTIONS = ["害怕", "憤怒/狂怒", "歡樂/玩耍", "滿意", "興趣", "uncertain"]
FEATURE_GROUPS = ["眼睛", "耳朵", "尾巴", "四肢", "行為"]

ANNOTATION_RULES = [
    "請先完整觀看，再進行標註",
    "每一段影片只選 1 類主情緒（主導情緒定義為在該片段中出現時間最長且最具代表性的情緒狀態）",
    "標註流程分為三步：Step 1 眼睛/耳朵；Step 2 尾巴/四肢；Step 3 行為（補充 或 輔助判定）",
    "每個步驟需依據可觀察特徵勾選，不憑主觀猜測",
    "若某一部位無法觀察，勾選「無法判斷」即可",
]

EMOTION_SCHEMA = {
    "害怕": {
        "definition": "由立即感知到的危險或危險威脅引起，表現為警惕、撤退或逃跑。",
        "core_features": {
            "眼睛": [
                "雙眼睜大",
                "瞳孔呈圓形且散大",
                "眨眼／半眨眼／緊閉雙眼／避免眼神接觸",
                "輕微恐懼時目光偏左",
            ],
            "耳朵": ["向側面或背面壓平", "耳廓不可見"],
        },
        "aux_features": {
            "尾巴": ["夾在身體下方", "繞在身體旁"],
            "四肢": ["毛豎起（炸毛）", "身體緊繃", "發抖", "壓低身體"],
            "行為": [
                "高度警戒",
                "受驚反應",
                "僵住不動",
                "躲藏",
                "逃避／迴避",
                "梳理毛髮",
                "缺乏進食、飲水、排泄或睡眠等基本維持行為",
            ],
        },
    },
    "憤怒/狂怒": {
        "definition": "由目標受挫、逃避受阻、探索受阻或資源競爭引起，表現為攻擊性或攻擊威脅。",
        "core_features": {
            "眼睛": ["瞳孔呈橢圓形且散大", "直視目標"],
            "耳朵": ["向側面旋轉", "可見內耳廓"],
        },
        "aux_features": {
            "尾巴": ["壓低且僵硬", "呈倒 L 形", "拍打地面", "快速左右或上下甩動（tail lash）"],
            "四肢": ["毛髮豎立（沿脊椎與尾巴）", "身體前傾", "臀部抬高", "拱背站立", "露出牙齒"],
            "行為": ["撲向或追逐目標", "用爪或口攻擊", "驅趕其他個體"],
        },
    },
    "歡樂/玩耍": {
        "definition": "表現為非功能性行為，包括運動遊戲、社交遊戲或與物體互動的遊戲。",
        "core_features": {
            "眼睛": ["瞳孔因興奮而放大／變圓", "瞳孔放鬆／變軟"],
            "耳朵": ["直立且面向前方"],
        },
        "aux_features": {
            "尾巴": ["垂直", "可能呈倒 U 形"],
            "四肢": ["可能出現玩耍表情（半張開嘴）", "拱背", "身體姿勢變化多樣"],
            "行為": [
                "攀爬、奔跑",
                "跳躍、拍打、撥弄",
                "用前肢抓住對方或物體",
                "翻滾、露腹、扭打、踢擊、追逐",
                "嗅聞、舔舐、咬、啃、丟擲物體",
                "潛行、撲擊等遊戲性捕獵行為",
            ],
        },
    },
    "滿意": {
        "definition": "由需求和願望得到滿足，以及對當前狀態的接受而產生，表現為休息、平靜與親和。",
        "core_features": {
            "眼睛": ["瞳孔呈較小、縮瞳的垂直卵圓形", "半睜"],
            "耳朵": ["直立且面向前方"],
        },
        "aux_features": {
            "尾巴": ["放鬆、靜止", "或輕微彎曲"],
            "四肢": ["坐著或蜷縮", "身體放鬆"],
            "行為": [
                "伸展",
                "打哈欠",
                "自我或互相梳理毛髮",
                "踩踏",
                "親暱行為（碰鼻、頂頭、磨蹭）",
                "翻滾、撒嬌",
                "進食",
                "抓物體",
            ],
        },
    },
    "興趣": {
        "definition": "由新奇或顯著刺激引起，表現為對刺激的注意、定向或尋求行為。",
        "core_features": {
            "眼睛": ["瞳孔放大／呈圓形", "目光偏右", "觀察某人或某物"],
            "耳朵": ["耳朵直立並朝向刺激物", "耳朵輕微抖動"],
        },
        "aux_features": {
            "尾巴": ["常見水平", "友善時可能豎起"],
            "四肢": ["可能用後腳站立", "前腳靠在物體上", "頭向前伸", "頭向右轉"],
            "行為": [
                "觀察個體或物體",
                "探索環境",
                "嗅聞、舔舐",
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


@st.dialog("⚠️ 標註確認")
def show_inconsistency_dialog(message: str, on_confirm_key: str):
    st.warning(message)
    st.markdown("**請確認你的標註是否正確？**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 是，確認標註無誤", use_container_width=True):
            st.session_state[on_confirm_key] = "confirmed"
            st.rerun()
    with col2:
        if st.button("❌ 否，重新標註", use_container_width=True):
            st.session_state[on_confirm_key] = "redo"
            st.rerun()


SHEET_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbzA_0AnSkFSeN6GFLr1wDsvx-l28-5a3s605l9CV6QwwTfcJ4GejNepx2yOIjX7M85m/exec"
SHEET_SECRET = "my_cat_annotation_secret"


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


def parse_json_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, float) and pd.isna(value):
        return []
    try:
        return json.loads(value)
    except Exception:
        return []


def build_neutral_feature_catalog(schema: dict):
    catalog = {"core": {}, "aux": {}}
    for emotion, item in schema.items():
        for group, options in item["core_features"].items():
            for feature in options:
                if feature not in catalog["core"]:
                    catalog["core"][feature] = {"groups": set(), "emotions": set()}
                catalog["core"][feature]["groups"].add(group)
                catalog["core"][feature]["emotions"].add(emotion)
        for group, options in item["aux_features"].items():
            for feature in options:
                if feature not in catalog["aux"]:
                    catalog["aux"][feature] = {"groups": set(), "emotions": set()}
                catalog["aux"][feature]["groups"].add(group)
                catalog["aux"][feature]["emotions"].add(emotion)
    return catalog


def build_feature_emotion_lookup(schema: dict):
    lookup = {"core": {}, "aux": {}}
    for emotion, item in schema.items():
        for group, options in item["core_features"].items():
            for feature in options:
                lookup["core"][feature] = {"emotion": emotion, "group": group}
        for group, options in item["aux_features"].items():
            for feature in options:
                lookup["aux"][feature] = {"emotion": emotion, "group": group}
    return lookup


def group_features_for_display(feature_catalog: dict, feature_type: str):
    grouped = {group: [] for group in FEATURE_GROUPS}
    for feature, meta in feature_catalog[feature_type].items():
        groups = sorted(list(meta.get("groups", [])))
        for group in FEATURE_GROUPS:
            if group in groups:
                grouped[group].append(feature)
                break
    return {k: sorted(v) for k, v in grouped.items() if v}


def build_feature_saved_values(selected_values, unknown_groups=None):
    selected_map = {feature: True for feature in (selected_values or [])}
    unknown_group_map = {group: True for group in (unknown_groups or [])}
    return selected_map, unknown_group_map


def render_feature_checkbox_grid(
    title: str,
    feature_type: str,
    groups_to_show: list,
    feature_catalog: dict,
    saved_values: dict,
    saved_unknown_groups: dict,
    page_index: int,
):
    st.markdown(title)
    selected = []
    unknown_groups = []
    grouped_features = group_features_for_display(feature_catalog, feature_type)

    for idx_group, group_name in enumerate(groups_to_show):
        features = grouped_features.get(group_name, [])
        if not features:
            continue

        unknown_key = f"{feature_type}_{page_index}_{group_name}_unknown_group"
        if unknown_key not in st.session_state:
            st.session_state[unknown_key] = saved_unknown_groups.get(group_name, False)

        feature_keys = []
        for idx, feature in enumerate(features):
            feature_key = f"{feature_type}_{page_index}_{group_name}_{idx}_{feature}_selected"
            feature_keys.append((feature, feature_key))
            if feature_key not in st.session_state:
                st.session_state[feature_key] = saved_values.get(feature, False)

        is_group_unknown = st.session_state.get(unknown_key, False)
        selected_in_group = [
            feature for feature, feature_key in feature_keys
            if st.session_state.get(feature_key, False)
        ]
        unknown_disabled = len(selected_in_group) > 0 and not is_group_unknown

        header_col1, header_col2 = st.columns([8, 2])
        with header_col1:
            st.markdown(f"### {group_name}")
        with header_col2:
            st.checkbox("無法判斷", key=unknown_key, disabled=unknown_disabled)

        is_group_unknown = st.session_state.get(unknown_key, False)

        if is_group_unknown:
            unknown_groups.append(group_name)
            for _, feature_key in feature_keys:
                st.session_state[feature_key] = False
            st.caption(f"{group_name}已標為無法判斷，因此不能再勾選其他特徵。")

        cols = st.columns(3)
        for idx, (feature, feature_key) in enumerate(feature_keys):
            with cols[idx % 3]:
                is_selected = st.checkbox(
                    feature,
                    key=feature_key,
                    disabled=is_group_unknown,
                )
                if is_selected and not is_group_unknown:
                    selected.append(feature)

        if idx_group != len(groups_to_show) - 1:
            st.divider()

    return selected, unknown_groups


def infer_group_emotion(selected_features, unknown_groups, feature_type, group_name):
    if group_name in unknown_groups:
        return {"status": "unknown", "emotion": None, "all_emotions": {}}

    group_features = [
        f for f in selected_features
        if FEATURE_LOOKUP[feature_type].get(f, {}).get("group") == group_name
    ]

    if not group_features:
        return {"status": "unknown", "emotion": None, "all_emotions": {}}

    counts = Counter(
        FEATURE_LOOKUP[feature_type][f]["emotion"] for f in group_features
    )
    all_emotions = dict(counts)
    top_count = max(counts.values())
    winners = [emo for emo, c in counts.items() if c == top_count]

    if len(winners) == 1:
        return {"status": "emotion", "emotion": winners[0], "all_emotions": all_emotions}
    return {"status": "conflict", "emotion": None, "all_emotions": all_emotions}


def get_group_selected_features(selected_features, feature_type, group_name):
    return [
        f for f in selected_features
        if FEATURE_LOOKUP[feature_type].get(f, {}).get("group") == group_name
    ]


def evaluate_step12(
    selected_core_all, unknown_core_groups,
    selected_aux_all, unknown_aux_groups
):
    eye = infer_group_emotion(selected_core_all, unknown_core_groups, "core", "眼睛")
    ear = infer_group_emotion(selected_core_all, unknown_core_groups, "core", "耳朵")
    tail = infer_group_emotion(selected_aux_all, unknown_aux_groups, "aux", "尾巴")
    limb = infer_group_emotion(selected_aux_all, unknown_aux_groups, "aux", "四肢")

    core_candidates = defaultdict(int)
    if eye["status"] == "emotion":
        core_candidates[eye["emotion"]] += 1
    if ear["status"] == "emotion":
        core_candidates[ear["emotion"]] += 1
    core_candidates = dict(core_candidates)

    secondary_supports = defaultdict(int)
    if tail["status"] == "emotion":
        secondary_supports[tail["emotion"]] += 1
    if limb["status"] == "emotion":
        secondary_supports[limb["emotion"]] += 1
    secondary_supports = dict(secondary_supports)

    cond1_emotions = [emo for emo, cnt in core_candidates.items() if cnt >= 2]

    cond2_emotions = [
        emo for emo, core_cnt in core_candidates.items()
        if core_cnt >= 1 and secondary_supports.get(emo, 0) >= 1
    ]

    all_met_emotions = list(set(cond1_emotions + cond2_emotions))

    if len(all_met_emotions) >= 2:
        return {
            "met": True,
            "emotion": None,
            "multi_emotion_conflict": True,
            "conflicting_emotions": all_met_emotions,
            "core_candidates": core_candidates,
            "secondary_supports": secondary_supports,
            "matched_condition": "conflict",
            "summary": f"⚠️ 多情緒衝突：{', '.join(all_met_emotions)} 同時達到條件 → 強制 uncertain（低等）",
        }

    if cond1_emotions:
        emo = cond1_emotions[0]
        return {
            "met": True,
            "emotion": emo,
            "multi_emotion_conflict": False,
            "conflicting_emotions": [],
            "core_candidates": core_candidates,
            "secondary_supports": secondary_supports,
            "matched_condition": "cond1",
            "summary": f"✅ 條件一達成：2 個主要特徵（眼+耳）一致 → {emo}",
        }

    if cond2_emotions:
        emo = cond2_emotions[0]
        return {
            "met": True,
            "emotion": emo,
            "multi_emotion_conflict": False,
            "conflicting_emotions": [],
            "core_candidates": core_candidates,
            "secondary_supports": secondary_supports,
            "matched_condition": "cond2",
            "summary": f"✅ 條件二達成：1 個主要特徵 + 1 個次要特徵一致 → {emo}",
        }

    return {
        "met": False,
        "emotion": None,
        "multi_emotion_conflict": False,
        "conflicting_emotions": [],
        "core_candidates": core_candidates,
        "secondary_supports": secondary_supports,
        "matched_condition": "none",
        "summary": "⏳ Step1+2 尚未達到情緒條件，需進入 Step3 輔助判定",
    }


def evaluate_step3_auxiliary(step12_result, selected_behavior, unknown_behavior):
    if unknown_behavior or not selected_behavior:
        return {
            "met": False,
            "emotion": None,
            "behavior_emotion": None,
            "matched_condition": "none",
            "multi_emotion_conflict": False,
            "conflicting_emotions": [],
            "confidence": "低等",
            "summary": "⚠️ 行為無法判斷，仍未達情緒條件 → uncertain（低等）",
        }

    behavior_feat = selected_behavior[0]
    beh_info = FEATURE_LOOKUP["aux"].get(behavior_feat)
    behavior_emotion = beh_info["emotion"] if beh_info else None

    if not behavior_emotion:
        return {
            "met": False,
            "emotion": None,
            "behavior_emotion": None,
            "matched_condition": "none",
            "multi_emotion_conflict": False,
            "conflicting_emotions": [],
            "confidence": "低等",
            "summary": "⚠️ 行為特徵無對應情緒，仍未達情緒條件 → uncertain（低等）",
        }

    core_candidates = step12_result.get("core_candidates", {})
    secondary_supports = step12_result.get("secondary_supports", {})

    cond3_emotions = [
        emo for emo, cnt in core_candidates.items()
        if cnt >= 1 and behavior_emotion == emo
    ]

    cond4_emotions = [
        emo for emo, cnt in secondary_supports.items()
        if cnt >= 1 and behavior_emotion == emo
    ]

    all_met_step3 = list(set(cond3_emotions + cond4_emotions))

    if len(all_met_step3) >= 2:
        return {
            "met": True,
            "emotion": None,
            "behavior_emotion": behavior_emotion,
            "matched_condition": "conflict",
            "multi_emotion_conflict": True,
            "conflicting_emotions": all_met_step3,
            "confidence": "低等",
            "summary": f"⚠️ 多情緒衝突（Step3）：{', '.join(all_met_step3)} → 強制 uncertain（低等）",
        }

    if cond3_emotions:
        emo = cond3_emotions[0]
        return {
            "met": True,
            "emotion": emo,
            "behavior_emotion": behavior_emotion,
            "matched_condition": "cond3",
            "multi_emotion_conflict": False,
            "conflicting_emotions": [],
            "confidence": "中等",
            "summary": f"✅ 條件三達成：1 個主要特徵 + 行為一致 → {emo}（中等）",
        }

    if cond4_emotions:
        emo = cond4_emotions[0]
        return {
            "met": True,
            "emotion": emo,
            "behavior_emotion": behavior_emotion,
            "matched_condition": "cond4",
            "multi_emotion_conflict": False,
            "conflicting_emotions": [],
            "confidence": "中等",
            "summary": f"✅ 條件四達成：1 個次要特徵 + 行為一致 → {emo}（中等）",
        }

    return {
        "met": False,
        "emotion": None,
        "behavior_emotion": behavior_emotion,
        "matched_condition": "none",
        "multi_emotion_conflict": False,
        "conflicting_emotions": [],
        "confidence": "低等",
        "summary": f"⚠️ 行為（{behavior_emotion}）無法與已選特徵形成條件 → uncertain（低等）",
    }


def evaluate_step3_supplement(step12_emotion, selected_behavior, unknown_behavior):
    if unknown_behavior or not selected_behavior:
        return {
            "behavior_emotion": None,
            "confidence": "中等",
            "summary": "行為無法判斷，維持中等信心",
        }

    behavior_feat = selected_behavior[0]
    beh_info = FEATURE_LOOKUP["aux"].get(behavior_feat)
    behavior_emotion = beh_info["emotion"] if beh_info else None

    if behavior_emotion == step12_emotion:
        return {
            "behavior_emotion": behavior_emotion,
            "confidence": "高等",
            "summary": f"✅ 行為與前述標註一致（{step12_emotion}）→ 高等信心",
        }
    else:
        return {
            "behavior_emotion": behavior_emotion,
            "confidence": "中等",
            "summary": f"⚠️ 行為（{behavior_emotion}）與前述標註（{step12_emotion}）不同 → 中等信心",
        }


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


FEATURE_CATALOG = build_neutral_feature_catalog(EMOTION_SCHEMA)
FEATURE_LOOKUP = build_feature_emotion_lookup(EMOTION_SCHEMA)


def init_session(videos):
    defaults = {
        "page": "instruction",
        "current_index": 0,
        "videos": videos,
        "completed": 0,
        "annotation_step": 1,
        "step1_selected_core": [],
        "step1_unknown_core": [],
        "step2_selected_aux": [],
        "step2_unknown_aux": [],
        "step12_result": None,
        "step12_confirmed": False,
        "step3_selected_behavior": [],
        "step3_unknown_behavior": False,
        "step3_result": None,
        "loaded_saved_record_video": None,
        "inconsistency_confirm": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_checkbox_widget_state(video_index: int):
    prefix_core = f"core_{video_index}_"
    prefix_aux = f"aux_{video_index}_"
    keys_to_delete = [
        k for k in list(st.session_state.keys())
        if k.startswith(prefix_core) or k.startswith(prefix_aux)
    ]
    for k in keys_to_delete:
        del st.session_state[k]


def reset_step_flow():
    video_index = st.session_state.current_index
    st.session_state.annotation_step = 1
    st.session_state.step1_selected_core = []
    st.session_state.step1_unknown_core = []
    st.session_state.step2_selected_aux = []
    st.session_state.step2_unknown_aux = []
    st.session_state.step12_result = None
    st.session_state.step12_confirmed = False
    st.session_state.step3_selected_behavior = []
    st.session_state.step3_unknown_behavior = False
    st.session_state.step3_result = None
    st.session_state["loaded_saved_record_video"] = None
    st.session_state["inconsistency_confirm"] = None
    reset_checkbox_widget_state(video_index)

    for key in [
        f"step12_check_done_{video_index}",
        f"behavior_single_{video_index}",
        f"behavior_unknown_{video_index}",
        f"step3_check_done_{video_index}",
        f"final_emotion_radio_{video_index}",
        f"final_emotion_force_{video_index}",
        f"inconsistency_confirm_{video_index}",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def get_suggested_emotion():
    step3 = st.session_state.get("step3_result") or {}
    step12 = st.session_state.get("step12_result") or {}

    if step12.get("multi_emotion_conflict") or step3.get("multi_emotion_conflict"):
        return "uncertain"

    if step3.get("mode") == "auxiliary" and step3.get("emotion"):
        return step3["emotion"]

    if step12.get("emotion"):
        return step12["emotion"]

    return "uncertain"


def is_force_uncertain():
    step3 = st.session_state.get("step3_result") or {}
    step12 = st.session_state.get("step12_result") or {}
    if step12.get("multi_emotion_conflict"):
        return True
    if step3.get("force_uncertain", False):
        return True
    if step3.get("multi_emotion_conflict"):
        return True
    return False


# 這裡已刪掉「進度列下方說明框」
def render_progress_banner():
    step = st.session_state.annotation_step

    labels = ["Step 1+2：眼耳 / 尾肢", "Step 3：行為", "Step 4：最終確認"]
    step_map = {1: 0, 2: 1, 3: 2}
    current_label_idx = step_map.get(step, 2)

    cols = st.columns(3)
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
st.caption("流程：Step 1+2（眼耳/尾肢）→ Step 3（行為）→ Step 4（最終確認）")

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
        st.session_state.page = "instruction"
        reset_step_flow()
        st.rerun()

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

    st.subheader("二、情緒定義與判斷條件")
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
        reset_checkbox_widget_state(st.session_state.current_index)
        st.session_state["loaded_saved_record_video"] = current_video_name

    render_progress_banner()

    if st.session_state.annotation_step == 1:
        st.markdown("## Step 1：眼睛、耳朵（主要特徵）")

        default_core_values, default_core_unknown = build_feature_saved_values(
            st.session_state.step1_selected_core,
            st.session_state.step1_unknown_core,
        )
        selected_core_all, unknown_core_groups = render_feature_checkbox_grid(
            "",
            "core",
            ["眼睛", "耳朵"],
            FEATURE_CATALOG,
            default_core_values,
            default_core_unknown,
            st.session_state.current_index,
        )

        st.divider()
        st.markdown("## Step 2：尾巴、四肢（次要特徵）")

        default_aux_values, default_aux_unknown = build_feature_saved_values(
            st.session_state.step2_selected_aux,
            st.session_state.step2_unknown_aux,
        )
        selected_aux_all, unknown_aux_groups = render_feature_checkbox_grid(
            "",
            "aux",
            ["尾巴", "四肢"],
            FEATURE_CATALOG,
            default_aux_values,
            default_aux_unknown,
            st.session_state.current_index,
        )

        st.divider()

        col_check, col_next = st.columns(2)

        with col_check:
            if st.button("檢查 Step 1+2", key=f"step12_check_{st.session_state.current_index}"):
                st.session_state.step1_selected_core = selected_core_all
                st.session_state.step1_unknown_core = unknown_core_groups
                st.session_state.step2_selected_aux = selected_aux_all
                st.session_state.step2_unknown_aux = unknown_aux_groups
                st.session_state.step12_result = evaluate_step12(
                    selected_core_all, unknown_core_groups,
                    selected_aux_all, unknown_aux_groups,
                )
                st.session_state.step12_confirmed = False
                st.rerun()

        result12 = st.session_state.step12_result
        can_continue = False

        if result12:
            if result12.get("multi_emotion_conflict"):
                st.markdown(
                    f'<div class="conflict-box"><b>{result12["summary"]}</b><br>'
                    f'多個情緒同時達到條件，信心程度為低等，最終只能選 uncertain。</div>',
                    unsafe_allow_html=True,
                )
                can_continue = True
            elif result12["met"]:
                st.markdown(
                    f'<div class="ok-box"><b>{result12["summary"]}</b></div>',
                    unsafe_allow_html=True,
                )
                can_continue = True
            else:
                st.markdown(
                    f'<div class="warn-box"><b>{result12["summary"]}</b><br>'
                    f'Step 1+2 尚未達到情緒條件，Step3 將作為輔助判定。</div>',
                    unsafe_allow_html=True,
                )
                can_continue = True

        with col_next:
            if st.button(
                "繼續 → Step 3",
                key=f"step12_next_{st.session_state.current_index}",
                disabled=not can_continue,
            ):
                st.session_state.annotation_step = 2
                st.rerun()

    elif st.session_state.annotation_step == 2:
        step12_result = st.session_state.step12_result

        if step12_result and step12_result.get("multi_emotion_conflict"):
            st.markdown(
                f'<div class="conflict-box" style="margin-bottom:12px;">'
                f'{step12_result["summary"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("## Step 3：行為（輔助判定）")
            st.caption("多情緒衝突，行為特徵作為輔助，但最終仍會強制為 uncertain（低等）。")
        elif step12_result and step12_result["met"]:
            st.markdown(
                f'<div class="ok-box" style="margin-bottom:12px;">'
                f'暫定情緒：<b>{step12_result["emotion"]}</b> ｜ {step12_result["summary"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("## Step 3：行為（補充）")
            st.caption("Step 1+2 已達條件，行為特徵作為補充。若行為一致 → 高等信心；若不同 → 中等信心。")
        else:
            st.markdown(
                '<div class="warn-box" style="margin-bottom:12px;">'
                'Step 1+2 尚未達條件，行為特徵作為<b>輔助判定</b>。'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown("## Step 3：行為（輔助判定）")
            st.caption("請選擇最能代表貓咪行為的特徵，作為輔助情緒條件判定。")

        behavior_options = group_features_for_display(FEATURE_CATALOG, "aux").get("行為", [])
        behavior_key = f"behavior_single_{st.session_state.current_index}"
        behavior_unknown_key = f"behavior_unknown_{st.session_state.current_index}"

        if behavior_key not in st.session_state:
            if st.session_state.step3_selected_behavior:
                st.session_state[behavior_key] = st.session_state.step3_selected_behavior[0]
            else:
                st.session_state[behavior_key] = None

        if behavior_unknown_key not in st.session_state:
            st.session_state[behavior_unknown_key] = st.session_state.step3_unknown_behavior

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
        col_check, col_next = st.columns(2)

        with col_check:
            if st.button("檢查 Step 3", key=f"step3_check_{st.session_state.current_index}"):
                st.session_state.step3_selected_behavior = selected_behavior
                st.session_state.step3_unknown_behavior = behavior_unknown

                if step12_result and step12_result.get("multi_emotion_conflict"):
                    beh_info = FEATURE_LOOKUP["aux"].get(selected_behavior[0]) if selected_behavior else None
                    behavior_emotion = beh_info["emotion"] if beh_info else None
                    st.session_state.step3_result = {
                        "mode": "conflict_forced",
                        "emotion": None,
                        "behavior_emotion": behavior_emotion,
                        "confidence": "低等",
                        "multi_emotion_conflict": True,
                        "conflicting_emotions": step12_result.get("conflicting_emotions", []),
                        "summary": f"⚠️ 多情緒衝突，強制 uncertain（低等）",
                        "force_uncertain": True,
                    }
                elif step12_result and step12_result["met"]:
                    result3 = evaluate_step3_supplement(
                        step12_result["emotion"],
                        selected_behavior,
                        behavior_unknown,
                    )
                    st.session_state.step3_result = {
                        "mode": "supplement",
                        "emotion": step12_result["emotion"],
                        "behavior_emotion": result3["behavior_emotion"],
                        "confidence": result3["confidence"],
                        "multi_emotion_conflict": False,
                        "conflicting_emotions": [],
                        "summary": result3["summary"],
                        "force_uncertain": False,
                    }
                else:
                    result3 = evaluate_step3_auxiliary(
                        step12_result or {},
                        selected_behavior,
                        behavior_unknown,
                    )
                    st.session_state.step3_result = {
                        "mode": "auxiliary",
                        "emotion": result3["emotion"],
                        "behavior_emotion": result3["behavior_emotion"],
                        "confidence": result3["confidence"],
                        "multi_emotion_conflict": result3.get("multi_emotion_conflict", False),
                        "conflicting_emotions": result3.get("conflicting_emotions", []),
                        "summary": result3["summary"],
                        "force_uncertain": not result3["met"],
                    }
                st.rerun()

        result3 = st.session_state.step3_result
        can_continue = False

        if result3:
            if result3.get("force_uncertain") or result3.get("multi_emotion_conflict"):
                st.markdown(
                    f'<div class="low-box"><b>{result3["summary"]}</b></div>',
                    unsafe_allow_html=True,
                )
            elif result3["confidence"] == "高等":
                st.markdown(
                    f'<div class="ok-box"><b>{result3["summary"]}</b></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="warn-box"><b>{result3["summary"]}</b></div>',
                    unsafe_allow_html=True,
                )
            can_continue = True

        with col_next:
            if st.button(
                "繼續 → Step 4（最終確認）",
                key=f"step3_next_{st.session_state.current_index}",
                disabled=not can_continue,
            ):
                st.session_state.annotation_step = 3
                st.session_state["inconsistency_confirm"] = None
                st.rerun()

    else:
        step12_result = st.session_state.step12_result or {}
        step3_result = st.session_state.step3_result or {}

        force_uncertain = is_force_uncertain()
        suggested_emotion = get_suggested_emotion()
        confidence = step3_result.get("confidence", "低等")

        step12_provisional = step12_result.get("emotion")
        step3_provisional = step3_result.get("emotion")

        st.markdown("## Step 4：最終情緒確認")

        st.markdown(f"- **Step 1+2 結果：** {step12_result.get('summary', '—')}")
        st.markdown(f"- **Step 3 結果：** {step3_result.get('summary', '—')}")

        if step12_result.get("multi_emotion_conflict") or step3_result.get("multi_emotion_conflict"):
            conflict_emos = step12_result.get("conflicting_emotions") or step3_result.get("conflicting_emotions", [])
            st.markdown(
                f'<div class="conflict-box">'
                f'<b>多情緒衝突：</b>以下情緒同時達到條件：{", ".join(conflict_emos)}<br>'
                f'信心程度：<b>低等</b>，最終情緒強制為 <b>uncertain</b>。'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif force_uncertain:
            st.markdown(
                f'<div class="low-box">'
                f'<b>信心程度：低等</b><br>'
                f'特徵未達情緒條件，最終情緒只能選 <b>uncertain</b>。'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif confidence == "高等":
            st.markdown(
                f'<div class="ok-box">'
                f'<b>信心程度：高等</b><br>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="warn-box">'
                f'<b>信心程度：中等</b><br>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if force_uncertain:
            selected_final = st.radio(
                "請選擇最終主導情緒",
                ["uncertain"],
                index=0,
                key=f"final_emotion_force_{st.session_state.current_index}",
            )
        else:
            selected_final = st.radio(
                "請選擇最終主導情緒",
                MAIN_EMOTIONS,
                index=None,
                key=f"final_emotion_radio_{st.session_state.current_index}",
            )

        inconsistency_msg = None
        if selected_final and not force_uncertain:
            if suggested_emotion and selected_final != suggested_emotion:
                inconsistency_msg = (
                    f"你選擇的最終情緒「{selected_final}」"
                    f"與前述標註不一致**。"
                )

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

        def build_record(final_emotion: str):
            if not annotator_name:
                st.error("請先在左側輸入標註者姓名或編號。")
                return None
            if not final_emotion:
                st.error("請先選擇最終主導情緒。")
                return None

            eye_selected = get_group_selected_features(st.session_state.step1_selected_core, "core", "眼睛")
            ear_selected = get_group_selected_features(st.session_state.step1_selected_core, "core", "耳朵")
            tail_selected = get_group_selected_features(st.session_state.step2_selected_aux, "aux", "尾巴")
            limb_selected = get_group_selected_features(st.session_state.step2_selected_aux, "aux", "四肢")

            conflicting = (
                step12_result.get("conflicting_emotions")
                or step3_result.get("conflicting_emotions")
                or []
            )

            return {
                "record_id": compute_record_id(annotator_name.strip(), current_video_name),
                "video_file": current_video_name,
                "eye_selected": json.dumps(eye_selected, ensure_ascii=False),
                "ear_selected": json.dumps(ear_selected, ensure_ascii=False),
                "tail_selected": json.dumps(tail_selected, ensure_ascii=False),
                "limb_selected": json.dumps(limb_selected, ensure_ascii=False),
                "behavior_selected": json.dumps(st.session_state.step3_selected_behavior, ensure_ascii=False),
                "step12_provisional_emotion": step12_provisional or "",
                "step3_provisional_emotion": step3_provisional or "",
                "suggested_emotion": suggested_emotion or "",
                "final_emotion": final_emotion,
                "final_matches_suggested": str(final_emotion == (suggested_emotion or "")),
                "multi_emotion_conflict": str(bool(conflicting)),
                "conflicting_emotions": json.dumps(conflicting, ensure_ascii=False),
                "confidence": confidence,
                "step12_condition": step12_result.get("matched_condition", "none"),
                "step12_summary": step12_result.get("summary", ""),
                "step3_mode": step3_result.get("mode", ""),
                "step3_summary": step3_result.get("summary", ""),
                "step1_selected_core_all": json.dumps(st.session_state.step1_selected_core, ensure_ascii=False),
                "step1_unknown_core_groups": json.dumps(st.session_state.step1_unknown_core, ensure_ascii=False),
                "step2_selected_aux_all": json.dumps(st.session_state.step2_selected_aux, ensure_ascii=False),
                "step2_unknown_aux_groups": json.dumps(st.session_state.step2_unknown_aux, ensure_ascii=False),
                "step3_unknown_behavior": str(st.session_state.step3_unknown_behavior),
                "note": note,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

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
                if st.button("✅ 是，確認標註無誤", use_container_width=True,
                             key=f"confirm_yes_{st.session_state.current_index}"):
                    st.session_state[confirm_key] = "confirmed"
                    st.rerun()
            with c2:
                if st.button("❌ 否，重新標註", use_container_width=True,
                             key=f"confirm_no_{st.session_state.current_index}"):
                    st.session_state[confirm_key] = None
                    reset_step_flow()
                    st.rerun()

        elif st.session_state.get(confirm_key) == "redo":
            st.session_state[confirm_key] = None
            reset_step_flow()
            st.rerun()

        else:
            col_save, col_sync = st.columns(2)

            with col_save:
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

            with col_sync:
                if st.button(
                    "☁️ 儲存並同步 Google Sheet",
                    key=f"save_sync_{st.session_state.current_index}",
                ):
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
        if st.button(
            "下一段",
            disabled=st.session_state.current_index >= len(st.session_state.videos) - 1,
        ):
            st.session_state.current_index += 1
            reset_step_flow()
            st.rerun()
