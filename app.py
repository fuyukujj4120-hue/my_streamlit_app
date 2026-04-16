import json
import hashlib
from collections import Counter, defaultdict
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
  

    /* 放大「請選擇最終主導情緒」這行標題 */
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
    "標註流程分為四步：Step 1 眼睛/耳朵；Step 2 尾巴/四肢；Step 3 行為；Step 4 最終情緒確認",
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
        "step3_selected_aux": [],
        "step3_unknown_aux": [],
        "step1_check_result": None,
        "step2_check_result": None,
        "step3_check_result": None,
        "step1_confirmed": False,
        "step2_confirmed": False,
        "step3_confirmed": False,
        "step1_confirmation_state": None,
        "step2_confirmation_state": None,
        "step3_confirmation_state": None,
        "loaded_saved_record_video": None,
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
    st.session_state.step3_selected_aux = []
    st.session_state.step3_unknown_aux = []
    st.session_state.step1_check_result = None
    st.session_state.step2_check_result = None
    st.session_state.step3_check_result = None
    st.session_state.step1_confirmed = False
    st.session_state.step2_confirmed = False
    st.session_state.step3_confirmed = False
    st.session_state["step1_confirmation_state"] = None
    st.session_state["step2_confirmation_state"] = None
    st.session_state["step3_confirmation_state"] = None
    st.session_state["loaded_saved_record_video"] = None
    reset_checkbox_widget_state(video_index)

    for key in [
        f"force_uncertain_{video_index}",
        f"step4_check_result_{video_index}",
        f"behavior_single_{video_index}",
        f"behavior_unknown_{video_index}",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def load_saved_record_into_step_state(saved_record):
    if not saved_record:
        return
    st.session_state.step1_selected_core = parse_json_list(saved_record.get("step1_selected_core_all", None))
    st.session_state.step1_unknown_core = parse_json_list(saved_record.get("step1_unknown_core_groups", None))
    st.session_state.step2_selected_aux = parse_json_list(saved_record.get("step2_selected_aux_all", None))
    st.session_state.step2_unknown_aux = parse_json_list(saved_record.get("step2_unknown_aux_groups", None))
    st.session_state.step3_selected_aux = parse_json_list(saved_record.get("step3_selected_aux_all", None))
    st.session_state.step3_unknown_aux = parse_json_list(saved_record.get("step3_unknown_aux_groups", None))


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
        return {"status": "unknown", "emotion": None}

    group_features = []
    for feature in selected_features:
        item = FEATURE_LOOKUP[feature_type].get(feature)
        if item and item["group"] == group_name:
            group_features.append(feature)

    if not group_features:
        return {"status": "unknown", "emotion": None}

    emotions = [FEATURE_LOOKUP[feature_type][f]["emotion"] for f in group_features]
    counts = Counter(emotions)
    top_count = max(counts.values())
    winners = [emo for emo, c in counts.items() if c == top_count]

    if len(winners) == 1:
        return {"status": "emotion", "emotion": winners[0]}
    return {"status": "conflict", "emotion": None}


def get_group_selected_features(selected_features, feature_type, group_name):
    result = []
    for feature in selected_features:
        item = FEATURE_LOOKUP[feature_type].get(feature)
        if item and item["group"] == group_name:
            result.append(feature)
    return result


def count_step4_supports(step1_result, step2_result, step3_result):
    overall = evaluate_step4_overall_result(step1_result, step2_result, step3_result)

    if overall["mode"] != "single":
        return 0, 0, 0

    final_emo = overall["valid_emotions"][0]

    core_candidates = step1_result.get("core_candidates", {}) if step1_result else {}
    secondary_supports = step2_result.get("secondary_supports", {}) if step2_result else {}
    behavior_emotion = step3_result.get("behavior_emotion") if step3_result else None

    core_cnt = core_candidates.get(final_emo, 0)
    sec_cnt = secondary_supports.get(final_emo, 0)
    beh_cnt = 1 if behavior_emotion == final_emo else 0

    return core_cnt, sec_cnt, beh_cnt


def check_step1(selected_core_all, unknown_core_groups):
    eye = infer_group_emotion(selected_core_all, unknown_core_groups, "core", "眼睛")
    ear = infer_group_emotion(selected_core_all, unknown_core_groups, "core", "耳朵")

    core_candidates = defaultdict(int)
    if eye["status"] == "emotion":
        core_candidates[eye["emotion"]] += 1
    if ear["status"] == "emotion":
        core_candidates[ear["emotion"]] += 1
    core_candidates = dict(core_candidates)

    if eye["status"] == "emotion" and ear["status"] == "emotion" and eye["emotion"] == ear["emotion"]:
        emo = eye["emotion"]
        return {
            "status": "一致",
            "tentative_emotion": emo,
            "core_candidates": core_candidates,
            "summary": "一致（2 核心特徵）",
            "needs_confirmation": False,
        }

    statuses = [eye["status"], ear["status"]]

    if statuses.count("emotion") == 1 and statuses.count("unknown") == 1:
        emo = eye["emotion"] if eye["status"] == "emotion" else ear["emotion"]
        return {
            "status": "部分一致",
            "tentative_emotion": emo,
            "core_candidates": core_candidates,
            "summary": "部分一致（1 核心特徵）",
            "needs_confirmation": False,
        }

    if eye["status"] == "unknown" and ear["status"] == "unknown":
        return {
            "status": "皆無法判斷",
            "tentative_emotion": "uncertain",
            "core_candidates": {},
            "summary": "皆無法判斷（0 核心特徵）",
            "needs_confirmation": True,
        }

    return {
        "status": "不一致",
        "tentative_emotion": "uncertain",
        "core_candidates": core_candidates,
        "summary": "不一致（核心候選不一致）",
        "needs_confirmation": True,
    }


def check_step2(step1_result, selected_aux_all, unknown_aux_groups):
    tail = infer_group_emotion(selected_aux_all, unknown_aux_groups, "aux", "尾巴")
    limb = infer_group_emotion(selected_aux_all, unknown_aux_groups, "aux", "四肢")

    core_candidates = step1_result.get("core_candidates", {}) or {}
    secondary_supports = defaultdict(int)

    if tail["status"] == "emotion" and tail["emotion"] in core_candidates:
        secondary_supports[tail["emotion"]] += 1
    if limb["status"] == "emotion" and limb["emotion"] in core_candidates:
        secondary_supports[limb["emotion"]] += 1

    secondary_supports = dict(secondary_supports)

    if core_candidates:
        combined = {}
        for emo, core_cnt in core_candidates.items():
            combined[emo] = core_cnt + secondary_supports.get(emo, 0)

        if combined:
            max_score = max(combined.values())
            winners = [emo for emo, score in combined.items() if score == max_score]

            if len(winners) == 1 and max_score > 0:
                winner = winners[0]
                core_cnt = core_candidates.get(winner, 0)
                sec_cnt = secondary_supports.get(winner, 0)
                return {
                    "status": "部分一致" if core_cnt == 1 and sec_cnt == 1 else "一致",
                    "tentative_emotion": winner,
                    "core_candidates": core_candidates,
                    "secondary_supports": secondary_supports,
                    "summary": f"{'部分一致' if core_cnt == 1 and sec_cnt == 1 else '一致'}（{core_cnt} 核心、{sec_cnt} 次要）",
                    "needs_confirmation": False,
                }

            if len(winners) >= 2 and max_score > 0:
                return {
                    "status": "不一致",
                    "tentative_emotion": "uncertain",
                    "core_candidates": core_candidates,
                    "secondary_supports": secondary_supports,
                    "summary": "不一致（兩種以上情緒同樣明顯）",
                    "needs_confirmation": True,
                }

    aux_counts = Counter()
    if tail["status"] == "emotion":
        aux_counts[tail["emotion"]] += 1
    if limb["status"] == "emotion":
        aux_counts[limb["emotion"]] += 1

    if len(aux_counts) == 1 and sum(aux_counts.values()) >= 1:
        emo = list(aux_counts.keys())[0]
        cnt = list(aux_counts.values())[0]
        return {
            "status": "部分一致" if cnt == 1 else "一致",
            "tentative_emotion": emo,
            "core_candidates": {},
            "secondary_supports": {emo: cnt},
            "summary": f"{'部分一致' if cnt == 1 else '一致'}（0 核心、{cnt} 次要）",
            "needs_confirmation": False,
        }

    if tail["status"] == "unknown" and limb["status"] == "unknown":
        return {
            "status": "皆無法判斷",
            "tentative_emotion": "uncertain",
            "core_candidates": core_candidates,
            "secondary_supports": secondary_supports,
            "summary": "皆無法判斷",
            "needs_confirmation": True,
        }

    return {
        "status": "不一致",
        "tentative_emotion": "uncertain",
        "core_candidates": core_candidates,
        "secondary_supports": secondary_supports,
        "summary": "不一致",
        "needs_confirmation": True,
    }


def check_step3(step1_result, step2_result, selected_aux_all, unknown_aux_groups):
    behavior = infer_group_emotion(selected_aux_all, unknown_aux_groups, "aux", "行為")
    behavior_emotion = behavior["emotion"] if behavior["status"] == "emotion" else None

    core_candidates = step1_result.get("core_candidates", {}) or {}
    secondary_supports = step2_result.get("secondary_supports", {}) or {}
    tentative_emotion = step2_result.get("tentative_emotion", "uncertain")

    if tentative_emotion != "uncertain":
        if behavior_emotion == tentative_emotion:
            return {
                "status": "一致",
                "final_emotion": tentative_emotion,
                "behavior_emotion": behavior_emotion,
                "summary": "一致（行為支持前述判斷）",
                "reason": "行為支持目前主情緒。",
                "needs_confirmation": False,
                "confidence": "高",
            }

        if behavior_emotion is None:
            return {
                "status": "部分一致",
                "final_emotion": tentative_emotion,
                "behavior_emotion": behavior_emotion,
                "summary": "部分一致（行為無法判斷）",
                "reason": "已有主情緒，但行為無法提供額外支持。",
                "needs_confirmation": False,
                "confidence": "中",
            }

        return {
            "status": "不一致",
            "final_emotion": "uncertain",
            "behavior_emotion": behavior_emotion,
            "summary": "不一致",
            "reason": "已有主情緒，但行為不支持目前主情緒。",
            "needs_confirmation": True,
            "confidence": "低",
        }

    if behavior_emotion:
        if core_candidates.get(behavior_emotion, 0) >= 1:
            return {
                "status": "部分一致",
                "final_emotion": behavior_emotion,
                "behavior_emotion": behavior_emotion,
                "summary": "部分一致（1 個主要特徵 + 輔助特徵一致）",
                "reason": "行為支持其中一個核心候選。",
                "needs_confirmation": False,
                "confidence": "中",
            }

        if secondary_supports.get(behavior_emotion, 0) >= 1:
            return {
                "status": "部分一致",
                "final_emotion": behavior_emotion,
                "behavior_emotion": behavior_emotion,
                "summary": "部分一致（1 個次要特徵 + 輔助特徵一致）",
                "reason": "行為支持其中一個次要候選。",
                "needs_confirmation": False,
                "confidence": "中",
            }

        return {
            "status": "不一致",
            "final_emotion": "uncertain",
            "behavior_emotion": behavior_emotion,
            "summary": "不一致",
            "reason": "行為有情緒，但無法支持任何目前候選情緒。",
            "needs_confirmation": True,
            "confidence": "低",
        }

    return {
        "status": "不一致",
        "final_emotion": "uncertain",
        "behavior_emotion": behavior_emotion,
        "summary": "不一致",
        "reason": "行為無法判斷，因此無法支持任一情緒。",
        "needs_confirmation": True,
        "confidence": "低",
    }


def evaluate_step4_overall_result(step1_result, step2_result, step3_result):
    core_candidates = step1_result.get("core_candidates", {}) if step1_result else {}
    secondary_supports = step2_result.get("secondary_supports", {}) if step2_result else {}
    behavior_emotion = step3_result.get("behavior_emotion") if step3_result else None

    total_support = defaultdict(lambda: {"core": 0, "secondary": 0, "behavior": 0})

    for emo, cnt in core_candidates.items():
        total_support[emo]["core"] = cnt
    for emo, cnt in secondary_supports.items():
        total_support[emo]["secondary"] = cnt
    if behavior_emotion:
        total_support[behavior_emotion]["behavior"] = 1

    valid_emotions = []
    for emo, support in total_support.items():
        core_cnt = support["core"]
        sec_cnt = support["secondary"]
        beh_cnt = support["behavior"]

        cond1 = core_cnt >= 2
        cond2 = core_cnt >= 1 and sec_cnt >= 1
        cond3 = core_cnt >= 1 and beh_cnt >= 1
        cond4 = sec_cnt >= 1 and beh_cnt >= 1

        if cond1 or cond2 or cond3 or cond4:
            valid_emotions.append(emo)

    if len(valid_emotions) >= 2:
        return {
            "mode": "multiple",
            "valid_emotions": valid_emotions,
            "message": "多種情緒",
            "force_uncertain": True,
        }

    if len(valid_emotions) == 1:
        return {
            "mode": "single",
            "valid_emotions": valid_emotions,
            "message": "已形成單一情緒條件",
            "force_uncertain": False,
        }

    return {
        "mode": "none",
        "valid_emotions": [],
        "message": "未達最低特徵條件",
        "force_uncertain": False,
    }


def evaluate_final_label_consistency(final_label, step1_result, step2_result, step3_result):
    overall = evaluate_step4_overall_result(step1_result, step2_result, step3_result)

    if overall["mode"] == "multiple":
        if final_label == "uncertain":
            return {
                "is_consistent": True,
                "message": "多種情緒",
                "confidence": "低信心",
            }
        return {
            "is_consistent": False,
            "message": "標註情緒與所選特徵條件不符",
            "confidence": "低信心",
        }

    if overall["mode"] == "single":
        checked_emotion = overall["valid_emotions"][0]
        if final_label == checked_emotion:
            return {
                "is_consistent": True,
                "message": "標註情緒與所選特徵條件相符",
                "confidence": "高信心",
            }
        if final_label == "uncertain":
            return {
                "is_consistent": True,
                "message": "標註為 uncertain",
                "confidence": "低信心",
            }
        return {
            "is_consistent": False,
            "message": "標註情緒與所選特徵條件不符",
            "confidence": "低信心",
        }

    if final_label == "uncertain":
        return {
            "is_consistent": True,
            "message": "標註為 uncertain",
            "confidence": "低信心",
        }

    return {
        "is_consistent": True,
        "message": "未達最低特徵條件",
        "confidence": "低信心",
    }


def render_check_result_box(title, result):
    st.markdown(f"### {title}")
    st.markdown(
        f"""
        <div class="result-box">
            <b>判定狀態：</b> {result['status']}<br>
            <b>結果摘要：</b> {result['summary']}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confirmation_ui(step_prefix: str, result: dict):
    state_key = f"{step_prefix}_confirmation_state"

    if state_key not in st.session_state:
        st.session_state[state_key] = None

    if not result.get("needs_confirmation", False):
        return True

    if st.session_state[state_key] == "yes":
        return True

    if st.session_state[state_key] == "no":
        return False

    st.markdown(
        f"""
        <div class="warn-box">
            <b>{result['status']}</b><br>
            請確認標註無誤：是 / 否
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_yes, col_no = st.columns(2)
    with col_yes:
        yes_clicked = st.button("是，標註無誤", key=f"{step_prefix}_confirm_yes")
    with col_no:
        no_clicked = st.button("否，重新標註", key=f"{step_prefix}_confirm_no")

    if yes_clicked:
        st.session_state[state_key] = "yes"
        st.rerun()

    if no_clicked:
        st.session_state[state_key] = "no"
        st.rerun()

    return None


FEATURE_CATALOG = build_neutral_feature_catalog(EMOTION_SCHEMA)
FEATURE_LOOKUP = build_feature_emotion_lookup(EMOTION_SCHEMA)

st.title(APP_TITLE)
st.caption("流程：Step 1 → Step 2 → Step 3 → Step 4（最終情緒確認）")

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
    st.markdown('<div class="definition-title">情緒定義快速查看</div>', unsafe_allow_html=True)

    button_labels = {
        "害怕": "😿 害怕",
        "憤怒/狂怒": "😾 憤怒/狂怒",
        "歡樂/玩耍": "😺 歡樂/玩耍",
        "滿意": "😽 滿意",
        "興趣": "🐾 興趣",
    }

    st.markdown('<div class="definition-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    emotion_names = list(EMOTION_SCHEMA.keys())

    for i, emotion_name in enumerate(emotion_names):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            if st.button(
                button_labels.get(emotion_name, emotion_name),
                key=f"sidebar_emotion_{emotion_name}",
                use_container_width=True,
            ):
                show_emotion_dialog(emotion_name)

    st.markdown('</div>', unsafe_allow_html=True)

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

    if st.session_state.annotation_step == 1:
        default_core_values, default_core_unknown_groups = build_feature_saved_values(
            st.session_state.step1_selected_core,
            st.session_state.step1_unknown_core,
        )

        selected_core_all, unknown_core_groups = render_feature_checkbox_grid(
            "## Step 1：先看眼睛、耳朵",
            "core",
            ["眼睛", "耳朵"],
            FEATURE_CATALOG,
            default_core_values,
            default_core_unknown_groups,
            st.session_state.current_index,
        )

        col_check, col_continue = st.columns(2)

        with col_check:
            if st.button("檢查標註", key=f"step1_check_{st.session_state.current_index}"):
                st.session_state.step1_selected_core = selected_core_all
                st.session_state.step1_unknown_core = unknown_core_groups
                st.session_state.step1_check_result = check_step1(selected_core_all, unknown_core_groups)
                st.session_state.step1_confirmed = False
                st.session_state["step1_confirmation_state"] = None
                st.rerun()

        result = st.session_state.step1_check_result
        can_continue = False

        if result:
            render_check_result_box("Step 1 檢查結果", result)
            confirmation_state = st.session_state.get("step1_confirmation_state", None)

            if result["needs_confirmation"] and confirmation_state is None:
                render_confirmation_ui("step1", result)
                can_continue = False
            elif confirmation_state == "yes":
                st.session_state.step1_confirmed = True
                st.markdown('<div class="ok-box"><b>標註已確認無誤</b></div>', unsafe_allow_html=True)
                can_continue = True
            elif confirmation_state == "no":
                st.session_state.step1_confirmed = False
                st.info("請重新標註後，再按一次「檢查標註」。")
                can_continue = False
            else:
                can_continue = True

        with col_continue:
            if st.button(
                "繼續標註",
                key=f"step1_next_{st.session_state.current_index}",
                disabled=not can_continue,
            ):
                st.session_state.annotation_step = 2
                st.rerun()

    elif st.session_state.annotation_step == 2:
        default_aux_values, default_aux_unknown_groups = build_feature_saved_values(
            st.session_state.step2_selected_aux,
            st.session_state.step2_unknown_aux,
        )

        selected_aux_all, unknown_aux_groups = render_feature_checkbox_grid(
            "## Step 2：看尾巴、四肢",
            "aux",
            ["尾巴", "四肢"],
            FEATURE_CATALOG,
            default_aux_values,
            default_aux_unknown_groups,
            st.session_state.current_index,
        )

        col_check, col_continue = st.columns(2)

        with col_check:
            if st.button("檢查標註", key=f"step2_check_{st.session_state.current_index}"):
                st.session_state.step2_selected_aux = selected_aux_all
                st.session_state.step2_unknown_aux = unknown_aux_groups
                st.session_state.step2_check_result = check_step2(
                    st.session_state.step1_check_result,
                    selected_aux_all,
                    unknown_aux_groups,
                )
                st.session_state.step2_confirmed = False
                st.session_state["step2_confirmation_state"] = None
                st.rerun()

        result = st.session_state.step2_check_result
        can_continue = False

        if result:
            render_check_result_box("Step 2 檢查結果", result)
            confirmation_state = st.session_state.get("step2_confirmation_state", None)

            if result["needs_confirmation"] and confirmation_state is None:
                render_confirmation_ui("step2", result)
                can_continue = False
            elif confirmation_state == "yes":
                st.session_state.step2_confirmed = True
                st.markdown('<div class="ok-box"><b>標註已確認無誤</b></div>', unsafe_allow_html=True)
                can_continue = True
            elif confirmation_state == "no":
                st.session_state.step2_confirmed = False
                st.info("請重新標註後，再按一次「檢查標註」。")
                can_continue = False
            else:
                can_continue = True

        with col_continue:
            if st.button(
                "繼續標註",
                key=f"step2_next_{st.session_state.current_index}",
                disabled=not can_continue,
            ):
                st.session_state.annotation_step = 3
                st.rerun()

    elif st.session_state.annotation_step == 3:
        st.markdown("## Step 3：看行為（單選）")

        behavior_options = group_features_for_display(FEATURE_CATALOG, "aux").get("行為", [])

        behavior_key = f"behavior_single_{st.session_state.current_index}"
        behavior_unknown_key = f"behavior_unknown_{st.session_state.current_index}"

        if behavior_key not in st.session_state:
            if st.session_state.step3_selected_aux:
                st.session_state[behavior_key] = st.session_state.step3_selected_aux[0]
            else:
                st.session_state[behavior_key] = None

        if behavior_unknown_key not in st.session_state:
            st.session_state[behavior_unknown_key] = "行為" in st.session_state.step3_unknown_aux

        col_u1, col_u2 = st.columns([8, 2])
        with col_u1:
            st.markdown("### 行為")
        with col_u2:
            behavior_unknown = st.checkbox("無法判斷", key=behavior_unknown_key)

        if behavior_unknown:
            st.session_state[behavior_key] = None
            selected_behavior = []
            unknown_aux_groups = ["行為"]
        else:
            selected_behavior_value = st.radio(
                "請選擇一個最主要的行為特徵",
                behavior_options,
                index=behavior_options.index(st.session_state[behavior_key]) if st.session_state[behavior_key] in behavior_options else None,
                key=behavior_key,
            )
            selected_behavior = [selected_behavior_value] if selected_behavior_value else []
            unknown_aux_groups = []

        col_check, col_continue = st.columns(2)

        with col_check:
            if st.button("檢查標註", key=f"step3_check_{st.session_state.current_index}"):
                st.session_state.step3_selected_aux = selected_behavior
                st.session_state.step3_unknown_aux = unknown_aux_groups
                st.session_state.step3_check_result = check_step3(
                    st.session_state.step1_check_result,
                    st.session_state.step2_check_result,
                    selected_behavior,
                    unknown_aux_groups,
                )
                st.session_state.step3_confirmed = False
                st.session_state["step3_confirmation_state"] = None
                st.rerun()

        result = st.session_state.step3_check_result
        can_continue = False

        if result:
            render_check_result_box("Step 3 檢查結果", result)
            if result.get("reason"):
                st.caption(f"原因：{result['reason']}")

            confirmation_state = st.session_state.get("step3_confirmation_state", None)

            if result["needs_confirmation"] and confirmation_state is None:
                render_confirmation_ui("step3", result)
                can_continue = False
            elif confirmation_state == "yes":
                st.session_state.step3_confirmed = True
                st.markdown('<div class="ok-box"><b>標註已確認無誤</b></div>', unsafe_allow_html=True)
                can_continue = True
            elif confirmation_state == "no":
                st.session_state.step3_confirmed = False
                st.info("請重新標註後，再按一次「檢查標註」。")
                can_continue = False
            else:
                can_continue = True

        with col_continue:
            if st.button(
                "繼續標註",
                key=f"step3_next_{st.session_state.current_index}",
                disabled=not can_continue,
            ):
                st.session_state.annotation_step = 4
                st.rerun()

    else:
        step1_result = st.session_state.step1_check_result
        step2_result = st.session_state.step2_check_result
        step3_result = st.session_state.step3_check_result
        default_note = saved_record.get("note", "") if saved_record else ""

        core_cnt, sec_cnt, beh_cnt = count_step4_supports(step1_result, step2_result, step3_result)
        overall_result = evaluate_step4_overall_result(step1_result, step2_result, step3_result)

        st.markdown("## Step 4：最終情緒確認")
        if step1_result:
            st.markdown(f"- Step 1：{step1_result['summary']}")
        if step2_result:
            st.markdown(f"- Step 2：{step2_result['summary']}")
        if step3_result:
            st.markdown(f"- Step 3：{step3_result['summary']}")

        if overall_result["mode"] == "single":
            st.markdown(f"- 目前一致核心支持數：{core_cnt}")
            st.markdown(f"- 目前一致次要支持數：{sec_cnt}")
            st.markdown(f"- 目前輔助（行為）支持數：{beh_cnt}")
        elif overall_result["mode"] == "multiple":
            st.markdown("- 目前狀態：多種情緒")
        else:
            st.markdown("- 目前狀態：未達最低特徵條件")

        step4_check_key = f"step4_check_result_{st.session_state.current_index}"
        if step4_check_key not in st.session_state:
            st.session_state[step4_check_key] = None

        force_uncertain_key = f"force_uncertain_{st.session_state.current_index}"
        if force_uncertain_key not in st.session_state:
            st.session_state[force_uncertain_key] = overall_result["force_uncertain"]

        if overall_result["force_uncertain"]:
            st.session_state[force_uncertain_key] = True
            st.markdown(
                """
                <div class="low-box">
                    <b>多種情緒</b><br>
                    已出現兩種以上情緒，最終只能選 uncertain
                </div>
                """,
                unsafe_allow_html=True,
            )

        force_uncertain = st.session_state[force_uncertain_key]

        if force_uncertain:
            selected_final = st.radio(
                "請選擇最終主導情緒",
                ["uncertain"],
                index=0,
                key=f"main_emotion_force_{st.session_state.current_index}",
            )
        else:
            selected_final = st.radio(
                "請選擇最終主導情緒",
                MAIN_EMOTIONS,
                index=None,
                key=f"main_emotion_radio_{st.session_state.current_index}",
            )

        col_check, col_save = st.columns(2)

        with col_check:
            if st.button("檢查標註", key=f"step4_check_{st.session_state.current_index}"):
                if selected_final is None:
                    st.session_state[step4_check_key] = {
                        "is_consistent": False,
                        "message": "尚未選擇最終情緒",
                        "confidence": "",
                    }
                else:
                    st.session_state[step4_check_key] = evaluate_final_label_consistency(
                        selected_final,
                        step1_result,
                        step2_result,
                        step3_result,
                    )
                st.rerun()

        step4_check_result = st.session_state[step4_check_key]

        if step4_check_result:
            if step4_check_result["message"] == "尚未選擇最終情緒":
                st.warning("尚未選擇最終情緒")
            elif step4_check_result["is_consistent"]:
                box_class = "ok-box" if step4_check_result["confidence"] == "高信心" else "low-box"
                st.markdown(
                    f"""
                    <div class="{box_class}">
                        <b>{step4_check_result['message']}</b><br>
                        <b>{step4_check_result['confidence']}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="low-box">
                        <b>{step4_check_result['message']}</b><br>
                        <b>{step4_check_result['confidence']}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        note = st.text_area(
            "備註",
            value=default_note,
            height=120,
            key=f"note_{st.session_state.current_index}",
        )

        with col_save:
            save_clicked = st.button("儲存本筆標註", key=f"save_step4_{st.session_state.current_index}")

        if save_clicked:
            if not annotator_name:
                st.error("請先在左側輸入標註者姓名或編號。")
                st.stop()

            if selected_final is None:
                st.error("請先選擇最終主導情緒。")
                st.stop()

            final_consistency = evaluate_final_label_consistency(selected_final, step1_result, step2_result, step3_result)

            eye_selected = get_group_selected_features(st.session_state.step1_selected_core, "core", "眼睛")
            ear_selected = get_group_selected_features(st.session_state.step1_selected_core, "core", "耳朵")
            tail_selected = get_group_selected_features(st.session_state.step2_selected_aux, "aux", "尾巴")
            limb_selected = get_group_selected_features(st.session_state.step2_selected_aux, "aux", "四肢")

            record = {
                "record_id": annotator_name.strip(),
                "video_file": current_video_name,
                "eye_selected": json.dumps(eye_selected, ensure_ascii=False),
                "ear_selected": json.dumps(ear_selected, ensure_ascii=False),
                "tail_selected": json.dumps(tail_selected, ensure_ascii=False),
                "limb_selected": json.dumps(limb_selected, ensure_ascii=False),
                "final_emotion": selected_final,
                "confidence": final_consistency["confidence"],
                "is_multi_emotion": "是" if evaluate_step4_overall_result(step1_result, step2_result, step3_result)["mode"] == "multiple" else "否",
                "step1_selected_core_all": json.dumps(st.session_state.step1_selected_core, ensure_ascii=False),
                "step1_unknown_core_groups": json.dumps(st.session_state.step1_unknown_core, ensure_ascii=False),
                "step2_selected_aux_all": json.dumps(st.session_state.step2_selected_aux, ensure_ascii=False),
                "step2_unknown_aux_groups": json.dumps(st.session_state.step2_unknown_aux, ensure_ascii=False),
                "step3_selected_aux_all": json.dumps(st.session_state.step3_selected_aux, ensure_ascii=False),
                "step3_unknown_aux_groups": json.dumps(st.session_state.step3_unknown_aux, ensure_ascii=False),
            }
            upsert_annotation(record, annotator_name)

            try:
                append_to_google_sheet(record, annotator_name)
                st.success("已儲存本筆標註，並同步到 Google Sheet。")
            except Exception as e:
                st.warning(f"本地已儲存，但同步 Google Sheet 失敗：{e}")

            st.session_state.completed = len(load_existing_annotations(annotator_name))

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
