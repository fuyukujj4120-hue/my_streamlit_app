from datetime import datetime
from pathlib import Path
import hashlib

import pandas as pd
import requests
import streamlit as st


# =========================================================
# 1. Streamlit 基本設定
# =========================================================
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
    div[data-testid="stRadio"] > label p {
        font-size: 22px !important;
        font-weight: 700 !important;
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
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        border-radius: 10px !important;
        font-size: 13px !important;
        padding: 8px 10px !important;
    }
    div[data-testid="column"] div[data-testid="stButton"] > button {
        border-radius: 10px !important;
        font-size: 15px !important;
        padding: 10px 18px !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2. 影片來源：Google Cloud Storage
# =========================================================
APP_TITLE = "貓咪情緒標註系統"

# 你的 Bucket 名稱就是 annotation_v1，影片放在 Bucket 根目錄。
# 例如 v001_c003 的網址會變成：
# https://storage.googleapis.com/annotation_v1/v001_c003.mp4
BASE_URL = "https://storage.googleapis.com/annotation_v1/"

CLIP_IDS = [
    "v641_c002",
    "v698_c004",
    "v603_c002",
    "v199_c004",
    "v728_c002",
    "v012_c011",
    "v350_c001",
    "v685_c001",
    "v514_c002",
    "v532_c003",
    "v229_c002",
    "v269_c001",
    "v551_c003",
    "v386_c001",
    "v618_c009",
    "v208_c001",
    "v743_c001",
    "v653_c003",
    "v378_c004",
    "v774_c003",
    "v532_c004",
    "v029_c004",
    "v134_c006",
    "v739_c002",
    "v560_c009",
    "v127_c001",
    "v591_c002",
    "v485_c001",
    "v637_c002",
    "v474_c002",
    "v176_c003",
    "v699_c002",
    "v431_c009",
    "v026_c001",
    "v714_c001",
    "v430_c001",
    "v765_c004",
    "v586_c001",
    "v242_c001",
    "v097_c002",
    "v629_c005",
    "v406_c002",
    "v666_c002",
    "v512_c002",
    "v166_c003",
    "v319_c003",
    "v772_c005",
    "v173_c012",
    "v334_c003",
    "v702_c001",
    "v757_c001",
    "v769_c001",
    "v028_c001",
    "v732_c003",
    "v212_c003",
    "v575_c001",
    "v182_c003",
    "v055_c004",
    "v699_c010",
    "v070_c001",
    "v439_c003",
    "v602_c002",
    "v644_c002",
    "v665_c002",
    "v002_c004",
    "v649_c009",
    "v344_c002",
    "v404_c007",
    "v049_c001",
    "v668_c003",
    "v566_c003",
    "v569_c002",
    "v205_c001",
    "v539_c010",
    "v164_c001",
    "v541_c006",
    "v001_c003",
    "v275_c002",
    "v593_c005",
    "v113_c002",
    "v135_c001",
    "v401_c017",
    "v131_c002",
    "v758_c002",
    "v538_c001",
    "v545_c004",
    "v160_c001",
    "v625_c001",
    "v080_c001",
    "v294_c001",
    "v089_c007",
    "v102_c001",
    "v421_c002",
    "v363_c003",
    "v156_c002",
    "v461_c002",
    "v435_c009",
    "v072_c002",
    "v205_c002",
    "v516_c004",
    "v481_c004",
    "v465_c007",
    "v288_c002",
    "v001_c005",
    "v321_c006",
    "v292_c005",
    "v430_c004",
    "v734_c009",
    "v195_c001",
    "v021_c002",
    "v505_c007",
    "v683_c001",
    "v065_c001",
    "v111_c001",
    "v332_c001",
    "v723_c001",
    "v133_c001",
    "v411_c001",
    "v356_c001",
    "v476_c002",
    "v005_c003",
    "v057_c004",
    "v496_c003",
    "v142_c001",
    "v437_c001",
    "v654_c001",
    "v092_c002",
    "v009_c005",
    "v306_c010",
    "v404_c003",
    "v761_c002",
    "v247_c004",
    "v225_c002",
    "v519_c003",
    "v734_c004",
    "v556_c001",
    "v524_c003",
    "v696_c002",
    "v259_c001",
    "v640_c002",
    "v621_c013",
    "v029_c003",
    "v046_c004",
    "v372_c003",
    "v361_c002",
    "v027_c003",
    "v198_c002",
    "v682_c002",
    "v150_c002",
    "v204_c004",
    "v704_c009",
    "v673_c002",
    "v659_c001",
    "v491_c003",
    "v043_c005",
    "v292_c021",
    "v092_c004",
    "v326_c009",
    "v365_c002",
    "v365_c001",
    "v065_c005",
    "v608_c001",
    "v330_c002",
    "v196_c010",
    "v599_c002",
    "v498_c002",
    "v564_c002",
    "v196_c009",
    "v196_c001",
    "v559_c003",
    "v085_c010",
    "v192_c003",
    "v107_c001",
    "v499_c004",
    "v170_c001",
    "v617_c001",
    "v116_c001",
    "v222_c002",
    "v186_c004",
    "v456_c004",
    "v388_c003",
    "v058_c005",
    "v059_c006",
    "v537_c006",
    "v214_c001",
    "v163_c004",
    "v311_c003",
    "v473_c001",
    "v446_c003",
    "v491_c001",
    "v375_c001",
    "v131_c001",
    "v338_c004",
    "v766_c003",
    "v265_c002",
    "v559_c001",
    "v121_c006",
    "v655_c011",
    "v175_c001",
    "v004_c010",
    "v422_c002",
    "v742_c003",
    "v035_c001",
    "v609_c004",
    "v324_c003",
    "v528_c002",
    "v022_c004",
    "v573_c001",
    "v218_c001",
    "v294_c003",
    "v670_c002",
    "v468_c005",
    "v404_c011",
    "v148_c002",
    "v343_c003",
    "v379_c003",
    "v401_c013",
    "v391_c005",
    "v717_c002",
    "v467_c001",
    "v016_c011",
    "v273_c002",
    "v477_c001",
    "v041_c001",
    "v536_c005",
    "v260_c003",
    "v263_c003",
    "v252_c001",
    "v454_c001",
    "v561_c003",
    "v451_c001",
    "v008_c011",
    "v417_c003",
    "v018_c021",
    "v741_c001",
    "v398_c001",
    "v122_c002",
    "v296_c001",
    "v678_c001",
    "v425_c005",
    "v192_c004",
    "v715_c002",
    "v656_c002",
    "v416_c001",
    "v601_c004",
    "v733_c002",
    "v730_c001",
    "v281_c006",
    "v353_c002",
    "v518_c011",
    "v749_c001",
    "v404_c015",
    "v531_c001",
    "v325_c001",
    "v411_c002",
    "v075_c004",
    "v611_c003",
    "v232_c005",
    "v307_c003",
    "v247_c001",
    "v691_c004",
    "v292_c022",
    "v597_c003",
    "v184_c004",
    "v711_c001",
    "v771_c003",
    "v623_c002",
    "v382_c002",
    "v139_c001",
    "v149_c002",
    "v754_c001",
    "v300_c003",
    "v604_c001",
    "v105_c002",
    "v601_c002",
    "v092_c006",
    "v428_c002",
    "v596_c002",
    "v083_c004",
    "v732_c002",
    "v632_c003",
    "v302_c006",
    "v578_c001",
    "v533_c006",
    "v700_c003",
    "v237_c001",
    "v750_c001",
    "v392_c001",
    "v481_c005",
    "v615_c001",
    "v764_c001",
    "v503_c001",
    "v462_c002",
    "v621_c009",
    "v582_c002",
    "v582_c003",
    "v285_c001",
    "v693_c002",
    "v550_c003",
    "v268_c001",
]

VIDEOS = [
    {
        "clip_id": clip_id,
        "name": f"{clip_id}.mp4",
        "url": f"{BASE_URL}{clip_id}.mp4",
    }
    for clip_id in CLIP_IDS
]


# =========================================================
# 3. 情緒類別
# =========================================================
MAIN_EMOTIONS = [
    "害怕",
    "憤怒/狂怒",
    "歡樂/玩耍",
    "滿意",
    "興趣",
    "中性/其他",
    "uncertain",
]

EMOTION_ICONS = {
    "害怕": "😿",
    "憤怒/狂怒": "😾",
    "歡樂/玩耍": "😺",
    "滿意": "😽",
    "興趣": "🐾",
    "中性/其他": "➖",
    "uncertain": "❓",
}

# 這裡只保留「情緒定義」，不再要求標註眼睛、耳朵、尾巴、身體或行為。
EMOTION_SCHEMA = {
    "害怕": {
        "definition": "由立即感知到的危險或危險的威脅引起，表現為警惕和試圖撤退或逃跑。"
    },
    "憤怒/狂怒": {
        "definition": "由執行行動或實現目標的願望受挫，或資源競爭引起，表現為攻擊性或攻擊威脅。"
    },
    "歡樂/玩耍": {
        "definition": "表現為非功能性行為，包括運動遊戲、社交遊戲或物件遊戲。"
    },
    "滿意": {
        "definition": "由需求和願望得到滿足而產生的正向情緒狀態，表現為休息、平靜和親和。"
    },
    "興趣": {
        "definition": "由新奇或顯著刺激引起，表現為注意、定向或探索行為。"
    },
}

DEFINITION_IMAGE_MAP = {
    "害怕": Path("images/fear.png"),
    "憤怒/狂怒": Path("images/anger.png"),
    "歡樂/玩耍": Path("images/joy.png"),
    "滿意": Path("images/contentment.png"),
    "興趣": Path("images/interest.png"),
}


# =========================================================
# 4. Google Sheet Apps Script
# =========================================================
# 沿用你原本的 Apps Script Web App。
SHEET_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbzA_0AnSkFSeN6GFLr1wDsvx-l28-5a3s605l9CV6QwwTfcJ4GejNepx2yOIjX7M85m/exec"
SHEET_SECRET = "my_cat_annotation_secret"


# =========================================================
# 5. 情緒定義 Dialog
# =========================================================
@st.dialog("情緒定義")
def show_emotion_dialog(emotion_name: str):
    item = EMOTION_SCHEMA[emotion_name]
    st.subheader(f"{EMOTION_ICONS.get(emotion_name, '')} {emotion_name}")

    img_path = DEFINITION_IMAGE_MAP.get(emotion_name)
    if img_path and img_path.exists():
        st.image(str(img_path), use_container_width=True)

    st.write(f"**定義：** {item['definition']}")


# =========================================================
# 6. Google Sheet 讀寫
# =========================================================
def append_to_google_sheet(record: dict, annotator_name: str):
    """
    為了相容你舊版 Apps Script / Google Sheet：
    - 新版主要欄位：clip_id / emotion
    - 同時送 video_file / final_emotion / step1_selected_emotion
    """
    payload = {
        **record,
        "annotator_name": annotator_name.strip(),
        "secret": SHEET_SECRET,
    }

    resp = requests.post(
        SHEET_WEBHOOK_URL,
        json=payload,
        timeout=20,
    )
    resp.raise_for_status()

    try:
        data = resp.json()
    except Exception:
        raise ValueError(
            "Google Apps Script 回傳的不是 JSON。回傳內容前 500 字："
            + resp.text[:500]
        )

    if not data.get("ok"):
        raise ValueError(data.get("error", "Unknown Google Sheet error"))


def load_annotations_from_google_sheet(annotator_name: str):
    if not annotator_name or not annotator_name.strip():
        return []

    params = {
        "secret": SHEET_SECRET,
        "action": "get_by_annotator",
        "annotator_name": annotator_name.strip(),
    }

    resp = requests.get(
        SHEET_WEBHOOK_URL,
        params=params,
        timeout=20,
    )

    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code}：{resp.text[:500]}")

    if not resp.text or not resp.text.strip():
        raise ValueError(
            "Google Apps Script 回傳空白內容，請確認 doGet 已部署為 Web App。"
        )

    try:
        data = resp.json()
    except Exception:
        raise ValueError(
            "Google Apps Script 回傳的不是 JSON。請檢查 Web App 部署權限與網址是否為 /exec。 "
            "回傳內容前 500 字：" + resp.text[:500]
        )

    if not data.get("ok"):
        raise ValueError(data.get("error", "Unknown Google Sheet read error"))

    return data.get("records", [])


# =========================================================
# 7. 影片 / 標註資料工具函式
# =========================================================
def load_video_files():
    return VIDEOS


def render_small_video(video_item: dict):
    st.video(video_item["url"])


def get_clip_id(video_item: dict):
    return video_item["clip_id"]


def get_video_file(video_item: dict):
    return video_item["name"]


def compute_store_key(annotator_name: str, clip_id: str):
    raw = f"{annotator_name.strip()}::{clip_id}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def get_annotations_store():
    if "annotations_store" not in st.session_state:
        st.session_state["annotations_store"] = {}
    return st.session_state["annotations_store"]


def get_record_emotion(record: dict):
    """相容新舊 Google Sheet 欄位。"""
    if not record:
        return None

    for key in ("emotion", "final_emotion", "step1_selected_emotion"):
        value = str(record.get(key, "") or "").strip()
        if value in MAIN_EMOTIONS:
            return value

    return None


def get_record_clip_id(record: dict):
    """
    新資料直接讀 clip_id。
    舊資料若只有 video_file，例如 v001_c003.mp4，則自動轉成 v001_c003。
    """
    if not record:
        return ""

    clip_id = str(record.get("clip_id", "") or "").strip()
    if clip_id:
        return clip_id

    video_file = str(record.get("video_file", "") or "").strip()
    if video_file:
        return Path(video_file).stem

    return ""


def normalize_record(record: dict, annotator_name: str):
    clip_id = get_record_clip_id(record)
    emotion = get_record_emotion(record)
    timestamp = str(record.get("timestamp", "") or "").strip()

    return {
        "annotator_name": annotator_name.strip(),
        "clip_id": clip_id,
        "emotion": emotion or "",
        "timestamp": timestamp,
    }


def get_saved_record(annotator_name: str, clip_id: str):
    if not annotator_name:
        return None

    return get_annotations_store().get(
        compute_store_key(annotator_name, clip_id)
    )


def upsert_annotation(record: dict, annotator_name: str):
    clip_id = record["clip_id"]
    key = compute_store_key(annotator_name, clip_id)
    get_annotations_store()[key] = record


def get_annotations_df(annotator_name: str):
    """只回傳目前 annotation_v1 這 300 支 clip 的標註。"""
    columns = ["annotator_name", "clip_id", "emotion", "timestamp"]

    if not annotator_name:
        return pd.DataFrame(columns=columns)

    name = annotator_name.strip()
    valid_clip_ids = set(CLIP_IDS)

    rows = [
        row
        for row in get_annotations_store().values()
        if row.get("annotator_name", "") == name
        and row.get("clip_id", "") in valid_clip_ids
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)

    for col in columns:
        if col not in df.columns:
            df[col] = ""

    df = df[columns]

    # 依照 CLIP_IDS 原本的指定順序排序，而不是照字母排序。
    order_map = {clip_id: i for i, clip_id in enumerate(CLIP_IDS)}
    df["__order"] = df["clip_id"].map(order_map)
    df = df.sort_values("__order").drop(columns="__order")

    return df.reset_index(drop=True)


def sync_google_sheet_records_to_session(annotator_name: str):
    records = load_annotations_from_google_sheet(annotator_name)
    valid_clip_ids = set(CLIP_IDS)

    for raw_record in records:
        record = normalize_record(raw_record, annotator_name)
        if record["clip_id"] in valid_clip_ids:
            upsert_annotation(record, annotator_name)

    return records


def count_completed(annotator_name: str):
    df = get_annotations_df(annotator_name)
    if df.empty:
        return 0

    return int((df["emotion"].astype(str).str.strip() != "").sum())


def find_first_unfinished_video_index(annotator_name: str):
    df = get_annotations_df(annotator_name)

    if df.empty:
        return 0

    finished_clip_ids = set(
        df.loc[
            df["emotion"].astype(str).str.strip() != "",
            "clip_id",
        ].astype(str)
    )

    for idx, video in enumerate(st.session_state.videos):
        if get_clip_id(video) not in finished_clip_ids:
            return idx

    return len(st.session_state.videos)


def load_progress_and_jump(annotator_name: str):
    name = annotator_name.strip()
    if not name:
        return

    records = sync_google_sheet_records_to_session(name)
    target_index = find_first_unfinished_video_index(name)

    st.session_state["annotator_name"] = name
    st.session_state["loaded_annotator_name"] = name
    st.session_state["completed"] = count_completed(name)
    st.session_state["current_index"] = target_index
    st.session_state["page"] = "annotation"

    st.session_state["google_sheet_load_message"] = (
        f"✅ 已讀取 {name} 的 Google Sheet 紀錄：原始紀錄 {len(records)} 筆，"
        f"本批 300 支影片目前完成 {st.session_state.completed} 支。"
    )


def clear_emotion_widget(video_index: int):
    key = f"emotion_{video_index}"
    if key in st.session_state:
        del st.session_state[key]


def go_to_instruction():
    st.session_state.page = "instruction"
    st.rerun()


def init_session(videos):
    defaults = {
        "page": "instruction",
        "current_index": 0,
        "videos": videos,
        "completed": 0,
        "annotations_store": {},
        "loaded_annotator_name": "",
        "google_sheet_load_message": "",
        "annotator_name": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_record(annotator_name: str, clip_id: str, emotion: str):
    """
    Streamlit / CSV 主要資料：
    annotator_name / clip_id / emotion / timestamp

    為了相容舊 Apps Script，同時送：
    video_file / step1_selected_emotion / final_emotion
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "annotator_name": annotator_name.strip(),
        "clip_id": clip_id,
        "video_file": f"{clip_id}.mp4",
        "emotion": emotion,
        "step1_selected_emotion": emotion,
        "final_emotion": emotion,
        "timestamp": timestamp,
    }


# =========================================================
# 8. 頁面標題
# =========================================================
st.markdown(
    f'<h1 style="font-size:28px;font-weight:800;color:#222;margin-bottom:2px;">'
    f'🐱 {APP_TITLE}</h1>'
    f'<p style="color:#888;font-size:14px;margin-top:0;margin-bottom:20px;">'
    f'觀看影片 → 選擇情緒 → 儲存並下一段</p>',
    unsafe_allow_html=True,
)

videos = load_video_files()
init_session(videos)


# =========================================================
# 9. Sidebar
# =========================================================
with st.sidebar:
    st.markdown(
        '<div style="font-size:17px;font-weight:800;color:#222;margin-bottom:12px;">'
        '📊 標註進度</div>',
        unsafe_allow_html=True,
    )

    total = len(st.session_state.videos)
    completed = st.session_state.completed
    pct = int(completed / total * 100) if total else 0

    st.markdown(
        f"""
        <div style="background:#f5f4ff;border-radius:10px;padding:12px 14px;
                    border:1px solid #d4d0f5;margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;font-size:13px;
                        color:#555;margin-bottom:6px;">
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

    annotator_name = st.text_input(
        "👤 標註者姓名 / 編號",
        value=st.session_state.get("annotator_name", ""),
        placeholder="請輸入姓名或編號…",
        help="輸入後會讀取同名 Google Sheet 紀錄，並跳到第一支尚未完成的影片。",
    )

    input_name = annotator_name.strip()
    loaded_name = st.session_state.get("loaded_annotator_name", "").strip()

    if input_name:
        st.session_state["annotator_name"] = input_name

        if input_name != loaded_name:
            try:
                load_progress_and_jump(input_name)
                st.rerun()
            except Exception as e:
                # Google Sheet 暫時讀不到時，仍允許繼續使用介面。
                st.session_state["loaded_annotator_name"] = input_name
                st.session_state["completed"] = count_completed(input_name)
                st.warning(f"讀取 Google Sheet 失敗：{e}")
    else:
        st.session_state["annotator_name"] = ""

    if st.session_state.get("google_sheet_load_message"):
        st.success(st.session_state["google_sheet_load_message"])

    if st.button("🏠 回到說明頁", use_container_width=True):
        go_to_instruction()

    if (
        st.session_state.page == "annotation"
        and st.session_state.current_index < len(st.session_state.videos)
    ):
        sidebar_video = st.session_state.videos[st.session_state.current_index]
        st.markdown("---")
        st.markdown(
            f'<div style="font-size:12px;color:#888;margin-bottom:4px;">'
            f'🎬 {get_clip_id(sidebar_video)}</div>'
            f'<div style="font-size:12px;color:#aaa;margin-bottom:6px;">'
            f'索引：{st.session_state.current_index + 1} / {total}</div>',
            unsafe_allow_html=True,
        )
        render_small_video(sidebar_video)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:13px;font-weight:700;color:#555;margin-bottom:8px;">'
        '📖 情緒定義快速查看</div>',
        unsafe_allow_html=True,
    )

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


# =========================================================
# 10. 說明頁
# =========================================================
if st.session_state.page == "instruction":
    st.subheader("標註規則")
    st.markdown("1. 請先完整觀看整段影片，再進行情緒判斷。")
    st.markdown("2. 每段影片只選擇 **一個最主要的情緒**。")
    st.markdown("3. 若不屬於五種主要情緒，可選擇 **中性/其他**。")
    st.markdown("4. 若兩種以上情緒同樣重要，或無法判定單一主要情緒，選擇 **uncertain**。")

    st.subheader("情緒定義與判斷參考")
    for emotion_name, item in EMOTION_SCHEMA.items():
        icon = EMOTION_ICONS.get(emotion_name, "")
        with st.expander(f"{icon} {emotion_name}", expanded=False):
            img_path = DEFINITION_IMAGE_MAP.get(emotion_name)
            if img_path and img_path.exists():
                st.image(str(img_path), use_container_width=True)
            st.write(f"**定義：** {item['definition']}")

    st.info("每支影片只需要選擇一次主要情緒，然後儲存。")

    start_disabled = (
        annotator_name.strip() == ""
        or len(st.session_state.videos) == 0
    )

    if st.button(
        "✅ 我已閱讀完畢，開始標註",
        disabled=start_disabled,
        type="primary",
    ):
        st.session_state.page = "annotation"
        st.rerun()


# =========================================================
# 11. 標註頁
# =========================================================
else:
    if len(st.session_state.videos) == 0:
        st.error("沒有可標註的影片。")
        st.stop()

    # 全部完成
    if st.session_state.current_index >= len(st.session_state.videos):
        st.success("🎉 這 300 支影片全部標註完成了！")

        df_all = get_annotations_df(annotator_name)
        if not df_all.empty:
            st.dataframe(df_all, use_container_width=True)

            csv_bytes = df_all.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ 下載標註結果 CSV",
                data=csv_bytes,
                file_name=f"annotations_{annotator_name.strip()}.csv",
                mime="text/csv",
            )
        st.stop()

    current_video = st.session_state.videos[st.session_state.current_index]
    current_clip_id = get_clip_id(current_video)
    current_video_file = get_video_file(current_video)

    saved_record = (
        get_saved_record(annotator_name, current_clip_id)
        if annotator_name
        else None
    )

    st.markdown(
        f'<div style="background:#f5f4ff;border:1.5px solid #d4d0f5;border-radius:10px;'
        f'padding:10px 16px;font-size:14px;color:#534ab7;font-weight:600;margin-bottom:12px;">'
        f'🎬 {current_clip_id} &nbsp;&nbsp; '
        f'<span style="color:#999;font-weight:400;">{current_video_file}</span></div>',
        unsafe_allow_html=True,
    )

    if saved_record:
        st.info("📝 這支影片已經標過，可以修改情緒後重新儲存。")

    st.markdown(
        '<div class="section-title">選擇情緒</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    emotion_key = f"emotion_{st.session_state.current_index}"
    saved_emotion = get_record_emotion(saved_record)

    if emotion_key not in st.session_state:
        st.session_state[emotion_key] = saved_emotion

    selected_emotion = st.radio(
        "請選擇這段影片最主要的情緒",
        MAIN_EMOTIONS,
        index=(
            MAIN_EMOTIONS.index(st.session_state[emotion_key])
            if st.session_state[emotion_key] in MAIN_EMOTIONS
            else None
        ),
        key=emotion_key,
    )

    if selected_emotion in EMOTION_SCHEMA:
        st.caption(
            f"{EMOTION_ICONS.get(selected_emotion, '')} "
            f"{EMOTION_SCHEMA[selected_emotion]['definition']}"
        )
    elif selected_emotion == "中性/其他":
        st.caption("➖ 不明顯屬於五種主要情緒，或偏中性、一般日常狀態。")
    elif selected_emotion == "uncertain":
        st.caption("❓ 無法判定單一主要情緒，或多種情緒同樣重要。")

    st.divider()

    # -----------------------------------------------------
    # CSV 預覽 / 下載
    # -----------------------------------------------------
    df_mine = get_annotations_df(annotator_name) if annotator_name else pd.DataFrame()
    csv_bytes = None

    if annotator_name and selected_emotion:
        preview_record = {
            "annotator_name": annotator_name.strip(),
            "clip_id": current_clip_id,
            "emotion": selected_emotion,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if not df_mine.empty:
            preview_df = pd.concat(
                [
                    df_mine[df_mine["clip_id"] != current_clip_id],
                    pd.DataFrame([preview_record]),
                ],
                ignore_index=True,
            )

            order_map = {clip_id: i for i, clip_id in enumerate(CLIP_IDS)}
            preview_df["__order"] = preview_df["clip_id"].map(order_map)
            preview_df = (
                preview_df
                .sort_values("__order")
                .drop(columns="__order")
                .reset_index(drop=True)
            )
        else:
            preview_df = pd.DataFrame([preview_record])

        csv_bytes = preview_df.to_csv(index=False).encode("utf-8-sig")

    c_download, c_save = st.columns([2, 3])

    with c_download:
        st.download_button(
            "⬇️ 下載我的標註 CSV",
            data=csv_bytes if csv_bytes is not None else b"",
            file_name=(
                f"annotations_{annotator_name.strip()}.csv"
                if annotator_name
                else "annotations.csv"
            ),
            mime="text/csv",
            disabled=(csv_bytes is None),
            use_container_width=True,
        )

    with c_save:
        save_clicked = st.button(
            "☁️ 儲存並下一段",
            type="primary",
            use_container_width=True,
            disabled=(
                not annotator_name.strip()
                or selected_emotion is None
            ),
            key=f"save_next_{st.session_state.current_index}",
        )

    # -----------------------------------------------------
    # 儲存
    # -----------------------------------------------------
    if save_clicked:
        record = build_record(
            annotator_name=annotator_name,
            clip_id=current_clip_id,
            emotion=selected_emotion,
        )

        # Session / CSV 只保留真正需要的四欄。
        local_record = {
            "annotator_name": record["annotator_name"],
            "clip_id": record["clip_id"],
            "emotion": record["emotion"],
            "timestamp": record["timestamp"],
        }

        upsert_annotation(local_record, annotator_name)
        st.session_state.completed = count_completed(annotator_name)

        try:
            append_to_google_sheet(record, annotator_name)
            st.toast("✅ 已同步到 Google Sheet")
        except Exception as e:
            st.warning(
                f"本次標註已保留在目前 Session，但同步 Google Sheet 失敗：{e}"
            )

        if st.session_state.current_index < len(st.session_state.videos) - 1:
            st.session_state.current_index += 1
            clear_emotion_widget(st.session_state.current_index)
            st.rerun()
        else:
            st.session_state.current_index = len(st.session_state.videos)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # -----------------------------------------------------
    # 上一段 / 下一段
    # -----------------------------------------------------
    col_prev, col_next = st.columns(2)

    with col_prev:
        if st.button(
            "◀ 上一段",
            disabled=st.session_state.current_index == 0,
            use_container_width=True,
        ):
            st.session_state.current_index -= 1
            st.rerun()

    with col_next:
        if st.button(
            "下一段 ▶",
            disabled=(
                st.session_state.current_index
                >= len(st.session_state.videos) - 1
            ),
            use_container_width=True,
        ):
            st.session_state.current_index += 1
            st.rerun()
