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

VIDEOS = [
    {
        "name": name,
        "url": f"{BASE_URL}{name}",
    }
    for name in video_names
]
