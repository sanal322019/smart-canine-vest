import os
import time
import numpy as np
from flask import Flask, jsonify, request
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from collections import deque

app = Flask(__name__)

# ================= PARAMETERS =================
MAX_POINTS = 200

STEP_NEG_THRESHOLD = -0.25
STEP_POS_THRESHOLD = 0.24

CUTOFF_FREQ = 2.0
SAMPLING_RATE = 20
SG_WINDOW = 11
SG_POLYORDER = 2
SAMPLE_TOLERANCE = 5

VALLEY_DELAY = 1.0

# ================= DATA STORAGE =================
stretch_data = deque(maxlen=MAX_POINTS)
ppg_data = deque(maxlen=MAX_POINTS)

step_state = 0
step_count = 0

sample_count = 0
valley_count = 0
counted_valley_abs = set()

minute_valley_count = 0
minute_valley_start = time.time()

last_valley_time = 0

last_peak = -9999
minute_beat_count = 0
minute_beat_start = time.time()

latest_map = "N/A"

# ================= FILTER =================
def lowpass_filter(data, cutoff=CUTOFF_FREQ, fs=SAMPLING_RATE, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, data)

# ================= ESP DATA UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():

    global step_state, step_count
    global sample_count, valley_count
    global counted_valley_abs
    global minute_valley_count
    global last_valley_time
    global last_peak
    global minute_beat_count
    global latest_map

    data = request.json

    roll = float(data["roll"])
    stretch = int(data["stretch"])
    ppg = int(data["ppg"])
    latest_map = data["map"]

    # ===== STEP DETECTION =====
    if step_state == 0:
        if roll <= STEP_NEG_THRESHOLD:
            step_state = 1
    elif step_state == 1:
        if roll >= STEP_POS_THRESHOLD:
            step_count += 1
            step_state = 0

    stretch_data.append(stretch)
    ppg_data.append(ppg)
    sample_count += 1

    # ===== BREATHING =====
    if len(stretch_data) >= SG_WINDOW:

        raw = np.array(stretch_data)

        lpf = lowpass_filter(raw)
        filtered = savgol_filter(lpf, SG_WINDOW, SG_POLYORDER)

        inverted = -filtered

        valleys,_ = find_peaks(inverted,distance=40,prominence=20)
        crests,_ = find_peaks(filtered,distance=40,prominence=20)

        valid=[]

        for v in valleys:
            left=[c for c in crests if c<v]

            if left:
                depth=filtered[max(left)]-filtered[v]

                if depth>=70:
                    valid.append(v)

        abs_indices=[
            sample_count-(MAX_POINTS-1-v)
            for v in valid
        ]

        if sample_count>=MAX_POINTS:

            for abs_idx in abs_indices:

                if not any(abs(abs_idx-c)<=SAMPLE_TOLERANCE for c in counted_valley_abs):

                    current=time.time()

                    if current-last_valley_time>VALLEY_DELAY:

                        valley_count+=1
                        minute_valley_count+=1
                        last_valley_time=current

                    counted_valley_abs.add(abs_idx)

    # ===== HEART RATE =====
    if len(ppg_data)>=SG_WINDOW:

        arr=np.array(ppg_data)

        filt=savgol_filter(arr,11,2)
        centered=filt-np.mean(filt)

        peaks,_=find_peaks(centered,distance=25,prominence=100)

        for p in peaks:

            abs_peak=sample_count-(MAX_POINTS-1-p)

            if abs_peak-last_peak>30:

                last_peak=abs_peak
                minute_beat_count+=1

    return jsonify({"status":"ok"})

# ================= API =================
@app.route("/data")
def data():

    return jsonify({
        "steps":step_count,
        "breaths":minute_valley_count,
        "heart":minute_beat_count,
        "map":latest_map
    })

# ================= DASHBOARD =================
@app.route("/")
def home():

    return """

    <h1>🐶 Smart Dog Vest</h1>

    <h2>🐾 Steps: <span id="s">0</span></h2>
    <h2>🫁 Breathing/min: <span id="b">0</span></h2>
    <h2>❤️ Heart/min: <span id="h">0</span></h2>

    <a id="m" target="_blank">Open Map</a>

<script>

setInterval(()=>{

fetch('/data')
.then(r=>r.json())
.then(d=>{

document.getElementById("s").innerText=d.steps
document.getElementById("b").innerText=d.breaths
document.getElementById("h").innerText=d.heart
document.getElementById("m").href=d.map

})

},1000)

</script>

"""

# ================= START =================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)