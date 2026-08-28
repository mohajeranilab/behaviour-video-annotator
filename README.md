# Active-Learning Annotation Studio

An active-learning annotation system that integrates a mouse behavior and sleep/wake detection pipeline with [Label Studio](https://labelstud.io/) for human-in-the-loop threshold tuning and behavioral dataset curation.

---

## 1. Quick Start Guide

### Step 1: Prerequisites
- Docker installed and running.
- Source video data placed in `./data/` (see [Data Formats](#2-data-formats)).

### Step 2: Configure Environment
Copy the environment template:
```bash
cp .env.example .env
```
*(Leave `LABEL_STUDIO_API_KEY` blank for now; you will obtain it after starting Label Studio).*

### Step 3: Set up annotation hotkey and behavior label
To rebind hotkeys or add custom categories, modify the `LABELING_CONFIG` XML template inside `backend/ls_client.py`:
```xml
<Choices name="category" toName="video" choice="multiple">
  <Choice value="Resting" hotkey="1, n" />
  <Choice value="Walking" hotkey="2, w" />
  ...
</Choices>
```
When the backend starts/restarts, it will automatically update the project configuration in Label Studio.

### Step 4: Launch Containers
```bash
docker compose up --build -d
```
> Wait approximately **30 seconds** after starting Docker for Label Studio to initialize its database and start the web server.

### Step 5: Obtain Label Studio API Key
1. Open your browser and navigate to **`http://localhost:8080`**.
2. Sign up an account.
3. In the top-right corner, click on your **User Avatar / Icon** → **Account & Settings**.
4. Copy the **Access Token**.
5. Open `.env` and paste your key:
   ```env
   LABEL_STUDIO_API_KEY=your_copied_token_here
   ```
   *(Ensure there are no leading/trailing spaces or quotes).*

### Step 6: Start the Backend Controller
Recreate the backend container with the updated token:
```bash
docker compose up -d backend
```
> **Important**: Always use `docker compose up -d backend` rather than `docker compose restart backend`. A plain restart does **not** reload modified variables from `.env`. Wait ~45 seconds for the backend to connect, create the project, and generate the initial waves. It may take longer time if there is no cached feature information.

### Step 7: Configure UI Settings in Label Studio
Open the project (**Mouse Behavior Annotation**) in Label Studio:
1. **Keyboard Shortcut Configuration**: Review or adjust keyboard shortcuts (under UI settings) for video playback (e.g. Space to Play/Pause) and Submit to facilitate annotation process.
2. **Enable Continuous Flow**: Click **"Label tasks as display"** to start annotation with automatic task loading on submit enabled.

### Step 8: Export Annotation result
The result can be exported on the UI panel, in the format of your choice (possible options includes json and csv). The metadata will include some of the information generated and stored by the backend controller (e.g. the predicted awakeness of the clip, the wave and index of the clips), and the annotation result as well as the information about the clip itself will also be recorded in the file.

---

### Other Tips & Functions
- **Hotkey for Labeling**: Rebind the hotkey of your choice and use them instead of clicking on the label makes the annotation process much more fluent.
- **Multi-Labeling**: Current system allows you to select more than one behavior. (e.g., `Rearing` + `Sniffing`)
- **Playback Controls**: Use the on-screen video speed selector (`0.5x`, `1.0x`, `1.5x`, `2.0x`) to accelerate the annotation process, and use the history/back buttons in the top navigation bar to revisit and revise previous clips if a mistake was made.

### View Live Logs
```bash
# View backend controller activity
docker compose logs -f backend

# View Label Studio server logs
docker compose logs -f labelstudio
```

### Inspect Threshold Convergence
All threshold updates and historical metrics are logged to:
- `state/controller.log`
- `state/controller_state.json` (under `threshold_history`)

### Reset the System
If you want to wipe all projects, user accounts, and state to start from scratch:
```bash
# 1. Stop containers and remove persistent database volume
docker compose down -v

# 2. Clear generated clips and controller state
rm -rf clips/*.mp4 state/controller_state.json

# 3. Start clean
docker compose up --build -d
```

---

## 2. Data Formats

### Input Data (`./data/`)
Place your source recordings and detection files into `./data/`:

| File Pattern | Description |
| :--- | :--- |
| `<prefix>.compressed.mp4` | The compressed continuous cage recording (e.g. `20260410_000000_dca632e87112.compressed.mp4`). |
| `<prefix>.detection_v1.parquet` | YOLO bounding box tracking data (`frame_index`, `x1, y1, x2, y2`, `confidence`). |
| `<prefix>.features.parquet` | Pre-extracted or cached motion features per frame. If absent, the backend automatically extracts them on first run. |

#### Feature Parquet Columns
- `frame_index`: integer; index
- `time_sec`: float; seconds
- `disp`: float; pixels
- `roi_diff`: float; mean pixel intensity
- `area_change`: float; ratio
- `confidence`: float; normalized scale
- `bbox_cx`, `bbox_cy`: float; pixels
- `bbox_w`, `bbox_h`: float; pixels

---

### Output Data

#### 1. Generated Clips (`./clips/`)
- Format: `<prefix>_<predicted_label>_<start_frame>_<end_frame>.mp4`
- Specifications: 3-second duration, H.264 (`libx264`), `yuv420p` pixel format.

#### 2. Label Studio Export Format
You can export annotations from Label Studio at any time in CSV or JSON format:

* **CSV Export (`project-*.csv`)**:
  - `id`: Unique task ID.
  - `category`: JSON list of selected behaviors (e.g. `["Walking"]` or `["Resting"]`).
  - `video`: Internal video path (`/data/local-files/?d=clips/<filename>.mp4`).
  - `video_prefix`: The recording prefix (e.g. `20260410_000000_dca632e87112`).
  - `predicted_label`: Model prediction at sampling time (`awake` or `sleeping`).
  - `start_sec` / `end_sec`: Clip start and end timestamps in seconds within the source recording.
  - `wave_id`: Active-learning wave number.
  - `annotator`: ID of the annotator.
  - `created_at` / `updated_at`: Timestamps.
  - `lead_time`: Time spent annotating the task (in seconds).

* **JSON Export (`project-*.json`)**: Full nested data structure containing task metadata, choice unique IDs, draft history, and user details.

---

## 3. Configuration & Environment Variables

### Variables (`.env`)

| Variable | Default | Description |
| :--- | :---: | :--- |
| `LABEL_STUDIO_API_KEY` | *(Required)* | Label Studio access token from Account & Settings. |
| `POLL_INTERVAL_SEC` | `20` | Interval (in seconds) at which the controller checks for wave completion. |
| `CLIPS_PER_WAVE` | `20` | Total number of clips generated per wave. |
| `SLEEP_CLIPS_PER_WAVE` | `3` | Number of predicted-sleeping clips per wave (the remainder are predicted-awake). |
| `CONFIG_PATH` | *(empty)* | Optional path to custom `DayNightConfig` JSON file. |

### Default Day / Night Thresholds

| Feature | Day Baseline (07:00 – 19:00) | Night Baseline (19:00 – 07:00) |
| :--- | :---: | :---: |
| `disp_thresh` (Displacement) | `8.7546` | `0.7576` |
| `roi_thresh` (ROI Movement) | `2.515` | `1.5247` |
| `area_thresh` (Active Area) | `0.0162` | `0.0165` |
---

## 4. System Overview & Architecture

The system automatically extracts video clips of mouse behavior from continuous home-cage recordings, pushes them to a local Label Studio instance for human verification and multi-class behavior labeling, and dynamically updates the detection thresholds of sleeping mice using a precision/recall feedback loop.

The model generate two waves of clips (each with `CLIPS_PER_WAVE` clips among which `SLEEP_CLIPS_PER_WAVE` are sleeping clips), and adjust the thresholds after the annotator finishes annotating current wave of clips before generating the next wave of clips. The threshold is adjusted by 5% if either recall or precision is lower than 90%.

For each wave of the video, at most 4 random clips (configurable in `pipeline/config.py`) are generated from the same video, and it is guaranteed that the annotator will not annotate overlapping clips. However, the sequence of annotation is currently fixed (randomized sequence might be preferred).
