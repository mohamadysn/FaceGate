# FaceGate

Real-time **face recognition** (desktop app + CLI) with optional eye / gaze
tracking. Built with Python, OpenCV, and InsightFace.

**Developed by [Mohamad Yassine](https://github.com/mohamadysn)** · Repository: [github.com/mohamadysn/FaceGate](https://github.com/mohamadysn/FaceGate)

**License:** [MIT](LICENSE)

**Documentation:** [mohamadysn.github.io/FaceGate](https://mohamadysn.github.io/FaceGate/) — sources in [`docs/`](docs/).

## Repository layout

```
FaceGate/
├── README.md
├── requirements.txt          # desktop app + recognition
├── requirements-train.txt    # optional ArcFace/RetinaFace remakes
├── docs/                     # Quarto website (GitHub Pages)
├── .github/workflows/docs.yml
├── .gitignore
└── eye-tracking/
    ├── app/                  # Desktop GUI + pipeline entry points
    │   ├── run_desktop.py / launch.py
    │   ├── FaceGate(.bat|.command)   # OS launchers
    │   └── install_desktop_shortcut.py
    ├── common/               # Shared library (gallery, engine, camera, …)
    ├── webcam-capture/
    ├── camera-calibration/
    ├── face-recognition/    # Enroll / recognize CLIs + gallery/
    ├── pupil-segmentation/
    ├── gaze-estimation/
    ├── performance-metrics/
    ├── docs/                 # Specs / slides (ODT, ODP)
    └── tests/
```

Personal gallery embeddings, calibration images, and logs are **gitignored**.

## Supported platforms

| OS | App | Camera backend | Desktop shortcut |
|----|-----|----------------|------------------|
| **Linux** | Yes (tested on Ubuntu) | V4L2 | `.desktop` via installer |
| **Windows** | Yes | DirectShow / MSMF | `.bat` + Start Menu `.lnk` |
| **macOS** | Yes | AVFoundation | `.command` |

Core stack is cross-platform Python (Tkinter + OpenCV + InsightFace). Platform-specific bits are limited to camera backends and shortcut installers.

## Requirements

- Python 3.10+
- Webcam (for live modes)
- Tkinter (usually included with Python; on Linux: `python3-tk`)

```bash
git clone https://github.com/mohamadysn/FaceGate.git
cd FaceGate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows** (cmd / PowerShell)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For experimental training code under `face-recognition/remake`:

```bash
pip install -r requirements-train.txt
```

## Desktop app (recommended)

```bash
cd eye-tracking
# Linux/macOS:
source ../.venv/bin/activate
python app/launch.py
# Windows:
#   ..\.venv\Scripts\activate
#   python app\launch.py
```

Or use the OS launcher in `eye-tracking/app/`:

| OS | Double-click / run |
|----|--------------------|
| Linux | `FaceGate` |
| Windows | `FaceGate.bat` |
| macOS | `FaceGate.command` |

Install a Desktop / Start-menu shortcut:

```bash
cd eye-tracking
python app/install_desktop_shortcut.py
```

- **Linux**: if the icon opens as text → right-click → **Allow Launching**
- **macOS**: first open may need right-click → **Open** (Gatekeeper)

### Features

| Page | What it does |
|------|----------------|
| Live recognition | Webcam ID with profiles `fast` / `balanced` / `accurate` |
| Camera enrollment | NEAR + FAR capture |
| Photo enrollment | Images + crop / zoom / rotate; merge into existing identity |
| Recognize image | Still-image matching |
| Gallery | List / delete identities |
| Settings | Profile, camera, provider, threshold |

## CLI quick start

```bash
cd eye-tracking
source ../.venv/bin/activate

python face-recognition/enroll.py --name Alice --provider CPUExecutionProvider
python face-recognition/enroll_image.py --name Alice --images photo.jpg
python face-recognition/recognize_live.py --profile balanced --provider CPUExecutionProvider
python face-recognition/recognize_image.py --image test.jpg --show
```

| Profile | Goal |
|---------|------|
| `fast` | Higher FPS |
| `balanced` | Default |
| `accurate` | Best quality |

Green box = known · Orange = Unknown

## Module map

| Module | Role | Entry |
|--------|------|--------|
| `app/` | Desktop GUI + unified pipeline | `python app/launch.py` |
| `face-recognition/` | Enroll & recognize faces | `enroll.py` / `recognize_live.py` |
| `webcam-capture/` | Webcam preview | `input-camera.py` |
| `camera-calibration/` | Intrinsics (chessboard) | `take_photo.py` + `calibration.py` |
| `pupil-segmentation/` | Eye / pupil localization | `run_demo.py` |
| `gaze-estimation/` | Gaze calibration & pointer | `run_calibration.py` |
| `performance-metrics/` | Session metrics / logs | pipeline `--enable-log` |

Specs / slides: `eye-tracking/docs/`

## FAIR practices

| Principle | How |
|-----------|-----|
| Findable | `gallery/` artifacts + named feature modules |
| Accessible | CLI `--help`, CPU default, desktop GUI |
| Interoperable | Cosine similarity on InsightFace embeddings |
| Reusable | Shared `common/` package |

## Tips

- Enroll in good lighting, face the camera
- Use several photos (near + far) for a stable embedding
- False Unknown → lower threshold (e.g. `0.35`)
- Wrong person → raise threshold (e.g. `0.45`)
- Re-enroll after big appearance changes

## Tests

```bash
cd eye-tracking
python -m unittest discover -s tests -v
```

Unit suite (~50 tests): gallery match/merge/remove, tracking confirmation, quality gates,
profiles, metrics, calibration YAML, camera helpers, desktop platform utilities.
No webcam / InsightFace required. CI runs on **every push** and pull request
([`.github/workflows/tests.yml`](.github/workflows/tests.yml)).

## Documentation site

Sources: [`docs/`](docs/) (Quarto). CI workflow: [`.github/workflows/docs.yml`](.github/workflows/docs.yml).

```bash
# Preview locally
quarto preview docs

# Build static site → docs/_site/
quarto render docs
```

On GitHub: **Settings → Pages → Source = GitHub Actions**. After the first green `Publish Quarto docs` run, the site is at [https://mohamadysn.github.io/FaceGate/](https://mohamadysn.github.io/FaceGate/).

## Privacy

The gallery stores **embeddings only** (no face photos). Do not commit
`gallery.json` / `embeddings.npy` if they contain real identities.
