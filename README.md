Static CT Clinical Reconstruction Demo

A full-screen interactive clinical CT reconstruction demo system with a Windows front-end client and a GPU-accelerated Linux back-end server.

The system is designed for operation on a Windows mini-PC, such as an Intel NUC, connected to a display by HDMI. The user interface is optimized for stylus-only interaction, with no reliance on mouse hover, keyboard input, or small UI controls. Computationally intensive simulation and reconstruction tasks are executed on a separate Linux machine learning server equipped with an NVIDIA RTX 4090 GPU.

## Project Overview

This project provides an interactive demonstration platform for comparing static CT system designs and reconstruction algorithms across representative clinical imaging cases.

The user selects:

- A clinical case
- A static CT system configuration
- A reconstruction algorithm
- Algorithm-specific reconstruction parameters

The selected configuration is sent to a GPU back-end server, which runs the requested simulation or reconstruction task and returns images, volumes, metrics, and status updates to the front-end interface.

## Intended Use

This software is intended as a research and demonstration platform for static CT image reconstruction. It is not intended for clinical diagnosis or real-time clinical decision making.

## System Architecture

The recommended architecture is a browser-based front end running locally on the Windows NUC, with a Python/FastAPI back end running inside a dedicated Docker container on the Linux GPU server.

```
+-----------------------------------------------------+
| Windows Mini-PC / Intel NUC                         |
|                                                     |
|  Full-Screen Browser UI                             |
|  - Touch/stylus optimized                           |
|  - No hover interactions                            |
|  - Large buttons and page-based workflow            |
|  - Runs in kiosk/full-screen mode over HDMI         |
+--------------------------+--------------------------+
                           |
                           | REST / WebSocket API
                           |
+--------------------------v--------------------------+
| Linux GPU Server                                     |
|                                                     |
|  Docker Container                                    |
|  - Python backend                                    |
|  - FastAPI server                                    |
|  - PyTorch / CUDA                                    |
|  - Reconstruction algorithms                         |
|  - Simulation engine                                 |
|  - Data storage and result cache                     |
|                                                     |
|  Hardware: NVIDIA RTX 4090                           |
+-----------------------------------------------------+
```

### Why Browser-Based UI?

Several front-end options were considered:

- **Option 1: C# Windows GUI**: A C# WPF or WinUI application would integrate well with Windows and could run full-screen. However, it would require Windows-specific development and a separate UI codebase from the Python reconstruction ecosystem.
- **Option 2: Python Qt GUI**: A PyQt or PySide GUI would allow Python-based front-end development, but deployment on Windows can become more fragile, especially when running full-screen, handling touch events, and packaging dependencies.
- **Option 3: Browser-Based UI**: A browser UI is the recommended option. It can run full-screen on the Windows NUC using kiosk mode and communicate cleanly with the Linux GPU server. The UI can be designed using large touch-friendly controls, page-based navigation, progress updates, and image viewers.

**Recommended stack:**

- **Frontend:**
  - React, Vue, or plain HTML/JavaScript
  - Full-screen browser kiosk mode
  - Touch/stylus-first UI design
- **Backend:**
  - Python
  - FastAPI
  - WebSocket support for progress updates
  - PyTorch/CUDA for deep and generative reconstruction
  - Docker container on Linux GPU server

## User Interaction Constraints

The interface is designed for stylus-only use.

**Important design constraints:**

- No hover-dependent menus
- No right-click interactions
- No keyboard required during normal operation
- Large buttons
- Large tabs or page navigation controls
- Clear selected states
- Explicit click/tap actions
- Drag-and-release support for image interaction
- Avoid tiny sliders unless they have large handles
- Use stepper buttons or preset parameter values when possible
- Always provide visible feedback after a tap

**The UI should support:**

- Tap to select
- Tap to advance
- Tap to go back
- Drag to pan
- Drag slider handles
- Pinch/zoom if supported by the display/browser
- Reset view button
- Confirm/run button

## Clinical Cases

The system should initially support four clinical case categories:

1. Head CT
2. Thoracic CT
3. Abdominal CT
4. Pelvic CT

Each case should have associated input data, display thumbnails, and metadata.

**Example case metadata:**

```json
{
  "case_id": "head_ct_001",
  "case_type": "Head CT",
  "description": "Representative head CT simulation case",
  "default_window": {
    "level": 40,
    "width": 80
  },
  "available_ground_truth": true
}
```

## Static CT System Configurations

The user should be able to choose between two static CT system designs.

### Oil Ring System
- **Name:** Oil Ring
- **Views:** 80
- **Focal spot size:** 1.5 mm
- **Description:** Lower-view static CT system with larger focal spot.

### Vacuum Ring System
- **Name:** Vacuum Ring
- **Views:** 240
- **Focal spot size:** 1.0 mm
- **Description:** Higher-view static CT system with smaller focal spot.

**Example configuration:**

```json
{
  "system_id": "oil_ring_80_view",
  "display_name": "Oil Ring",
  "num_views": 80,
  "focal_spot_mm": 1.5,
  "description": "80-view static CT system with 1.5 mm focal spot"
}
```

```json
{
  "system_id": "vacuum_ring_240_view",
  "display_name": "Vacuum Ring",
  "num_views": 240,
  "focal_spot_mm": 1.0,
  "description": "240-view static CT system with 1.0 mm focal spot"
}
```

## Reconstruction Algorithms

The system should support four reconstruction modes.

### 1. FBP Recon
Filtered backprojection reconstruction.

**Example parameters:**
```json
{
  "algorithm_id": "fbp",
  "display_name": "FBP Recon",
  "parameters": {
    "cutoff_frequency": 0.8,
    "filter_type": "hann"
  }
}
```
**Suggested UI controls:**
- Filter type: Ram-Lak, Hann, Hamming
- Cutoff frequency: 0.2 to 1.0
- Slice thickness or axial averaging, optional

### 2. Iterative Recon
Model-based or regularized iterative reconstruction.

**Example parameters:**
```json
{
  "algorithm_id": "iterative",
  "display_name": "Iterative Recon",
  "parameters": {
    "regularization_weight": 0.01,
    "num_iterations": 50
  }
}
```
**Suggested UI controls:**
- Regularization weight
- Number of iterations
- Regularizer type, optional
- Edge-preserving strength, optional

### 3. Deep Recon
Deep-learning-based reconstruction or enhancement.

**Example parameters:**
```json
{
  "algorithm_id": "deep_recon",
  "display_name": "Deep Recon",
  "parameters": {
    "deep_prior_weight": 0.5,
    "model_checkpoint": "default_deep_recon.pt"
  }
}
```
**Suggested UI controls:**
- Deep prior weight
- Model/checkpoint selection
- Denoising strength
- Data consistency strength

### 4. Generative Recon
Diffusion, score-based, or posterior-sampling reconstruction.

**Example parameters:**
```json
{
  "algorithm_id": "generative_recon",
  "display_name": "Generative Recon",
  "parameters": {
    "prior_weight": 0.5,
    "data_consistency_weight": 1.0,
    "num_sampling_steps": 100
  }
}
```
**Suggested UI controls:**
- Generative prior weight
- Data consistency weight
- Number of sampling steps
- Random seed
- Early stopping level
- Uncertainty samples, optional

## Recommended UI Workflow

The interface should be organized as a page-based workflow rather than a dense desktop-style GUI.

**Recommended pages:**

1. Start / Home
2. Clinical Case Selection
3. Static CT System Selection
4. Reconstruction Algorithm Selection
5. Parameter Selection
6. Simulation / Reconstruction Run
7. Results Viewer
8. Comparison / Export

## Proposed Page Layout

### Page 1: Home
**Purpose:** Introduce the demo and start a new reconstruction.
**Main controls:**
- Start New Simulation
- Load Previous Result
- Settings
- Exit Full Screen / Admin Mode, optional

### Page 2: Case Selection
**Purpose:** Select the anatomical case.
**Large selectable cards:**
- Head CT
- Thoracic CT
- Abdominal CT
- Pelvic CT

Each card should include:
- Case name
- Representative thumbnail
- Short description
- Selected state

### Page 3: Static CT System Selection
**Purpose:** Choose the scanner configuration.
**Selectable cards:**
- Oil Ring (80 views, 1.5 mm focal spot)
- Vacuum Ring (240 views, 1.0 mm focal spot)

**Optional comparison panel:**
| Property | Oil Ring | Vacuum Ring |
| :--- | :--- | :--- |
| Views | 80 | 240 |
| Focal spot | 1.5 mm | 1.0 mm |
| Expected scan sampling | Sparse | Higher |
| Expected image quality | Lower | Higher |

### Page 4: Reconstruction Algorithm
**Purpose:** Select reconstruction method.
**Large selectable cards:**
- FBP Recon
- Iterative Recon
- Deep Recon
- Generative Recon

Each card should include a short explanation and expected runtime category.

### Page 5: Parameter Selection
**Purpose:** Adjust algorithm-specific settings.
The controls on this page should change depending on the selected algorithm.

**Recommended control style:**
- Large sliders
- Plus/minus stepper buttons
- Preset buttons: Low, Medium, High
- Reset to Default button
- Advanced Settings toggle

### Page 6: Simulation / Reconstruction
**Purpose:** Run the selected job on the backend server.
**Display:**
- Selected case, scanner, algorithm, and parameters
- Run / Cancel buttons
- Progress bar
- Current status message

### Page 7: Results Viewer
**Purpose:** View reconstructed images.
**Required controls:**
- Slice scroll
- Window/level presets
- Zoom / Pan / Reset view
- Toggle ground truth/reference (if available)
- Toggle difference image (if available)
- Display reconstruction metrics

### Page 8: Comparison / Export
**Purpose:** Compare multiple reconstructions and save results.

---

## Backend API Design

The backend should expose a small API for configuration, job submission, job status, and result retrieval.

### API Endpoints
- `GET /api/health`
- `GET /api/cases`
- `GET /api/systems`
- `GET /api/algorithms`
- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/status`
- `GET /api/jobs/{job_id}/results`
- `GET /api/jobs/{job_id}/image/{slice_index}`
- `WebSocket /api/jobs/{job_id}/stream`

---

## Suggested Repository Structure

```
static-ct-demo/
├── README.md
├── docker/
│   ├── Dockerfile.backend
│   └── docker-compose.yml
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── cases.py
│   │   │   ├── systems.py
│   │   │   ├── algorithms.py
│   │   │   └── jobs.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── job_manager.py
│   │   │   └── schemas.py
│   │   ├── recon/
│   │   │   ├── fbp.py
│   │   │   ├── iterative.py
│   │   │   ├── deep_recon.py
│   │   │   └── generative_recon.py
│   │   ├── simulation/
│   │   │   ├── projection.py
│   │   │   ├── geometry.py
│   │   │   └── focal_spot.py
│   │   └── data/
│   │       ├── cases.json
│   │       ├── systems.json
│   │       └── algorithms.json
│   ├── requirements.txt
│   └── run_backend.sh
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx
│       ├── api/
│       │   └── client.js
│       ├── pages/
│       │   ├── HomePage.jsx
│       │   ├── CaseSelectionPage.jsx
│       │   ├── SystemSelectionPage.jsx
│       │   ├── AlgorithmSelectionPage.jsx
│       │   ├── ParameterSelectionPage.jsx
│       │   ├── RunPage.jsx
│       │   └── ResultsPage.jsx
│       ├── components/
│       │   ├── LargeButton.jsx
│       │   ├── SelectionCard.jsx
│       │   ├── ParameterSlider.jsx
│       │   ├── ProgressPanel.jsx
│       │   └── ImageViewer.jsx
│       └── styles/
│           └── touchscreen.css
├── configs/
│   ├── cases.json
│   ├── systems.json
│   └── algorithms.json
├── data/
│   ├── input_cases/
│   └── sample_results/
├── results/
├── scripts/
│   ├── launch_kiosk_windows.ps1
│   ├── start_backend.sh
│   └── test_api.py
└── docs/
    ├── architecture.md
    ├── ui_workflow.md
    └── backend_api.md
```

## Backend Implementation Plan

### Phase 1: API Skeleton
Implement the FastAPI backend with placeholder reconstruction functions.
- Create API endpoints
- Create job manager
- Return fake progress updates and sample images

### Phase 2: Frontend Prototype
Implement the full-screen workflow with placeholder data.
- Build page-based UI with large selection cards
- Add algorithm-specific parameter controls
- Add run page with progress bar and results viewer
- Add kiosk-mode styling

### Phase 3: Backend Reconstruction Integration
Connect real reconstruction code.
- Add projection simulation and geometry configs
- Add FBP, iterative, deep, and generative recon
- Save outputs to structured result folders

### Phase 4: Result Visualization
Add interactive image and metric display.
- Slice viewer with window/level presets
- Side-by-side comparison and difference images
- Metrics and runtime display

### Phase 5: Deployment
Prepare the system for demonstration use.
- Dockerize backend with GPU support
- Add Windows kiosk launch script and error handling

---

## Running the Backend
```bash
cd docker
docker compose up --build
```
The backend should be available at `http://<server-ip>:8000`.

## Running the Frontend
```bash
cd frontend
npm install
npm run dev
```
The frontend should be available at `http://localhost:5173`.

---

## Touchscreen and Stylus UI Rules
- Minimum button height: 64 px (Preferred: 80-120 px)
- Minimum spacing between controls: 16 px
- Avoid hover-only controls and nested menus
- Use large buttons, full-card selection, and explicit Back/Continue buttons
- Use visible selected states and providing feedback after a tap

---

## Development Milestones
1. **Clickable UI Mockup:** Demonstrate full user workflow without real reconstruction.
2. **Backend Job System:** Submit jobs from frontend to backend and track progress.
3. **FBP and Iterative Reconstruction:** Connect first real reconstruction algorithms.
4. **Deep Recon and Generative Recon:** Add learned reconstruction algorithms.
5. **Demo-Ready System:** Prepare for reliable hands-on demonstration.

## Windows Kiosk Mode

The Windows NUC can launch the app in full-screen browser mode for a dedicated "appliance" feel.

**Example PowerShell script (`launch_kiosk.ps1`):**

```powershell
$AppUrl = "http://localhost:5173" # Or the remote URL if served from axis03

# Launch Microsoft Edge in Kiosk Mode
Start-Process "msedge.exe" -ArgumentList "--kiosk $AppUrl --edge-kiosk-type=fullscreen --no-first-run"

# Alternative: Google Chrome
# Start-Process "chrome.exe" -ArgumentList "--kiosk $AppUrl --no-first-run"
```

For a permanent installation, this script can be added to the Windows Startup folder (`shell:startup`).

## Networking and Port Forwarding

The demo system involves communication between the Windows NUC (client) and the `axis03` Linux GPU server (backend).

### 1. Container to Host (`axis03`)
The backend FastAPI server runs inside a Docker container. We map the container's internal port to the host:
- **FastAPI Backend:** Container port `8000` -> `axis03:8000`
- **Vite Frontend (if containerized):** Container port `5173` -> `axis03:5173`

In `docker-compose.yml`:
```yaml
services:
  backend:
    ports:
      - "8000:8000"
  frontend:
    ports:
      - "5173:5173"
```

### 2. Host (`axis03`) to NUC Client
There are two primary ways to connect the NUC to the services on `axis03`:

**Option A: Direct Local Network Access (Recommended)**
If the NUC and `axis03` are on the same local network, the NUC can access the services directly using the server's IP address:
- **Frontend:** `http://<axis03-ip>:5173`
- **Backend API:** `http://<axis03-ip>:8000`

**Option B: SSH Tunneling (If network access is restricted)**
If direct port access is blocked, use an SSH tunnel from the NUC to `axis03`. This makes the remote services appear as `localhost` on the NUC:

On the Windows NUC (using PowerShell or Command Prompt):
```bash
ssh -L 5173:localhost:5173 -L 8000:localhost:8000 user@axis03
```
Now the NUC client can access the app at `http://localhost:5173`.

## License
To be determined.

## Status
Early planning and prototype development.

