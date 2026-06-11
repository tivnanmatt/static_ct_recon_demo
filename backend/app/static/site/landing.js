class SequencePlayer {
    constructor(canvasId, datasetId) {
        this.canvasId = canvasId;
        this.datasetId = datasetId;
        
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            const parentId = `preview-${datasetId}`;
            const parent = document.getElementById(parentId);
            if (parent) {
                // Keep the GIF background, add canvas on top
                this.canvas = document.createElement('canvas');
                this.canvas.id = canvasId;
                this.canvas.style.width = '100%';
                this.canvas.style.height = '100%';
                this.canvas.style.display = 'block';
                this.canvas.style.position = 'absolute';
                this.canvas.style.top = '0';
                this.canvas.style.left = '0';
                parent.appendChild(this.canvas);
            }
        }
        
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.allSlices = []; 
        this.currentFlatIndex = 0;
        this.isLocked = false;
        this.lockedPatientId = null;
        this.lockedSliceIdx = 0;
        this.isPlaying = false;
        this.cache = new Map();
        this.manifest = null;
    }

    async init(manifest = null) {
        if (!this.ctx) return;
        this.manifest = manifest;
        
        if (this.manifest) {
            this.allSlices = [];
            const pIds = Object.keys(this.manifest).slice(0, 5);
            for (const pId of pIds) {
                const data = this.manifest[pId];
                const start = Math.floor(data.slice_count * 0.35);
                const count = Math.min(data.slice_count - start, 15);
                for (let i = 0; i < count; i++) {
                    this.allSlices.push({ pId: pId, sIdx: start + i });
                }
            }
        }
    }

    setLockedState(isLocked, patientId = null, sliceIndex = 0) {
        this.isLocked = isLocked;
        this.lockedPatientId = patientId;
        this.lockedSliceIdx = parseInt(sliceIndex);
        if (isLocked && patientId !== null) {
            this.showLockedFrame();
        }
    }

    async showLockedFrame() {
        if (!this.lockedPatientId) return;
        const img = await this.getFrame(this.lockedPatientId, this.lockedSliceIdx);
        if (img && this.isLocked && this.lockedPatientId === this.lockedPatientId) {
            this.draw(img);
        }
    }

    start() {
        if (this.isPlaying || !this.ctx) return;
        this.isPlaying = true;
        this.animate();
    }

    async animate() {
        if (!this.isPlaying) return;

        if (!this.isLocked && this.allSlices.length > 0) {
            const entry = this.allSlices[this.currentFlatIndex];
            const img = await this.getFrame(entry.pId, entry.sIdx);
            if (img && !this.isLocked) {
                this.draw(img);
            }
            this.currentFlatIndex = (this.currentFlatIndex + 1) % this.allSlices.length;
        }

        setTimeout(() => requestAnimationFrame(() => this.animate()), 40);
    }

    draw(img) {
        if (!this.ctx) return;
        if (this.canvas.width !== img.width || this.canvas.height !== img.height) {
            this.canvas.width = img.width;
            this.canvas.height = img.height;
        }
        this.ctx.drawImage(img, 0, 0);
    }

    async getFrame(pId, idx) {
        const key = `${pId}-${idx}-${windowWidth}-${windowLevel}`;
        if (this.cache.has(key)) return this.cache.get(key);
        
        return new Promise(resolve => {
            const img = new Image();
            img.onload = () => {
                if (this.cache.size > 500) this.cache.clear();
                this.cache.set(key, img);
                resolve(img);
            };
            img.onerror = () => resolve(null);
            img.src = `/api/preview/${this.datasetId}/${encodeURIComponent(pId)}/${idx}?ww=${windowWidth}&wl=${windowLevel}`;
        });
    }
}

let selectedDataset = null;
let currentManifest = null;
let currentPatientIds = [];
const players = {};
const datasetDefaults = {}; // Store WW/WL defaults per dataset
let workflowState = 'START'; // 'START', 'LOADED', 'SIMULATED'
let isFbpRunComplete = false;
const stageRunState = {
    'simulate-ct-data': { controller: null, jobId: null },
    'eigen-fbp-recon': { controller: null, jobId: null },
    'model-based-iterative-recon': { controller: null, jobId: null },
    'deep-learning-recon': { controller: null, jobId: null },
    'generative-ai-recon': { controller: null, jobId: null },
    'evaluation': { controller: null, jobId: null }
};

let currentX0List = null;
let x0CycleInterval = null;

let lastStreamData = null;
let currentDisplayMode = 'sample'; // 'sample', 'animation', or 'mean'
let animationCycleIdx = 0;

// Permanent visual cycling timer at 12 fps (83ms) for Display Mode B (much faster, cinematic cycling)
setInterval(() => {
    if (lastStreamData) {
        const M = (lastStreamData.x0_list && lastStreamData.x0_list.length) || 1;
        if (M > 1) {
            animationCycleIdx = (animationCycleIdx + 1) % M;
            // The generative-process image (xt) animates over batch samples in animation,
            // mean, and hallucination modes.
            if (currentDisplayMode === 'animation' || currentDisplayMode === 'mean' || currentDisplayMode === 'hallucination') {
                updateDisplay();
            }
        }
    }
}, 83);

const globalUpdateImageView = (container, src) => {
    if (!container) return;
    let img = container.querySelector('img');
    if (!img) {
        container.innerHTML = '';
        img = document.createElement('img');
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        img.style.backgroundColor = 'black';
        container.appendChild(img);
    }
    img.src = src;
};

function updateDisplay() {
    if (!lastStreamData) return;

    const boxGt = document.getElementById('gen-box-gt');
    const boxFbp = document.getElementById('gen-box-full-fbp');
    const boxXt = document.getElementById('gen-box-xt');
    const boxX0 = document.getElementById('gen-box-x0');

    if (lastStreamData.gt) globalUpdateImageView(boxGt, lastStreamData.gt);
    if (lastStreamData.full_fbp) globalUpdateImageView(boxFbp, lastStreamData.full_fbp);

    const M = (lastStreamData.x0_list && lastStreamData.x0_list.length) || 1;

    if (currentDisplayMode === 'hallucination') {
        // Hallucination map: log-variance across the posterior samples (needs N > 1).
        // The generative-process image (xt) cycles through the batch samples.
        const idx = animationCycleIdx % M;
        const xtImg = (lastStreamData.xt_list && lastStreamData.xt_list[idx]) || lastStreamData.xt;
        if (xtImg) globalUpdateImageView(boxXt, xtImg);
        if (M > 1 && lastStreamData.hallucination_map) {
            globalUpdateImageView(boxX0, lastStreamData.hallucination_map);
        } else if (boxX0) {
            boxX0.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#aaa;font-size:0.8rem;text-align:center;padding:8px;">Hallucination Map needs more than 1 sample (set Num Samples &gt; 1)</div>';
        }
    } else if (M <= 1 || currentDisplayMode === 'sample') {
        // Option A: Single Sample Mode (Sample 0)
        if (lastStreamData.xt) globalUpdateImageView(boxXt, lastStreamData.xt);
        if (lastStreamData.x0) globalUpdateImageView(boxX0, lastStreamData.x0);
    } else if (currentDisplayMode === 'animation') {
        // Option B: Multi-Sample Animation Mode (cycles through the batch samples)
        const idx = animationCycleIdx % M;
        const xtImg = (lastStreamData.xt_list && lastStreamData.xt_list[idx]) || lastStreamData.xt;
        const x0Img = (lastStreamData.x0_list && lastStreamData.x0_list[idx]) || lastStreamData.x0;

        if (xtImg) globalUpdateImageView(boxXt, xtImg);
        if (x0Img) globalUpdateImageView(boxX0, x0Img);
    } else if (currentDisplayMode === 'mean') {
        // Option C: Generative Mean Mode — the generative-process image (xt) cycles
        // through the batch samples; the reconstruction shows the posterior mean.
        const idx = animationCycleIdx % M;
        const xtImg = (lastStreamData.xt_list && lastStreamData.xt_list[idx]) || lastStreamData.xt;
        if (xtImg) globalUpdateImageView(boxXt, xtImg);

        const meanX0Img = lastStreamData.mean_x0 || lastStreamData.x0;
        if (meanX0Img) globalUpdateImageView(boxX0, meanX0Img);
    }
}

// Persistent view settings
let windowWidth = 350;
let windowLevel = 50;

function isAbortError(error) {
    return error && error.name === 'AbortError';
}

function sleep(ms, signal) {
    return new Promise((resolve, reject) => {
        const onAbort = () => {
            clearTimeout(timer);
            reject(new DOMException('Aborted', 'AbortError'));
        };

        const timer = setTimeout(() => {
            if (signal) signal.removeEventListener('abort', onAbort);
            resolve();
        }, ms);

        if (signal) {
            if (signal.aborted) {
                onAbort();
                return;
            }
            signal.addEventListener('abort', onAbort, { once: true });
        }
    });
}

function createStageRun(stage) {
    const controller = new AbortController();
    stageRunState[stage].controller = controller;
    stageRunState[stage].jobId = null;

    document.querySelectorAll(`[data-stop-button="${stage}"]`).forEach((stopBtn) => {
        stopBtn.disabled = false;
    });
    return controller;
}

function finishStageRun(stage, controller) {
    if (stageRunState[stage]?.controller !== controller) return;
    stageRunState[stage].controller = null;
    stageRunState[stage].jobId = null;

    document.querySelectorAll(`[data-stop-button="${stage}"]`).forEach((stopBtn) => {
        stopBtn.disabled = true;
    });
}

function stopStageRun(stage) {
    const state = stageRunState[stage];
    if (!state || !state.controller) return;

    if (state.jobId) {
        // Use iterative stop for generative as well since they share the registry
        fetch(`/api/reconstruct/iterative/stop/${state.jobId}`, { method: 'POST' }).catch((error) => {
            console.warn(`Failed to request stop for ${stage}`, error);
        });
    }

    state.controller.abort();
}

function abortActiveRuns() {
    Object.keys(stageRunState).forEach((stage) => stopStageRun(stage));
}

function resetStageProgress(stage, status = 'Idle') {
    const fill = document.querySelector(`[data-progress-fill="${stage}"]`);
    const text = document.querySelector(`[data-progress-status="${stage}"]`);
    const nextBtn = document.querySelector(`[data-next-stage="${stage}"]`);

    if (fill) fill.style.width = '0%';
    if (text) text.textContent = status;
    if (nextBtn) nextBtn.disabled = true;
}

function clearCanvas(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function resetSimulationStage(status = 'Idle') {
    clearCanvas('geometry-canvas');
    clearCanvas('sinogram-canvas');
    resetStageProgress('simulate-ct-data', status);
    if (workflowState !== 'START') {
        workflowState = 'LOADED';
    }
    updateWorkflowUI();
}

function resetEigenStage(status = 'Idle') {
    const boxUnfiltered = document.getElementById('recon-box-unfiltered');
    const boxFiltered = document.getElementById('recon-box-filtered');
    const timingUnfiltered = document.getElementById('timing-unfiltered');
    const timingFiltered = document.getElementById('timing-filtered');

    if (boxUnfiltered) boxUnfiltered.innerHTML = '';
    if (boxFiltered) boxFiltered.innerHTML = '';
    if (timingUnfiltered) timingUnfiltered.textContent = '-- ms';
    if (timingFiltered) timingFiltered.textContent = '-- ms';
    resetStageProgress('eigen-fbp-recon', status);
    updateWorkflowUI();
}

function resetIterativeStage(status = 'Idle') {
    const boxLoss = document.getElementById('iter-box-loss');
    const boxLive = document.getElementById('iter-box-live');

    // Keep iter-box-init (FlashFBP initialization) visible across resets/stops.
    if (boxLoss) boxLoss.innerHTML = '';
    if (boxLive) boxLive.innerHTML = '';
    resetStageProgress('model-based-iterative-recon', status);
    updateWorkflowUI();
}

function resetNeuralStage(status = 'Idle') {
    const boxInit = document.getElementById('dlr-box-init');
    const boxFinal = document.getElementById('dlr-box-final');
    const timingInit = document.getElementById('timing-dlr-init');
    const timingFinal = document.getElementById('timing-dlr-final');

    if (boxInit) boxInit.innerHTML = '';
    if (boxFinal) boxFinal.innerHTML = '';
    if (timingInit) timingInit.textContent = '-- ms';
    if (timingFinal) timingFinal.textContent = '-- ms';
    resetStageProgress('deep-learning-recon', status);
    updateWorkflowUI();
}

function buildIterativeJobId() {
    return `iter-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function buildGenerativeJobId() {
    return `gen-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function updateWorkflowUI() {
    const sidebarSim = document.querySelector('[data-stage-button="simulate-ct-data"]');
    const sidebarRecons = document.querySelectorAll('[data-stage-button$="-recon"], [data-stage-button="evaluation"], [data-stage-button="motion-scope"]');
    
    const runSim = document.querySelector('[data-run-button="simulate-ct-data"]');
    const runRecons = document.querySelectorAll('[data-run-button$="-recon"]');
    
    const nextToSim = document.getElementById('next-to-simulate');
    const nextRecons = document.querySelectorAll('[data-next-stage]');

    if (workflowState === 'START') {
        if (sidebarSim) sidebarSim.disabled = true;
        sidebarRecons.forEach(b => b.disabled = true);
        if (runSim) runSim.disabled = true;
        runRecons.forEach(b => b.disabled = true);
        if (nextToSim) nextToSim.disabled = true;
        nextRecons.forEach(b => b.disabled = true);
    } else if (workflowState === 'LOADED') {
        if (sidebarSim) sidebarSim.disabled = false;
        sidebarRecons.forEach(b => b.disabled = true);
        if (runSim) runSim.disabled = false;
        runRecons.forEach(b => b.disabled = true);
        // nextToSim is handled by the load-patient-button completion
    } else if (workflowState === 'SIMULATED') {
        if (sidebarSim) sidebarSim.disabled = false;
        sidebarRecons.forEach(b => b.disabled = false);
        if (runSim) runSim.disabled = false;
        runRecons.forEach(b => b.disabled = false);
        if (nextToSim) nextToSim.disabled = false;
        // nextRecons are enabled individually as stages complete
    }
}

function resetWorkflow() {
    abortActiveRuns();
    workflowState = 'START';
    isFbpRunComplete = false;

    clearCanvas('geometry-canvas');
    clearCanvas('sinogram-canvas');

    const reconBoxGt = document.getElementById('recon-box-gt');
    const reconBoxSino = document.getElementById('recon-box-sino');
    const reconBoxUnfiltered = document.getElementById('recon-box-unfiltered');
    const reconBoxFiltered = document.getElementById('recon-box-filtered');
    const iterBoxGt = document.getElementById('iter-box-gt');
    const iterBoxSino = document.getElementById('iter-box-sino');
    const iterBoxInit = document.getElementById('iter-box-init');
    const iterBoxLoss = document.getElementById('iter-box-loss');
    const iterBoxLive = document.getElementById('iter-box-live');
    const timingSino = document.getElementById('timing-sino');
    const timingUnfiltered = document.getElementById('timing-unfiltered');
    const timingFiltered = document.getElementById('timing-filtered');
    const gifLinkContainer = document.getElementById('sim-gif-link-container');
    const gifLink = document.getElementById('sim-gif-link');

    [
        reconBoxGt,
        reconBoxSino,
        reconBoxUnfiltered,
        reconBoxFiltered,
        iterBoxGt,
        iterBoxSino,
        iterBoxInit,
        iterBoxLoss,
        iterBoxLive,
    ].forEach((element) => {
        if (element) element.innerHTML = '';
    });

    if (timingSino) timingSino.textContent = '--';
    if (timingUnfiltered) timingUnfiltered.textContent = '-- ms';
    if (timingFiltered) timingFiltered.textContent = '-- ms';
    if (gifLinkContainer) gifLinkContainer.style.display = 'none';
    if (gifLink) gifLink.href = '#';

    // Clear cycle interval
    if (x0CycleInterval) {
        clearInterval(x0CycleInterval);
        x0CycleInterval = null;
    }
    currentX0List = null;

    // Reset Langevin walk controls
    const walkCheckbox = document.getElementById('langevin-walk-checkbox');
    const walkLabel = document.getElementById('langevin-walk-label');
    if (walkCheckbox) {
        walkCheckbox.checked = false;
        walkCheckbox.disabled = true;
        walkCheckbox.style.cursor = 'not-allowed';
        walkCheckbox.style.opacity = '0.5';
    }
    if (walkLabel) {
        walkLabel.style.cursor = 'not-allowed';
        walkLabel.style.opacity = '0.5';
        walkLabel.style.color = '#888';
    }
    
    // Reset all progress fills and statuses
    document.querySelectorAll('[data-progress-fill]').forEach(fill => fill.style.width = '0%');
    document.querySelectorAll('[data-progress-status]').forEach(status => status.textContent = 'Idle');
    document.querySelectorAll('[data-stop-button]').forEach(btn => btn.disabled = true);
    
    updateWorkflowUI();
}

async function initDatasetPreviews() {
    const resp = await fetch('/api/datasets');
    const datasets = await resp.json();
    for (const ds of datasets) {
        players[ds.id] = new SequencePlayer(`canvas-${ds.id}`, ds.id);
        datasetDefaults[ds.id] = { ww: ds.default_ww, wl: ds.default_wl };
    }
}

async function selectDataset(datasetId) {
    resetWorkflow();
    
    if (selectedDataset === datasetId) {
        // Reset player for the dataset being unselected
        const oldPlayer = players[datasetId];
        if (oldPlayer) {
            oldPlayer.setLockedState(false);
            if (oldPlayer.canvas) oldPlayer.canvas.style.display = 'none';
        }
        
        selectedDataset = null;
        currentManifest = null;
        document.getElementById('patient-selection-controls').style.display = 'none';
        document.querySelectorAll('.dataset-card').forEach(card => {
            card.classList.remove('selected', 'dimmed');
        });
        return;
    }

    selectedDataset = datasetId;
    
    // Set default Window/Level for the dataset if available
    if (datasetDefaults[datasetId]) {
        windowWidth = datasetDefaults[datasetId].ww;
        windowLevel = datasetDefaults[datasetId].wl;
        
        const wwSlider = document.getElementById('window-width-slider');
        const wlSlider = document.getElementById('window-level-slider');
        
        if (wwSlider) {
            wwSlider.value = windowWidth;
            document.getElementById('ww-display').textContent = windowWidth;
        }
        if (wlSlider) {
            wlSlider.value = windowLevel;
            document.getElementById('wl-display').textContent = windowLevel;
        }
    }

    document.querySelectorAll('.dataset-card').forEach(card => {
        const isSelected = card.dataset.dataset === datasetId;
        card.classList.toggle('selected', isSelected);
        card.classList.toggle('dimmed', !isSelected);
        
        // Hide other canvases, show only active one
        const player = players[card.dataset.dataset];
        if (player && player.canvas) {
            player.canvas.style.display = isSelected ? 'block' : 'none';
        }
    });

    try {
        const resp = await fetch(`/static/precomputed/${datasetId}_manifest.json`);
        currentManifest = await resp.json();
        currentPatientIds = Object.keys(currentManifest);
    } catch (e) {
        console.error("Failed to load manifest", e);
        return;
    }

    const player = players[datasetId];
    if (player) {
        await player.init(currentManifest);
        player.setLockedState(true);
    }

    document.getElementById('patient-selection-controls').style.display = 'flex';
    
    const pSlider = document.getElementById('patient-slider');
    pSlider.max = Math.max(0, currentPatientIds.length - 1);
    pSlider.value = 0;
    updatePatientDisplay(0);
}

async function updatePatientDisplay(index) {
    if (!selectedDataset || !currentManifest) return;
    
    // Changing the patient also resets simulation/reconstruction state
    resetWorkflow();
    
    const pId = currentPatientIds[index];
    const patientData = currentManifest[pId];
    const sSlider = document.getElementById('slice-slider');
    const sIdx = sSlider ? sSlider.value : 0;
    
    document.getElementById('patient-id-display').textContent = pId || "N/A";
    
    // Fetch real DICOM metadata from server
    try {
        const infoResp = await fetch(`/api/dicom-info/${selectedDataset}/${pId}/${sIdx}`);
        const info = await infoResp.json();
        
        const metaDump = Object.entries(info)
            .map(([k, v]) => `<span style="color:#001b5e;font-weight:700">${k}:</span> ${v}`)
            .join('; ');
        
        const dumpEl = document.getElementById('dicom-metadata-text');
        if (dumpEl) {
            dumpEl.innerHTML = metaDump || "No DICOM metadata available.";
        }
    } catch (e) {
        console.error("Failed to fetch DICOM info", e);
    }
    
    if (patientData) {
        if (sSlider) {
            sSlider.max = Math.max(0, patientData.slice_count - 1);
            // Don't reset slice to middle if we're just updating info
            // sSlider.value = Math.floor(patientData.slice_count / 2);
        }
        
        updateLiveFrame();
    }
    checkLoadEligibility();
}

function updateLiveFrame() {
    if (!selectedDataset || !currentManifest) return;
    
    const pIdx = document.getElementById('patient-slider').value;
    const pId = currentPatientIds[pIdx];
    const sIdx = document.getElementById('slice-slider').value;
    const patientData = currentManifest[pId];

    if (patientData) {
        document.getElementById('slice-index-display').textContent = sIdx;
        
        if (patientData.positions && patientData.positions[sIdx] !== undefined) {
            document.getElementById('slice-pos-display').textContent = `${parseFloat(patientData.positions[sIdx]).toFixed(1)} mm`;
        }
        if (patientData.instances && patientData.instances[sIdx] !== undefined) {
            document.getElementById('slice-inst-display').textContent = `Inst: ${patientData.instances[sIdx]}`;
        }

        const player = players[selectedDataset];
        if (player) {
            player.setLockedState(true, pId, sIdx);
        }

        // Draw active patient frame onto simulation patient canvas if it exists
        drawSimulationStagePatientFrame(pId, sIdx);

        // Auto-load patient into evaluation view if evaluation stage is active
        const evalPanel = document.querySelector('[data-stage-panel="evaluation"]');
        if (evalPanel && evalPanel.classList.contains('active')) {
            prepareEvaluationComparison();
        }
    }
}

async function drawSimulationStagePatientFrame(pId, sIdx) {
    const canvas = document.getElementById('sim-patient-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const key = `${pId}-${sIdx}-${windowWidth}-${windowLevel}`;
    const player = players[selectedDataset];
    if (player) {
        const img = await player.getFrame(pId, sIdx);
        if (img) {
            if (canvas.width !== img.width || canvas.height !== img.height) {
                canvas.width = img.width;
                canvas.height = img.height;
            }
            ctx.drawImage(img, 0, 0);
        }
    }
}

function checkLoadEligibility() {
    const btn = document.getElementById('load-patient-button');
    btn.disabled = !(selectedDataset && currentPatientIds.length > 0);
}

function handleStageNavigation() {
    document.querySelectorAll('[data-next-stage]').forEach(btn => {
        btn.addEventListener('click', () => {
            const currentStage = btn.getAttribute('data-next-stage');
            const map = {
                'load-patient': 'simulate-ct-data',
                'simulate-ct-data': 'eigen-fbp-recon',
                'eigen-fbp-recon': 'model-based-iterative-recon',
                'model-based-iterative-recon': 'deep-learning-recon',
                'deep-learning-recon': 'generative-ai-recon'
            };
            const target = map[currentStage];
            if (target) switchStage(target);
        });
    });

    document.querySelectorAll('.stage-button').forEach(btn => {
        btn.addEventListener('click', () => {
            const stageId = btn.getAttribute('data-stage-button');
            switchStage(stageId);
            
            // If they manually click back to Load Patient, should we reset?
            // The user said "and click it again, it resets". 
            // This probably refers to selecting/loading a patient again.
        });
    });
}

function switchStage(stageId) {
    document.querySelectorAll('.stage-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.stage-button').forEach(b => b.classList.remove('active'));
    
    const panel = document.querySelector(`[data-stage-panel="${stageId}"]`);
    const btn = document.querySelector(`[data-stage-button="${stageId}"]`);
    
    if (panel) panel.classList.add('active');
    if (btn) btn.classList.add('active');

    if (stageId === 'eigen-fbp-recon') {
        const boxUnfiltered = document.getElementById('recon-box-unfiltered');
        const boxFiltered = document.getElementById('recon-box-filtered');
        const timingUnfiltered = document.getElementById('timing-unfiltered');
        const timingFiltered = document.getElementById('timing-filtered');
        
        if (boxUnfiltered) boxUnfiltered.innerHTML = '';
        if (boxFiltered) boxFiltered.innerHTML = '';
        if (timingUnfiltered) timingUnfiltered.textContent = '-- ms';
        if (timingFiltered) timingFiltered.textContent = '-- ms';

        prepareReconComparison();
        updateFilterPlot();
    }

    if (stageId === 'model-based-iterative-recon') {
        const boxLoss = document.getElementById('iter-box-loss');
        const boxLive = document.getElementById('iter-box-live');
        if (boxLoss) boxLoss.innerHTML = '';
        if (boxLive) boxLive.innerHTML = '';
        // Do not blank iter-box-init: the FlashFBP initialization stays visible while
        // prepareIterativeComparison() recomputes it (server-side, even if FlashFBP was never run).
        prepareIterativeComparison();
    }

    if (stageId === 'deep-learning-recon') {
        const boxInit = document.getElementById('dlr-box-init');
        const boxFinal = document.getElementById('dlr-box-final');
        const timingInit = document.getElementById('timing-dlr-init');
        const timingFinal = document.getElementById('timing-dlr-final');
        if (boxInit) boxInit.innerHTML = '';
        if (boxFinal) boxFinal.innerHTML = '';
        if (timingInit) timingInit.textContent = '-- ms';
        if (timingFinal) timingFinal.textContent = '-- ms';
        prepareNeuralComparison();
    }

    if (stageId === 'evaluation') {
        prepareEvaluationComparison();
    }

    if (stageId === 'motion-scope') {
        updateMotionScopeGifs();
    }
}

// MotionScope: load the precomputed animation GIFs that match the current radio selections.
function updateMotionScopeGifs() {
    const views = document.querySelector('input[name="motion-views"]:checked')?.value || '80';
    const clock = document.querySelector('input[name="motion-clock"]:checked')?.value || 'no';
    const method = document.querySelector('input[name="motion-method"]:checked')?.value || 'flashfbp';

    const base = '/motion-animations';
    const setGif = (boxId, src) => {
        const box = document.getElementById(boxId);
        if (!box) return;
        box.innerHTML = '';
        const img = document.createElement('img');
        // Cache-bust so switching radios always restarts the animation from frame 0.
        img.src = `${src}?t=${Date.now()}`;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        img.style.backgroundColor = 'black';
        box.appendChild(img);
    };

    setGif('motion-box-gt', `${base}/gt_clock_${clock}.gif`);
    [4, 20, 80].forEach(rps => {
        setGif(`motion-box-rps${rps}`,
            `${base}/recon_views_${views}_clock_${clock}_method_${method}_rps_${rps}.gif`);
    });

    const status = document.querySelector('[data-progress-status="motion-scope"]');
    if (status) status.textContent = `Playing ${views} views · ${method} · clock ${clock}`;
}

async function prepareIterativeComparison() {
    const pIdx = document.getElementById('patient-slider').value;
    const pId = currentPatientIds[pIdx];
    const sIdx = document.getElementById('slice-slider').value;
    const numSources = parseInt(document.getElementById('sim-geometry-select').value);

    const boxGT = document.getElementById('iter-box-gt');
    const boxSino = document.getElementById('iter-box-sino');
    const boxInit = document.getElementById('iter-box-init');

    const updateImageView = (container, src, isSino = false) => {
        if (!container) return;
        container.innerHTML = '';
        const img = document.createElement('img');
        img.src = src;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = isSino ? 'fill' : 'contain';
        img.style.backgroundColor = 'black';
        container.appendChild(img);
    };

    try {
        const [respGT, respSino, respInit] = await Promise.all([
            fetch(`/api/preview/${selectedDataset}/${pId}/${sIdx}?ww=${windowWidth}&wl=${windowLevel}`),
            fetch(`/api/reconstruct/full-sinogram/${selectedDataset}/${pId}/${sIdx}/${numSources}`),
            fetch(`/api/reconstruct/fbp/${selectedDataset}/${pId}/${sIdx}/${numSources}?step=filter&ww=${windowWidth}&wl=${windowLevel}`)
        ]);

        const resGT = await respGT.blob();
        const resSino = await respSino.json();
        const resInit = await respInit.json();
        
        if (boxGT) updateImageView(boxGT, URL.createObjectURL(resGT));
        if (resSino.sinogram_image) {
            updateImageView(boxSino, resSino.sinogram_image, true);
        }
        if (boxInit && resInit.reconstruction_image) {
            updateImageView(boxInit, resInit.reconstruction_image);
        }
    } catch (e) {
        console.error("Failed to prepare iterative comparison", e);
    }
}

async function prepareReconComparison() {
    const pIdx = document.getElementById('patient-slider').value;
    const pId = currentPatientIds[pIdx];
    const sIdx = document.getElementById('slice-slider').value;
    const numSources = parseInt(document.getElementById('sim-geometry-select').value);

    const boxGT = document.getElementById('recon-box-gt');
    const boxSino = document.getElementById('recon-box-sino');
    const boxUnfiltered = document.getElementById('recon-box-unfiltered');
    const boxFiltered = document.getElementById('recon-box-filtered');
    const timingSino = document.getElementById('timing-sino');
    const timingUnfiltered = document.getElementById('timing-unfiltered');
    const timingFiltered = document.getElementById('timing-filtered');

    const sliderLow = document.getElementById('fbp-low-slider');
    const sliderMid = document.getElementById('fbp-mid-slider');
    const sliderHigh = document.getElementById('fbp-high-slider');

    const wLow = sliderLow ? parseFloat(sliderLow.value) / 100.0 : 1.0;
    const wMid = sliderMid ? parseFloat(sliderMid.value) / 100.0 : 1.0;
    const wHigh = sliderHigh ? parseFloat(sliderHigh.value) / 100.0 : 1.0;

    const updateImageView = (container, src, isSino = false) => {
        if (!container) return;
        container.innerHTML = '';
        const img = document.createElement('img');
        img.src = src;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = isSino ? 'fill' : 'contain';
        img.style.backgroundColor = 'black';
        container.appendChild(img);
    };

    try {
        const [respGT, respSino, respUnfiltered, respFiltered] = await Promise.all([
            fetch(`/api/preview/${selectedDataset}/${pId}/${sIdx}?ww=${windowWidth}&wl=${windowLevel}`),
            fetch(`/api/reconstruct/full-sinogram/${selectedDataset}/${pId}/${sIdx}/${numSources}`),
            fetch(`/api/reconstruct/fbp/${selectedDataset}/${pId}/${sIdx}/${numSources}?step=unfiltered&ww=${windowWidth}&wl=${windowLevel}`),
            fetch(`/api/reconstruct/fbp/${selectedDataset}/${pId}/${sIdx}/${numSources}?step=filter&ww=${windowWidth}&wl=${windowLevel}&w_low=${wLow}&w_mid=${wMid}&w_high=${wHigh}`)
        ]);

        const resGT = await respGT.blob();
        const resSino = await respSino.json();
        const resUnfiltered = await respUnfiltered.json();
        const resFiltered = await respFiltered.json();
        
        if (boxGT) updateImageView(boxGT, URL.createObjectURL(resGT));
        if (resSino.sinogram_image) {
            updateImageView(boxSino, resSino.sinogram_image, true);
            if (timingSino) timingSino.textContent = `${numSources} Views`;
        }
        if (resUnfiltered.reconstruction_image) {
            updateImageView(boxUnfiltered, resUnfiltered.reconstruction_image);
            if (timingUnfiltered) timingUnfiltered.textContent = `${resUnfiltered.recon_time_ms.toFixed(1)} ms`;
        }
        if (resFiltered.reconstruction_image) {
            updateImageView(boxFiltered, resFiltered.reconstruction_image);
            if (timingFiltered) timingFiltered.textContent = `${resFiltered.recon_time_ms.toFixed(1)} ms`;
        }
    } catch (e) {
        console.error("Failed to prepare recon comparison", e);
    }
}

async function updateFilterPlot() {
    try {
        const sliderLow = document.getElementById('fbp-low-slider');
        const sliderMid = document.getElementById('fbp-mid-slider');
        const sliderHigh = document.getElementById('fbp-high-slider');

        const wLow = sliderLow ? parseFloat(sliderLow.value) / 100.0 : 1.0;
        const wMid = sliderMid ? parseFloat(sliderMid.value) / 100.0 : 1.0;
        const wHigh = sliderHigh ? parseFloat(sliderHigh.value) / 100.0 : 1.0;
        
        const resp = await fetch(`/api/filter/plot?w_low=${wLow}&w_mid=${wMid}&w_high=${wHigh}`);
        const data = await resp.json();
        if (data.image) {
            const plotImg = document.getElementById('fbp-filter-plot-img');
            if (plotImg) plotImg.src = data.image;
        }
    } catch (e) {
        console.error("Failed to update FBP filter plot", e);
    }
}

async function updateFbpReconstruction() {
    try {
        const pIdx = document.getElementById('patient-slider').value;
        if (!currentPatientIds || currentPatientIds.length === 0) return;
        const pId = currentPatientIds[pIdx];
        if (!pId) return;
        const sIdx = document.getElementById('slice-slider').value;
        const numSources = parseInt(document.getElementById('sim-geometry-select').value);

        const sliderLow = document.getElementById('fbp-low-slider');
        const sliderMid = document.getElementById('fbp-mid-slider');
        const sliderHigh = document.getElementById('fbp-high-slider');

        const wLow = sliderLow ? parseFloat(sliderLow.value) / 100.0 : 1.0;
        const wMid = sliderMid ? parseFloat(sliderMid.value) / 100.0 : 1.0;
        const wHigh = sliderHigh ? parseFloat(sliderHigh.value) / 100.0 : 1.0;

        const boxFiltered = document.getElementById('recon-box-filtered');
        const timingFiltered = document.getElementById('timing-filtered');

        const updateImageView = (container, src) => {
            if (!container) return;
            container.innerHTML = '';
            const img = document.createElement('img');
            img.src = src;
            img.style.width = '100%';
            img.style.height = '100%';
            img.style.objectFit = 'contain';
            img.style.backgroundColor = 'black';
            container.appendChild(img);
        };

        const resp = await fetch(`/api/reconstruct/fbp/${selectedDataset}/${pId}/${sIdx}/${numSources}?step=filter&ww=${windowWidth}&wl=${windowLevel}&w_low=${wLow}&w_mid=${wMid}&w_high=${wHigh}`);
        const res = await resp.json();
        
        if (res.reconstruction_image) {
            updateImageView(boxFiltered, res.reconstruction_image);
            if (timingFiltered) {
                timingFiltered.textContent = `${res.recon_time_ms.toFixed(1)} ms`;
            }
        }
    } catch (e) {
        console.error("Failed to dynamically update FBP reconstruction:", e);
    }
}

async function prepareNeuralComparison() {
    const pIdx = document.getElementById('patient-slider').value;
    const pId = currentPatientIds[pIdx];
    const sIdx = document.getElementById('slice-slider').value;
    const numSources = parseInt(document.getElementById('sim-geometry-select').value);

    const boxGT = document.getElementById('dlr-box-gt');
    const boxSino = document.getElementById('dlr-box-sino');
    const timingSino = document.getElementById('timing-dlr-sino');

    const updateImageView = (container, src, isSino = false) => {
        if (!container) return;
        container.innerHTML = '';
        const img = document.createElement('img');
        img.src = src;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = isSino ? 'fill' : 'contain';
        img.style.backgroundColor = 'black';
        container.appendChild(img);
    };

    try {
        const [respGT, respSino] = await Promise.all([
            fetch(`/api/preview/${selectedDataset}/${pId}/${sIdx}?ww=${windowWidth}&wl=${windowLevel}`),
            fetch(`/api/reconstruct/full-sinogram/${selectedDataset}/${pId}/${sIdx}/${numSources}`)
        ]);

        const resGT = await respGT.blob();
        const resSino = await respSino.json();

        if (boxGT) updateImageView(boxGT, URL.createObjectURL(resGT));
        if (resSino.sinogram_image) {
            updateImageView(boxSino, resSino.sinogram_image, true);
            if (timingSino) timingSino.textContent = `${numSources} Views`;
        }
    } catch (e) {
        console.error('Failed to prepare NeuralSpark comparison', e);
    }
}

async function prepareEvaluationComparison() {
    const pIdx = document.getElementById('patient-slider').value;
    if (!currentPatientIds || currentPatientIds.length === 0) return;
    const pId = currentPatientIds[pIdx];
    const sIdx = document.getElementById('slice-slider').value;
    const numSources = parseInt(document.getElementById('sim-geometry-select').value);

    const boxGT = document.getElementById('eval-box-gt');
    const boxFBP = document.getElementById('eval-box-fbp');
    const boxMBIR = document.getElementById('eval-box-mbir');
    const boxDLR = document.getElementById('eval-box-dlr');
    const boxGEN = document.getElementById('eval-box-gen');

    const timingFBP = document.getElementById('timing-eval-fbp');
    const timingMBIR = document.getElementById('timing-eval-mbir');
    const timingDLR = document.getElementById('timing-eval-dlr');
    const timingGEN = document.getElementById('timing-eval-gen');

    const updateImageView = (container, src) => {
        if (!container) return;
        container.innerHTML = '';
        const img = document.createElement('img');
        img.src = src;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        img.style.backgroundColor = 'black';
        container.appendChild(img);
    };

    const showLoading = (container) => {
        if (!container) return;
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#aaa;font-size:0.8rem;">Benchmarking...</div>';
    };

    [boxGT, boxFBP, boxMBIR, boxDLR, boxGEN].forEach(showLoading);
    [timingFBP, timingMBIR, timingDLR, timingGEN].forEach(t => { if (t) t.textContent = '-- ms'; });

    try {
        const resp = await fetch(`/api/evaluation/reconstructions/${selectedDataset}/${pId}/${sIdx}/${numSources}?ww=${windowWidth}&wl=${windowLevel}`);
        const res = await resp.json();

        if (res.error) {
            console.error("Evaluation error:", res.error);
            return;
        }

        if (res.gt) updateImageView(boxGT, res.gt);
        if (res.fbp) updateImageView(boxFBP, res.fbp);
        if (res.mbir) updateImageView(boxMBIR, res.mbir);
        if (res.dlr) updateImageView(boxDLR, res.dlr);
        if (res.gen) updateImageView(boxGEN, res.gen);

        if (timingFBP && res.fbp_time) timingFBP.textContent = `${res.fbp_time.toFixed(1)} ms`;
        if (timingMBIR && res.mbir_time) timingMBIR.textContent = `${res.mbir_time.toFixed(1)} ms`;
        if (timingDLR && res.dlr_time) timingDLR.textContent = `${res.dlr_time.toFixed(1)} ms`;
        if (timingGEN && res.gen_time) timingGEN.textContent = `${res.gen_time.toFixed(1)} ms`;

    } catch (e) {
        console.error("Failed to prepare evaluation comparative reconstructions:", e);
    }
}

async function runEvaluationBenchmark(progressBar, statusText, runButton, controller) {
    const signal = controller.signal;
    const pIdx = document.getElementById('patient-slider').value;
    const pId = currentPatientIds[pIdx];
    const sIdx = document.getElementById('slice-slider').value;
    const numSources = parseInt(document.getElementById('sim-geometry-select').value);
    const exposureMas = parseFloat(document.getElementById('sim-exposure-select').value);

    const scope = document.querySelector('input[name="eval-scope"]:checked')?.value || 'single_patient';
    const numPatients = document.getElementById('eval-num-patients-slider').value;

    // Pull each reconstruction method's settings from its own stage controls so the
    // benchmark runs every algorithm exactly as the user configured it elsewhere.
    const numVal = (id, dflt) => { const el = document.getElementById(id); return el ? parseFloat(el.value) : dflt; };
    const radioVal = (name, dflt) => { const el = document.querySelector(`input[name="${name}"]:checked`); return el ? el.value : dflt; };

    // FlashFBP filter gains (sliders are 0-100 %)
    const wLow = numVal('fbp-low-slider', 100) / 100.0;
    const wMid = numVal('fbp-mid-slider', 100) / 100.0;
    const wHigh = numVal('fbp-high-slider', 100) / 100.0;

    // FidelityMBIR (TV and LR sliders are log10)
    const iters = numVal('iter-count-slider', 30);
    const tv = Math.pow(10, numVal('tv-strength-slider', 4.0));
    const lr = Math.pow(10, numVal('lr-slider', -1));
    const precond = document.getElementById('use-precond-check')?.checked ?? true;

    // GenerativeVision (sigma sliders are log10 HU std -> attenuation units, scaleOnly)
    const genSteps = numVal('diffusion-steps-slider', 10);
    const genLangevin = numVal('langevin-steps-slider', 0);
    const genSigmaMax = HU_to_atten(Math.pow(10, numVal('sigma-max-slider', 3)), true);
    const genSigmaMin = HU_to_atten(Math.pow(10, numVal('sigma-min-slider', 0)), true);
    const genSolver = radioVal('diffusion-solver', 'heun');
    const genTemperature = numVal('diffusion-temperature-slider', 0);
    const genNumSamples = numVal('num-samples-slider', 1);
    const genModelVariant = radioVal('diffusion-model-variant', 'base');

    const boxGT = document.getElementById('eval-box-gt');
    const boxFBP = document.getElementById('eval-box-fbp');
    const boxMBIR = document.getElementById('eval-box-mbir');
    const boxDLR = document.getElementById('eval-box-dlr');
    const boxGEN = document.getElementById('eval-box-gen');

    const updateImageView = (container, src) => {
        if (!container) return;
        container.innerHTML = '';
        const img = document.createElement('img');
        img.src = src;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        img.style.backgroundColor = 'black';
        container.appendChild(img);
    };

    try {
        statusText.textContent = 'Connecting to evaluation stream...';
        progressBar.style.width = '5%';

        const streamUrl = `/api/evaluation/run?scope=${scope}&num_patients=${numPatients}&dataset_id=${selectedDataset}&patient_id=${pId}&slice_index=${sIdx}&n_source=${numSources}&exposure_mas=${exposureMas}&ww=${windowWidth}&wl=${windowLevel}`
            + `&w_low=${wLow}&w_mid=${wMid}&w_high=${wHigh}`
            + `&iters=${iters}&tv=${tv}&lr=${lr}&precond=${precond}`
            + `&steps=${genSteps}&langevin_steps=${genLangevin}&sigma_max=${genSigmaMax}&sigma_min=${genSigmaMin}`
            + `&solver=${genSolver}&temperature=${genTemperature}&num_samples=${genNumSamples}&model_variant=${genModelVariant}`;
        const response = await fetch(streamUrl, { signal });
        if (!response.ok || !response.body) {
            throw new Error(`Evaluation stream failed with HTTP ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let chunkRemainder = '';

        while (true) {
            let result;
            try {
                result = await reader.read();
            } catch (e) {
                console.warn("Evaluation stream read interrupted", e);
                break;
            }
            const { value, done } = result;
            if (done) break;
            
            const chunk = chunkRemainder + decoder.decode(value);
            const lines = chunk.split('\n');
            chunkRemainder = lines.pop();
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        if (data.error) {
                            throw new Error(data.error);
                        }

                        // 1. Update progress & status
                        const progress = (data.index / data.total) * 100;
                        progressBar.style.width = `${progress}%`;
                        statusText.textContent = `Benchmarking: Patient ${data.patient_id} (${data.index}/${data.total}) completed`;

                        // 2. Update the per-method 2x2 metric violin panels (cumulative across samples)
                        if (data.plots) {
                            const setPlot = (id, b64) => {
                                if (!b64) return;
                                const im = document.getElementById(id);
                                if (im) im.src = `data:image/png;base64,${b64}`;
                            };
                            setPlot('eval-plot-fbp', data.plots.fbp);
                            setPlot('eval-plot-mbir', data.plots.mbir);
                            setPlot('eval-plot-dlr', data.plots.dlr);
                            setPlot('eval-plot-gen', data.plots.gen);
                        }

                        // 3. Update reconstructions live for every sampled patient
                        if (data.reconstructions) {
                            const r = data.reconstructions;
                            if (r.gt && boxGT) updateImageView(boxGT, `data:image/png;base64,${r.gt}`);
                            if (r.fbp && boxFBP) updateImageView(boxFBP, `data:image/png;base64,${r.fbp}`);
                            if (r.mbir && boxMBIR) updateImageView(boxMBIR, `data:image/png;base64,${r.mbir}`);
                            if (r.dlr && boxDLR) updateImageView(boxDLR, `data:image/png;base64,${r.dlr}`);
                            if (r.gen && boxGEN) updateImageView(boxGEN, `data:image/png;base64,${r.gen}`);
                        }
                    } catch (e) {
                        console.error("JSON Parse error in evaluation stream", e);
                    }
                }
            }
        }

        if (!signal.aborted) {
            statusText.textContent = 'Benchmark evaluation complete';
            progressBar.style.width = '100%';
        }
    } catch (e) {
        if (isAbortError(e)) {
            statusText.textContent = 'Benchmark Stopped';
            return;
        }
        console.error(e);
        statusText.textContent = `Benchmark evaluation failed: ${e.message || 'Network Error'}`;
    } finally {
        finishStageRun('evaluation', controller);
        if (runButton) runButton.disabled = false;
    }
}

function handleSimulationRuns() {
    document.querySelectorAll('[data-run-button]').forEach(btn => {
        btn.addEventListener('click', () => {
            const stage = btn.getAttribute('data-run-button');
            console.log(`DEBUG [landing.js]: Run button clicked for stage: ${stage}`);
            
            const fill = document.querySelector(`[data-progress-fill="${stage}"]`);
            const status = document.querySelector(`[data-progress-status="${stage}"]`);
            const nextBtn = document.querySelector(`[data-next-stage="${stage}"]`);
            const controller = createStageRun(stage);
            
            btn.disabled = true;

            if (stage === 'simulate-ct-data') {
                runSimulationAnimation(fill, status, nextBtn, controller);
                return;
            }

            if (stage === 'eigen-fbp-recon') {
                runEigenFBPRecon(fill, status, nextBtn, controller);
                return;
            }

            if (stage === 'model-based-iterative-recon') {
                console.log(`DEBUG [landing.js]: calling runIterativeRecon`);
                runIterativeRecon(fill, status, nextBtn, controller);
                return;
            }

            if (stage === 'deep-learning-recon') {
                runNeuralSpeedRecon(fill, status, nextBtn, controller);
                return;
            }

            if (stage === 'generative-ai-recon') {
                runGenerativeVisionRecon(fill, status, nextBtn, controller);
                return;
            }

            let progress = 0;
            status.textContent = "Processing...";
            const interval = setInterval(() => {
                progress += 2;
                fill.style.width = `${Math.min(progress, 100)}%`;
                if (progress >= 100) {
                    clearInterval(interval);
                    status.textContent = "Complete";
                    if (nextBtn) nextBtn.disabled = false;
                }
            }, 50);
        });
    });

    document.querySelectorAll('[data-stop-button]').forEach(btn => {
        btn.addEventListener('click', () => {
            const stage = btn.getAttribute('data-stop-button');
            stopStageRun(stage);
        });
    });
}

/**
 * ANIMATED SIMULATION ENGINE
 * Runs one full forward projection, then plays a lightweight acquisition animation
 * while progressively revealing the already-computed sinogram.
 */
async function runSimulationAnimation(progressBar, statusText, nextButton, controller) {
    const signal = controller.signal;
    const geoCanvas = document.getElementById('geometry-canvas');
    const sinoCanvas = document.getElementById('sinogram-canvas');
    if (!geoCanvas || !sinoCanvas) return;

    const geoCtx = geoCanvas.getContext('2d');
    const sinoCtx = sinoCanvas.getContext('2d');

    const numSources = parseInt(document.getElementById('sim-geometry-select').value);
    const mAs = document.getElementById('sim-exposure-select').value;
    
    const pIdx = document.getElementById('patient-slider').value;
    const pId = currentPatientIds[pIdx];
    const sIdx = document.getElementById('slice-slider').value;

    statusText.textContent = `Performing one-shot GPU forward projection and photon-noise simulation...`;

    try {
        const fullResp = await fetch(`/api/simulate/full-forward/${selectedDataset}/${pId}/${sIdx}/${numSources}?ww=${windowWidth}&wl=${windowLevel}&mAs=${mAs}`, { signal });
        const fullData = await fullResp.json();

        if (fullData.error) {
            statusText.textContent = `Error: ${fullData.error}`;
            return;
        }

        // Load Base Images
        const [imgBase, imgSino, imgMask] = await Promise.all([
            new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = `data:image/png;base64,${fullData.geometry_base}`; }),
            new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = `data:image/png;base64,${fullData.sinogram_full}`; }),
            new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = `data:image/png;base64,${fullData.sinogram_mask}`; })
        ]);

        geoCanvas.width = imgBase.width;
        geoCanvas.height = imgBase.height;
        sinoCanvas.width = imgSino.width;
        sinoCanvas.height = imgSino.height;

        statusText.textContent = `Pre-loading animation overlays for maximum performance...`;

        const totalSources = numSources;
        const overlayImages = [];
        const loadPromises = [];

        for (let i = 0; i < totalSources; i++) {
            const img = new Image();
            const p = new Promise(resolve => {
                img.onload = () => resolve(img);
                img.onerror = () => resolve(null);
                img.src = `/static/sim_animations/${totalSources}/frame_${i}.png`;
            });
            overlayImages.push(img);
            loadPromises.push(p);
        }

        await Promise.all(loadPromises);

        statusText.textContent = `Looping through simulation sequence at maximum hardware speed...`;

        let currentSource = 0;

        for (currentSource = 0; currentSource < totalSources; currentSource++) {
            if (signal.aborted || workflowState === 'START') break;

            const imgOver = overlayImages[currentSource];

            // 1. Draw Geometry
            geoCtx.clearRect(0, 0, geoCanvas.width, geoCanvas.height);
            geoCtx.drawImage(imgBase, 0, 0);
            if (imgOver) {
                geoCtx.drawImage(imgOver, 0, 0);
            }

            // 2. Draw Sinogram with "Curtain"
            sinoCtx.clearRect(0, 0, sinoCanvas.width, sinoCanvas.height);
            sinoCtx.drawImage(imgSino, 0, 0);
            
            const progress = (currentSource + 1) / totalSources;

            // Plot area boundaries (15% to 85% of figure width and height)
            const plotTop = 0.15 * sinoCanvas.height;
            const plotHeight = 0.7 * sinoCanvas.height;

            const curtainTop = plotTop + progress * plotHeight;
            const curtainHeight = (plotTop + plotHeight) - curtainTop;

            if (curtainHeight > 0) {
                // Draw the transparent mask cropped so it only covers the unacquired part
                sinoCtx.drawImage(
                    imgMask,
                    0, curtainTop, sinoCanvas.width, curtainHeight,  // source crop
                    0, curtainTop, sinoCanvas.width, curtainHeight   // destination position
                );
            }

            // Update Progress
            progressBar.style.width = `${progress * 100}%`;
            statusText.textContent = `Views acquired: ${currentSource + 1} / ${totalSources}`;
            
            // Minimal pause to trigger browser repaint and run as fast as possible
            await sleep(1, signal);
        }

        if (!signal.aborted && workflowState !== 'START') {
            statusText.textContent = "Acquisition Complete (High Fidelity)";
            workflowState = 'SIMULATED';
            updateWorkflowUI();
            if (nextButton) nextButton.disabled = false;
        }

    } catch (e) {
        if (isAbortError(e)) {
            resetSimulationStage('Simulation Stopped');
            return;
        }
        console.error("Simulation failed", e);
        statusText.textContent = "Simulation Error";
    } finally {
        finishStageRun('simulate-ct-data', controller);
    }
}

async function runEigenFBPRecon(progressBar, statusText, nextButton, controller) {
    const signal = controller.signal;
    const pIdx = document.getElementById('patient-slider').value;
    const pId = currentPatientIds[pIdx];
    const sIdx = document.getElementById('slice-slider').value;
    const numSources = parseInt(document.getElementById('sim-geometry-select').value);

    // Find the display containers for this stage
    const boxUnfiltered = document.getElementById('recon-box-unfiltered');
    const boxFiltered = document.getElementById('recon-box-filtered');
    const timingUnfiltered = document.getElementById('timing-unfiltered');
    const timingFiltered = document.getElementById('timing-filtered');
    
    const updateImageView = (container, src) => {
        if (!container) return;
        container.innerHTML = '';
        const img = document.createElement('img');
        img.src = src;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        img.style.backgroundColor = 'black';
        container.appendChild(img);
    };

    try {
        // Reset Views
        if (boxUnfiltered) boxUnfiltered.innerHTML = '';
        if (boxFiltered) boxFiltered.innerHTML = '';
        if (timingUnfiltered) timingUnfiltered.textContent = '-- ms';
        if (timingFiltered) timingFiltered.textContent = '-- ms';

        statusText.textContent = 'Preparing unfiltered backprojection once...';
        progressBar.style.width = '25%';

        const sliderLow = document.getElementById('fbp-low-slider');
        const sliderMid = document.getElementById('fbp-mid-slider');
        const sliderHigh = document.getElementById('fbp-high-slider');

        const wLow = sliderLow ? parseFloat(sliderLow.value) / 100.0 : 1.0;
        const wMid = sliderMid ? parseFloat(sliderMid.value) / 100.0 : 0.5;
        const wHigh = sliderHigh ? parseFloat(sliderHigh.value) / 100.0 : 0.2;

        const resp = await fetch(`/api/reconstruct/runtime/${selectedDataset}/${pId}/${sIdx}/${numSources}?initial_step=unfiltered&final_step=filter&ww=${windowWidth}&wl=${windowLevel}&w_low=${wLow}&w_mid=${wMid}&w_high=${wHigh}`, { signal });
        const res = await resp.json();

        if (res.error) {
            statusText.textContent = `FlashFBP Failed: ${res.error}`;
            return;
        }

        if (res.initial_image) {
            updateImageView(boxUnfiltered, res.initial_image);
            timingUnfiltered.textContent = `${res.initial_time_ms.toFixed(1)} ms`;
        }

        statusText.textContent = 'Applying 4096-mode sparse eigen filter...';
        progressBar.style.width = '70%';
        await sleep(600, signal);

        if (res.final_image) {
            updateImageView(boxFiltered, res.final_image);
            timingFiltered.textContent = `${res.final_time_ms.toFixed(1)} ms`;
            progressBar.style.width = '100%';
            statusText.textContent = '4096-mode Sparse Eigen FlashFBP complete';
            if (nextButton) nextButton.disabled = false;
            isFbpRunComplete = true;
        } else {
            statusText.textContent = 'FlashFBP final recon missing';
        }
    } catch (e) {
        if (isAbortError(e)) {
            resetEigenStage('Reconstruction Stopped');
            return;
        }
        console.error(e);
        statusText.textContent = "Network Error during reconstruction";
    } finally {
        finishStageRun('eigen-fbp-recon', controller);
        updateWorkflowUI();
    }
}

async function runNeuralSpeedRecon(progressBar, statusText, nextButton, controller) {
    const signal = controller.signal;
    const pIdx = document.getElementById('patient-slider').value;
    const pId = currentPatientIds[pIdx];
    const sIdx = document.getElementById('slice-slider').value;
    const numSources = parseInt(document.getElementById('sim-geometry-select').value);
    const exposureMas = parseFloat(document.getElementById('sim-exposure-select').value);

    const boxInit = document.getElementById('dlr-box-init');
    const boxFinal = document.getElementById('dlr-box-final');
    const timingInit = document.getElementById('timing-dlr-init');
    const timingFinal = document.getElementById('timing-dlr-final');

    const updateImageView = (container, src) => {
        if (!container) return;
        container.innerHTML = '';
        const img = document.createElement('img');
        img.src = src;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        img.style.backgroundColor = 'black';
        container.appendChild(img);
    };

    try {
        // Run FBP initialization first if it hasn't been run
        if (!isFbpRunComplete) {
            statusText.textContent = "FBP initialization required. Running FBP first...";
            try {
                await runEigenFBPRecon(
                    document.querySelector('[data-progress-fill="eigen-fbp-recon"]'),
                    document.querySelector('[data-progress-status="eigen-fbp-recon"]'),
                    document.querySelector('[data-next-stage="eigen-fbp-recon"]'),
                    createStageRun('eigen-fbp-recon')
                );
            } catch (fbpError) {
                console.error("FBP pre-run failed of NeuralSpark:", fbpError);
            }
        }

        if (boxInit) boxInit.innerHTML = '';
        if (boxFinal) boxFinal.innerHTML = '';
        if (timingInit) timingInit.textContent = '-- ms';
        if (timingFinal) timingFinal.textContent = '-- ms';

        statusText.textContent = 'Preparing Full FBP initialization once...';
        progressBar.style.width = '25%';

        const resp = await fetch(`/api/reconstruct/dlr/${selectedDataset}/${pId}/${sIdx}/${numSources}?exposure_mas=${exposureMas}&ww=${windowWidth}&wl=${windowLevel}`, { signal });
        const res = await resp.json();

        if (res.error) {
            statusText.textContent = `NeuralSpark Failed: ${res.error}`;
            return;
        }

        if (res.initial_image) {
            updateImageView(boxInit, res.initial_image);
            timingInit.textContent = `${res.initial_time_ms.toFixed(1)} ms`;
        }

        statusText.textContent = 'Applying NeuralSpark restoration step...';
        progressBar.style.width = '70%';
        await sleep(500, signal);

        if (res.final_image) {
            updateImageView(boxFinal, res.final_image);
            timingFinal.textContent = `${res.final_time_ms.toFixed(1)} ms`;
            progressBar.style.width = '100%';
            statusText.textContent = 'NeuralSpark reconstruction complete';
            if (nextButton) nextButton.disabled = false;
        } else {
            statusText.textContent = 'NeuralSpark final recon missing';
        }
    } catch (e) {
        if (isAbortError(e)) {
            resetNeuralStage('Reconstruction Stopped');
            return;
        }
        console.error(e);
        statusText.textContent = 'Network Error during NeuralSpark reconstruction';
    } finally {
        finishStageRun('deep-learning-recon', controller);
        updateWorkflowUI();
    }
}

async function runGenerativeVisionRecon(progressBar, statusText, nextButton, controller) {
    const signal = controller.signal;
    const pIdx = document.getElementById('patient-slider').value;
    const pId = currentPatientIds[pIdx];
    const sIdx = document.getElementById('slice-slider').value;
    const numSources = parseInt(document.getElementById('sim-geometry-select').value);
    const exposureMas = parseFloat(document.getElementById('sim-exposure-select').value);

    // Params from UI
    const steps = document.getElementById('diffusion-steps-slider').value;
    const sigmaMaxHU = Math.pow(10, parseFloat(document.getElementById('sigma-max-slider').value));
    const sigmaMinHU = Math.pow(10, parseFloat(document.getElementById('sigma-min-slider').value));
    const temperature = parseFloat(document.getElementById('diffusion-temperature-slider').value);
    
    // Convert HU std to attenuation std (scaleOnly=true)
    const sigmaMax = HU_to_atten(sigmaMaxHU, true);
    const sigmaMin = HU_to_atten(sigmaMinHU, true);

    const solver = document.querySelector('input[name="diffusion-solver"]:checked')?.value || 'heun';
    const modelVariant = document.querySelector('input[name="diffusion-model-variant"]:checked')?.value || 'base';

    const numSamples = parseInt(document.getElementById('num-samples-slider')?.value || '4');
    const langevinSteps = parseInt(document.getElementById('langevin-steps-slider')?.value || '100');

    lastStreamData = null;

    try {
        const jobId = buildGenerativeJobId();
        stageRunState['generative-ai-recon'].jobId = jobId;

        console.log(`DEBUG [Generative]: Starting combined sampling with JobID: ${jobId}, steps=${steps}, numSamples=${numSamples}, langevinSteps=${langevinSteps}, variant=${modelVariant}`);
        statusText.textContent = `Starting GenerativeVision Diffusion (${steps} steps)...`;
        progressBar.style.width = '10%';

        const streamUrl = `/api/reconstruct/generative/${selectedDataset}/${pId}/${sIdx}/${numSources}?steps=${steps}&sigma_max=${sigmaMax}&sigma_min=${sigmaMin}&solver=${solver}&temperature=${temperature}&exposure_mas=${exposureMas}&ww=${windowWidth}&wl=${windowLevel}&job_id=${jobId}&num_samples=${numSamples}&langevin_steps=${langevinSteps}&model_variant=${modelVariant}`;
        const response = await fetch(streamUrl, { signal });
        if (!response.ok || !response.body) {
            throw new Error(`Generative stream failed with HTTP ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let chunkRemainder = '';

        while (true) {
            let result;
            try {
                result = await reader.read();
            } catch (e) {
                console.warn("Stream interrupted during read", e);
                break;
            }
            const { value, done } = result;
            if (done) break;
            
            const chunk = chunkRemainder + decoder.decode(value);
            const lines = chunk.split('\n');
            chunkRemainder = lines.pop();
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        if (data.error) {
                            console.error("Generative Backend Error:", data.error);
                            throw new Error(data.error);
                        }
                        
                        lastStreamData = data;
                        updateDisplay();

                        const progress = 10 + (data.step / data.total_steps) * 90;
                        progressBar.style.width = `${progress}%`;
                        
                        // Check if we are in the Langevin walk phase
                        if (data.step > parseInt(steps)) {
                            const langCurrStep = data.step - parseInt(steps);
                            statusText.textContent = `Langevin Random Walk (at σ_min): Step ${langCurrStep} / ${langevinSteps} (σ_min = ${atten_to_HU(data.sigma, true).toFixed(2)} HU)`;
                        } else {
                            statusText.textContent = `Null-space diffusion: Step ${data.step} / ${data.total_steps} (σ=${atten_to_HU(data.sigma, true).toFixed(2)} HU)`;
                        }
                    } catch (e) {
                        console.error("JSON Parse error in stream", e);
                        throw e;
                    }
                }
            }
        }
        
        if (!signal.aborted) {
            statusText.textContent = 'GenerativeVision sampling complete';
            progressBar.style.width = '100%';
        }
    } catch (e) {
        if (isAbortError(e)) {
            statusText.textContent = 'Sampling Stopped';
            console.log("DEBUG [Generative]: Sampling aborted by user");
            return;
        }
        console.error(e);
        statusText.textContent = `GenerativeVision failed: ${e.message || 'Network Error'}`;
    } finally {
        finishStageRun('generative-ai-recon', controller);
        updateWorkflowUI();
    }
}

let sliderDebounce = null;
const SLIDER_SLEEP = 100; // 100ms intentional sleep/debounce for clinical stability

window.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll('.dataset-card').forEach(card => {
        card.addEventListener('click', () => selectDataset(card.dataset.dataset));
    });
    
    document.getElementById('patient-slider').addEventListener('input', (e) => {
        if (sliderDebounce) clearTimeout(sliderDebounce);
        sliderDebounce = setTimeout(() => {
            updatePatientDisplay(e.target.value);
        }, SLIDER_SLEEP);
    });

    document.getElementById('slice-slider').addEventListener('input', (e) => {
        if (sliderDebounce) clearTimeout(sliderDebounce);
        sliderDebounce = setTimeout(() => {
            updatePatientDisplay(document.getElementById('patient-slider').value);
        }, SLIDER_SLEEP);
    });

    document.getElementById('window-width-slider').addEventListener('input', (e) => {
        windowWidth = parseInt(e.target.value);
        document.getElementById('ww-display').textContent = windowWidth;
        if (sliderDebounce) clearTimeout(sliderDebounce);
        sliderDebounce = setTimeout(() => updateLiveFrame(), SLIDER_SLEEP);
    });

    document.getElementById('window-level-slider').addEventListener('input', (e) => {
        windowLevel = parseInt(e.target.value);
        document.getElementById('wl-display').textContent = windowLevel;
        if (sliderDebounce) clearTimeout(sliderDebounce);
        sliderDebounce = setTimeout(() => updateLiveFrame(), SLIDER_SLEEP);
    });

    // --- Window/Level presets (Load Patient) ---
    document.querySelectorAll('[id^="wlpreset-"]').forEach((btn) => {
        btn.addEventListener('click', () => {
            const ww = parseInt(btn.getAttribute('data-ww'));
            const wl = parseInt(btn.getAttribute('data-wl'));
            windowWidth = ww;
            windowLevel = wl;
            const wwS = document.getElementById('window-width-slider');
            const wlS = document.getElementById('window-level-slider');
            if (wwS) wwS.value = ww;
            if (wlS) wlS.value = wl;
            document.getElementById('ww-display').textContent = ww;
            document.getElementById('wl-display').textContent = wl;
            // Highlight the active preset (same behavior as the FlashFBP presets).
            document.querySelectorAll('[id^="wlpreset-"]').forEach((b) => {
                b.classList.remove('active');
                b.style.backgroundColor = '';
                b.style.color = '';
            });
            btn.classList.add('active');
            btn.style.backgroundColor = '#0a63a8';
            btn.style.color = 'white';
            updateLiveFrame();
        });
    });

    // --- STAGE 3: EIGEN-FBP FILTER LISTENERS ---
    const fbpLowSlider = document.getElementById('fbp-low-slider');
    const fbpMidSlider = document.getElementById('fbp-mid-slider');
    const fbpHighSlider = document.getElementById('fbp-high-slider');

    const updateFbpSliderDisplays = () => {
        if (fbpLowSlider) {
            const disp = document.getElementById('fbp-low-display');
            if (disp) disp.textContent = `${fbpLowSlider.value}%`;
        }
        if (fbpMidSlider) {
            const disp = document.getElementById('fbp-mid-display');
            if (disp) disp.textContent = `${fbpMidSlider.value}%`;
        }
        if (fbpHighSlider) {
            const disp = document.getElementById('fbp-high-display');
            if (disp) disp.textContent = `${fbpHighSlider.value}%`;
        }
    };

    if (fbpLowSlider) {
        fbpLowSlider.addEventListener('input', (e) => {
            const disp = document.getElementById('fbp-low-display');
            if (disp) disp.textContent = `${e.target.value}%`;
            document.querySelectorAll('[id^="preset-"]').forEach(btn => {
                btn.classList.remove('active');
                btn.style.backgroundColor = '';
                btn.style.color = '';
            });
            if (sliderDebounce) clearTimeout(sliderDebounce);
            sliderDebounce = setTimeout(() => {
                updateFilterPlot();
                if (workflowState === 'SIMULATED') {
                    updateFbpReconstruction();
                }
            }, SLIDER_SLEEP);
        });
    }
    if (fbpMidSlider) {
        fbpMidSlider.addEventListener('input', (e) => {
            const disp = document.getElementById('fbp-mid-display');
            if (disp) disp.textContent = `${e.target.value}%`;
            document.querySelectorAll('[id^="preset-"]').forEach(btn => {
                btn.classList.remove('active');
                btn.style.backgroundColor = '';
                btn.style.color = '';
            });
            if (sliderDebounce) clearTimeout(sliderDebounce);
            sliderDebounce = setTimeout(() => {
                updateFilterPlot();
                if (workflowState === 'SIMULATED') {
                    updateFbpReconstruction();
                }
            }, SLIDER_SLEEP);
        });
    }
    if (fbpHighSlider) {
        fbpHighSlider.addEventListener('input', (e) => {
            const disp = document.getElementById('fbp-high-display');
            if (disp) disp.textContent = `${e.target.value}%`;
            document.querySelectorAll('[id^="preset-"]').forEach(btn => {
                btn.classList.remove('active');
                btn.style.backgroundColor = '';
                btn.style.color = '';
            });
            if (sliderDebounce) clearTimeout(sliderDebounce);
            sliderDebounce = setTimeout(() => {
                updateFilterPlot();
                if (workflowState === 'SIMULATED') {
                    updateFbpReconstruction();
                }
            }, SLIDER_SLEEP);
        });
    }

    const setFbpSliders = (low, mid, high) => {
        if (fbpLowSlider) fbpLowSlider.value = low;
        if (fbpMidSlider) fbpMidSlider.value = mid;
        if (fbpHighSlider) fbpHighSlider.value = high;
        updateFbpSliderDisplays();
        updateFilterPlot();
        if (workflowState === 'SIMULATED') {
            updateFbpReconstruction();
        }
    };

    const sharpBtn = document.getElementById('preset-sharp-btn');
    const rampBtn = document.getElementById('preset-ramp-btn');
    const softBtn = document.getElementById('preset-soft-btn');

    const setActivePreset = (btnId) => {
        document.querySelectorAll('[id^="preset-"]').forEach(btn => {
            btn.classList.remove('active');
            btn.style.backgroundColor = '';
            btn.style.color = '';
        });
        const activeBtn = document.getElementById(btnId);
        if (activeBtn) {
            activeBtn.classList.add('active');
            activeBtn.style.backgroundColor = '#0a63a8';
            activeBtn.style.color = 'white';
        }
    };

    if (sharpBtn) {
        sharpBtn.addEventListener('click', () => {
            setFbpSliders(100, 80, 50);
            setActivePreset('preset-sharp-btn');
        });
    }
    if (rampBtn) {
        rampBtn.addEventListener('click', () => {
            setFbpSliders(100, 100, 100);
            setActivePreset('preset-ramp-btn');
        });
    }
    if (softBtn) {
        softBtn.addEventListener('click', () => {
            setFbpSliders(100, 30, 5);
            setActivePreset('preset-soft-btn');
        });
    }

    // Initialize FBP displays & chart
    updateFbpSliderDisplays();
    updateFilterPlot();

    // --- STAGE 6: GENERATIVE VISION LISTENERS ---
    const genStepsSlider = document.getElementById('diffusion-steps-slider');
    const sigmaMaxSlider = document.getElementById('sigma-max-slider');
    const sigmaMinSlider = document.getElementById('sigma-min-slider');
    const tempSlider = document.getElementById('diffusion-temperature-slider');

    if (genStepsSlider) {
        genStepsSlider.addEventListener('input', (e) => {
            document.getElementById('diffusion-steps-display').textContent = e.target.value;
        });
    }
    const formatSigma = (val) => {
        if (val >= 100) return val.toFixed(0);
        if (val >= 10) return val.toFixed(1);
        if (val >= 1) return val.toFixed(2);
        if (val >= 0.001) return val.toFixed(4);
        return val.toExponential(1);
    };

    if (sigmaMaxSlider) {
        sigmaMaxSlider.addEventListener('input', (e) => {
            const minSlider = document.getElementById('sigma-min-slider');
            if (minSlider && parseFloat(minSlider.value) > parseFloat(e.target.value)) {
                minSlider.value = e.target.value;
                const valHUMin = Math.pow(10, parseFloat(e.target.value));
                document.getElementById('sigma-min-display').textContent = formatSigma(valHUMin);
            }
            const valHU = Math.pow(10, parseFloat(e.target.value));
            document.getElementById('sigma-max-display').textContent = formatSigma(valHU);
        });
    }
    if (sigmaMinSlider) {
        sigmaMinSlider.addEventListener('input', (e) => {
            const maxSlider = document.getElementById('sigma-max-slider');
            if (maxSlider && parseFloat(e.target.value) > parseFloat(maxSlider.value)) {
                e.target.value = maxSlider.value;
            }
            const valHU = Math.pow(10, parseFloat(e.target.value));
            document.getElementById('sigma-min-display').textContent = formatSigma(valHU);
        });
    }
    if (tempSlider) {
        tempSlider.addEventListener('input', (e) => {
            document.getElementById('diffusion-temperature-display').textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    const langevinWalkCheck = document.getElementById('langevin-walk-checkbox');
    if (langevinWalkCheck) {
        langevinWalkCheck.addEventListener('change', (e) => {
            if (e.target.checked) {
                // Trigger Langevin walk automatically by simulating click on run button
                const runButton = document.querySelector('[data-run-button="generative-ai-recon"]');
                if (runButton) {
                    runButton.click();
                }
            } else {
                // Abort the walk stream
                stopStageRun('generative-ai-recon');
            }
        });
    }

    const langevinStepsSlider = document.getElementById('langevin-steps-slider');
    if (langevinStepsSlider) {
        langevinStepsSlider.addEventListener('input', (e) => {
            const displayObj = document.getElementById('langevin-steps-display');
            if (displayObj) displayObj.textContent = e.target.value;
        });
    }

    const numSamplesSlider = document.getElementById('num-samples-slider');
    if (numSamplesSlider) {
        numSamplesSlider.addEventListener('input', (e) => {
            const displayObj = document.getElementById('num-samples-display');
            if (displayObj) displayObj.textContent = e.target.value;
        });
    }

    // Setup Display Mode Toggle via Radio Buttons (A, B, C)
    const modeRadios = document.querySelectorAll('input[name="display-mode"]');
    modeRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            currentDisplayMode = radio.value;
            console.log(`DEBUG [Generative]: Switched display mode to ${currentDisplayMode}`);
            updateDisplay();
        });
    });

    // --- STAGE 4: ITERATIVE RECON LISTENERS ---
    const iterCountSlider = document.getElementById('iter-count-slider');
    const tvSlider = document.getElementById('tv-strength-slider');
    const lrSlider = document.getElementById('lr-slider');
    const precondCheck = document.getElementById('use-precond-check');

    if (iterCountSlider) {
        iterCountSlider.addEventListener('input', (e) => {
            document.getElementById('iter-count-display').textContent = e.target.value;
        });
    }
    if (tvSlider) {
        tvSlider.addEventListener('input', (e) => {
            const val = Math.pow(10, parseFloat(e.target.value));
            let displayVal;
            if (val < 0.0001) displayVal = "0";
            else if (val >= 10000) displayVal = val.toExponential(1);
            else if (val >= 10) displayVal = val.toFixed(1);
            else if (val >= 1) displayVal = val.toFixed(2);
            else displayVal = val.toFixed(4);
            document.getElementById('tv-strength-display').textContent = displayVal;
        });
    }
    if (lrSlider) {
        lrSlider.addEventListener('input', (e) => {
            const val = Math.pow(10, parseFloat(e.target.value));
            document.getElementById('lr-display').textContent = val.toFixed(3);
        });
    }
    if (precondCheck) {
        precondCheck.addEventListener('change', (e) => {
            if (e.target.checked) {
                lrSlider.value = -1; // 10^-1 = 0.1
                document.getElementById('lr-display').textContent = "0.100";
            } else {
                lrSlider.value = -7; // 10^-7 for raw GD
                document.getElementById('lr-display').textContent = "0.000";
            }
        });
    }

    document.getElementById('load-patient-button').addEventListener('click', async () => {
        const fill = document.querySelector(`[data-progress-fill="load-patient"]`);
        const status = document.querySelector(`[data-progress-status="load-patient"]`);
        const pIdx = document.getElementById('patient-slider').value;
        const pId = currentPatientIds[pIdx];
        const sIdx = document.getElementById('slice-slider').value;
        
        status.textContent = `Processing Patient ${pId}...`;
        fill.style.width = '20%';

        try {
            const resp = await fetch(`/api/stage-patient/${selectedDataset}/${pId}/${sIdx}`);
            const result = await resp.json();
            
            if (result.status === 'success') {
                fill.style.width = '100%';
                status.textContent = `Patient ${pId} Staged (256x256 @ 1.6mm)`;
                document.getElementById('next-to-simulate').disabled = false;
                workflowState = 'LOADED';
                updateWorkflowUI();
            } else {
                status.textContent = "Processing Failed: " + result.message;
            }
        } catch (e) {
            console.error(e);
            status.textContent = "Network Error during processing";
        }
    });

    // --- STAGE 7: EVALUATION TAB LISTENERS ---
    const evalScopeRadios = document.querySelectorAll('input[name="eval-scope"]');
    const evalPatientsRow = document.getElementById('eval-num-patients-row');
    const evalSlider = document.getElementById('eval-num-patients-slider');
    const evalDisplay = document.getElementById('eval-num-patients-display');
    const btnRunEval = document.getElementById('btn-run-evaluation');

    if (evalScopeRadios) {
        evalScopeRadios.forEach(radio => {
            radio.addEventListener('change', () => {
                if (radio.value === 'single_patient') {
                    if (evalPatientsRow) evalPatientsRow.style.display = 'none';
                } else {
                    if (evalPatientsRow) evalPatientsRow.style.display = 'grid';
                }
            });
        });
    }

    if (evalSlider && evalDisplay) {
        evalSlider.addEventListener('input', (e) => {
            evalDisplay.textContent = e.target.value;
        });
    }

    if (btnRunEval) {
        btnRunEval.addEventListener('click', () => {
            const fill = document.querySelector('[data-progress-fill="evaluation"]');
            const status = document.querySelector('[data-progress-status="evaluation"]');
            
            const controller = createStageRun('evaluation');
            btnRunEval.disabled = true;
            runEvaluationBenchmark(fill, status, btnRunEval, controller);
        });
    }

    // --- MOTIONSCOPE TAB LISTENERS ---
    document.querySelectorAll('input[name="motion-views"], input[name="motion-clock"], input[name="motion-method"]')
        .forEach(radio => radio.addEventListener('change', updateMotionScopeGifs));

    handleStageNavigation();
    handleSimulationRuns();
    initDatasetPreviews();
    updateWorkflowUI();
});

async function runIterativeRecon(progressBar, statusText, nextButton, controller) {
    console.log("DEBUG [landing.js]: runIterativeRecon started");
    const signal = controller.signal;
    const pIdx = document.getElementById('patient-slider').value;
    const pId = currentPatientIds[pIdx];
    const sIdx = document.getElementById('slice-slider').value;
    const numSources = parseInt(document.getElementById('sim-geometry-select').value);

    // Params
    const iters = document.getElementById('iter-count-slider').value;
    const tv = Math.pow(10, parseFloat(document.getElementById('tv-strength-slider').value));
    const lr = Math.pow(10, parseFloat(document.getElementById('lr-slider').value));
    const usePrecond = document.getElementById('use-precond-check').checked;

    console.log(`DEBUG [landing.js]: pId=${pId}, sIdx=${sIdx}, numSources=${numSources}, iters=${iters}, tv=${tv}, lr=${lr}, precond=${usePrecond}`);

    const boxLoss = document.getElementById('iter-box-loss');
    const boxLive = document.getElementById('iter-box-live');

    const updateImageView = (container, src) => {
        if (!container) return;
        container.innerHTML = '';
        const img = document.createElement('img');
        img.src = src;
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        img.style.backgroundColor = 'black';
        container.appendChild(img);
    };

    try {
        // 1. Run FBP initialization first if it hasn't been run
        if (!isFbpRunComplete) {
            statusText.textContent = "FBP initialization required. Running FBP first...";
            try {
                await runEigenFBPRecon(
                    document.querySelector('[data-progress-fill="eigen-fbp-recon"]'),
                    document.querySelector('[data-progress-status="eigen-fbp-recon"]'),
                    document.querySelector('[data-next-stage="eigen-fbp-recon"]'),
                    createStageRun('eigen-fbp-recon')
                );
            } catch (fbpError) {
                console.error("FBP pre-run failed of MBIR:", fbpError);
            }
        }

        // 2. Start Streaming Iterative Recon
        statusText.textContent = `Starting FidelityMBIR (TV=${tv < 1e-5 ? "0" : tv.toFixed(5)})...`;
        progressBar.style.width = '20%';

        const jobId = buildIterativeJobId();
        stageRunState['model-based-iterative-recon'].jobId = jobId;
        const streamUrl = `/api/reconstruct/iterative/${selectedDataset}/${pId}/${sIdx}/${numSources}?iters=${iters}&tv=${tv}&lr=${lr}&precond=${usePrecond}&ww=${windowWidth}&wl=${windowLevel}&job_id=${jobId}`;
        console.log(`DEBUG [landing.js]: fetching stream from ${streamUrl}`);
        
        const response = await fetch(streamUrl, { signal });
        console.log(`DEBUG [landing.js]: response status: ${response.status}`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let chunkRemainder = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = chunkRemainder + decoder.decode(value);
            const lines = chunk.split('\n');
            
            // Last element might be incomplete
            chunkRemainder = lines.pop();
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        if (data.error) {
                            statusText.textContent = "Error: " + data.error;
                            break;
                        }
                        if (data.image) {
                            updateImageView(boxLive, data.image);
                            if (data.loss_plot) {
                                updateImageView(boxLoss, data.loss_plot);
                            }
                            const progress = 20 + (data.iteration / data.total) * 80;
                            progressBar.style.width = `${progress}%`;
                            statusText.textContent = `FidelityMBIR: Iteration ${data.iteration} / ${data.total}`;
                        }
                    } catch (e) {
                        console.warn("Error parsing SSE chunk", e);
                    }
                }
            }
        }

        if (!signal.aborted) {
            statusText.textContent = "FidelityMBIR Complete";
            progressBar.style.width = '100%';
            if (nextButton) nextButton.disabled = false;
        }

    } catch (e) {
        if (isAbortError(e)) {
            resetIterativeStage('FidelityMBIR Stopped');
            return;
        }
        console.error(e);
        statusText.textContent = "FidelityMBIR Failed";
    } finally {
        finishStageRun('model-based-iterative-recon', controller);
        updateWorkflowUI();
    }
}
