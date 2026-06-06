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
const stageRunState = {
    'simulate-ct-data': { controller: null, jobId: null },
    'eigen-fbp-recon': { controller: null, jobId: null },
    'model-based-iterative-recon': { controller: null, jobId: null },
    'deep-learning-recon': { controller: null, jobId: null },
    'generative-ai-recon': { controller: null, jobId: null }
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
            if (currentDisplayMode === 'animation') {
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

    if (M <= 1 || currentDisplayMode === 'sample') {
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
        // Option C: Generative Mean Mode
        if (lastStreamData.xt) globalUpdateImageView(boxXt, lastStreamData.xt);

        // Reconstruction should show the mean_x0 (which is the average posterior mean)
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
    const boxInit = document.getElementById('iter-box-init');
    const boxLoss = document.getElementById('iter-box-loss');
    const boxLive = document.getElementById('iter-box-live');

    if (boxInit) boxInit.innerHTML = '';
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
    const sidebarRecons = document.querySelectorAll('[data-stage-button$="-recon"]');
    
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
    }

    if (stageId === 'model-based-iterative-recon') {
        const boxInit = document.getElementById('iter-box-init');
        const boxLoss = document.getElementById('iter-box-loss');
        const boxLive = document.getElementById('iter-box-live');
        if (boxInit) boxInit.innerHTML = '';
        if (boxLoss) boxLoss.innerHTML = '';
        if (boxLive) boxLive.innerHTML = '';
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
    const timingSino = document.getElementById('timing-sino');

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
        console.error("Failed to prepare recon comparison", e);
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
        console.error('Failed to prepare NeuralSpeed comparison', e);
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
        const [imgBase, imgSino] = await Promise.all([
            new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = `data:image/png;base64,${fullData.geometry_base}`; }),
            new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = `data:image/png;base64,${fullData.sinogram_full}`; })
        ]);

        geoCanvas.width = imgBase.width;
        geoCanvas.height = imgBase.height;
        sinoCanvas.width = imgSino.width;
        sinoCanvas.height = imgSino.height;

        statusText.textContent = `Animating acquisition overlay while reusing the precomputed sinogram...`;

        let currentSource = 0;
        const totalSources = numSources;

        for (currentSource = 0; currentSource < totalSources; currentSource++) {
            if (signal.aborted || workflowState === 'START') break;

            const overResp = await fetch(`/api/simulate/overlay/${totalSources}/${currentSource}`, { signal });
            const overData = await overResp.json();
            
            const imgOver = await new Promise(r => { 
                const i = new Image(); 
                i.onload = () => r(i); 
                i.src = `data:image/png;base64,${overData.overlay}`; 
            });

            // 1. Draw Geometry
            geoCtx.clearRect(0, 0, geoCanvas.width, geoCanvas.height);
            geoCtx.drawImage(imgBase, 0, 0);
            geoCtx.drawImage(imgOver, 0, 0);

            // 2. Draw Sinogram with "Curtain"
            sinoCtx.clearRect(0, 0, sinoCanvas.width, sinoCanvas.height);
            sinoCtx.drawImage(imgSino, 0, 0);
            
            const progress = (currentSource + 1) / totalSources;
            const curtainY = progress * sinoCanvas.height;
            sinoCtx.fillStyle = 'black';
            sinoCtx.fillRect(0, curtainY, sinoCanvas.width, sinoCanvas.height - curtainY);

            // Update Progress
            progressBar.style.width = `${progress * 100}%`;
            statusText.textContent = `Views acquired: ${currentSource + 1} / ${totalSources}`;
            
            // Brief pause for visual effect - if it's too fast, we can't see the detail
            // Adjust delay to control total animation time
            await sleep(10, signal);
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

        const resp = await fetch(`/api/reconstruct/runtime/${selectedDataset}/${pId}/${sIdx}/${numSources}?initial_step=unfiltered&final_step=filter&ww=${windowWidth}&wl=${windowLevel}`, { signal });
        const res = await resp.json();

        if (res.error) {
            statusText.textContent = `EigenFBP Failed: ${res.error}`;
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
            statusText.textContent = '4096-mode Sparse Eigen FBP complete';
            if (nextButton) nextButton.disabled = false;
        } else {
            statusText.textContent = 'EigenFBP final recon missing';
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
        if (boxInit) boxInit.innerHTML = '';
        if (boxFinal) boxFinal.innerHTML = '';
        if (timingInit) timingInit.textContent = '-- ms';
        if (timingFinal) timingFinal.textContent = '-- ms';

        statusText.textContent = 'Preparing Full FBP initialization once...';
        progressBar.style.width = '25%';

        const resp = await fetch(`/api/reconstruct/dlr/${selectedDataset}/${pId}/${sIdx}/${numSources}?exposure_mas=${exposureMas}&ww=${windowWidth}&wl=${windowLevel}`, { signal });
        const res = await resp.json();

        if (res.error) {
            statusText.textContent = `NeuralSpeed Failed: ${res.error}`;
            return;
        }

        if (res.initial_image) {
            updateImageView(boxInit, res.initial_image);
            timingInit.textContent = `${res.initial_time_ms.toFixed(1)} ms`;
        }

        statusText.textContent = 'Applying NeuralSpeed restoration step...';
        progressBar.style.width = '70%';
        await sleep(500, signal);

        if (res.final_image) {
            updateImageView(boxFinal, res.final_image);
            timingFinal.textContent = `${res.final_time_ms.toFixed(1)} ms`;
            progressBar.style.width = '100%';
            statusText.textContent = 'NeuralSpeed reconstruction complete';
            if (nextButton) nextButton.disabled = false;
        } else {
            statusText.textContent = 'NeuralSpeed final recon missing';
        }
    } catch (e) {
        if (isAbortError(e)) {
            resetNeuralStage('Reconstruction Stopped');
            return;
        }
        console.error(e);
        statusText.textContent = 'Network Error during NeuralSpeed reconstruction';
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

    const numSamples = parseInt(document.getElementById('num-samples-slider')?.value || '4');
    const langevinSteps = parseInt(document.getElementById('langevin-steps-slider')?.value || '100');

    lastStreamData = null;

    try {
        const jobId = buildGenerativeJobId();
        stageRunState['generative-ai-recon'].jobId = jobId;

        console.log(`DEBUG [Generative]: Starting combined sampling with JobID: ${jobId}, steps=${steps}, numSamples=${numSamples}, langevinSteps=${langevinSteps}`);
        statusText.textContent = `Starting GenerativeVision Diffusion (${steps} steps)...`;
        progressBar.style.width = '10%';

        const streamUrl = `/api/reconstruct/generative/${selectedDataset}/${pId}/${sIdx}/${numSources}?steps=${steps}&sigma_max=${sigmaMax}&sigma_min=${sigmaMin}&solver=${solver}&temperature=${temperature}&exposure_mas=${exposureMas}&ww=${windowWidth}&wl=${windowLevel}&job_id=${jobId}&num_samples=${numSamples}&langevin_steps=${langevinSteps}`;
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
        // 2. Start Streaming Iterative Recon
        statusText.textContent = `Starting HighFidelityMBIR (TV=${tv < 1e-5 ? "0" : tv.toFixed(5)})...`;
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
                            statusText.textContent = `HighFidelityMBIR: Iteration ${data.iteration} / ${data.total}`;
                        }
                    } catch (e) {
                        console.warn("Error parsing SSE chunk", e);
                    }
                }
            }
        }

        if (!signal.aborted) {
            statusText.textContent = "HighFidelityMBIR Complete";
            progressBar.style.width = '100%';
            if (nextButton) nextButton.disabled = false;
        }

    } catch (e) {
        if (isAbortError(e)) {
            resetIterativeStage('HighFidelityMBIR Stopped');
            return;
        }
        console.error(e);
        statusText.textContent = "HighFidelityMBIR Failed";
    } finally {
        finishStageRun('model-based-iterative-recon', controller);
        updateWorkflowUI();
    }
}
