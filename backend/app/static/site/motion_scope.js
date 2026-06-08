// Motion Scope Precomputed Player (Case 7 Only, Instant client-side render)
const state = {
    time: 0,            // 0 - 980 in steps of 20
    views: '80',        // '80' or '240'
    clock: 'no',        // 'no' or 'yes'
    method: 'flashfbp', // 'flashfbp' or 'neuralspark'
    isPlaying: true,    // play state
    lastTick: 0
};

const elements = {
    patientPreview: document.getElementById('motion-patient-preview'),
    recon4: document.getElementById('motion-recon-4rps'),
    recon20: document.getElementById('motion-recon-20rps'),
    recon80: document.getElementById('motion-recon-80rps'),

    timeSlider: document.getElementById('motion-time-slider'),
    timeLabel: document.getElementById('motion-time-label'),
    playPauseBtn: document.getElementById('motion-play-pause-btn'),

    btnFlashfbp: document.getElementById('motion-btn-flashfbp'),
    btnNeuralspark: document.getElementById('motion-btn-neuralspark'),
    generalStatus: document.getElementById('motion-general-status')
};

// Update the 4 frame displays instantly based on current states
function updateImages(t) {
    if (!elements.patientPreview || !elements.recon4 || !elements.recon20 || !elements.recon80) return;

    // 1. Ground Truth (Case 7 Patient State)
    const gtUrl = `/static/precomputed/motion_clean/gt/clock_${state.clock}/frame_${t}.png`;
    elements.patientPreview.src = gtUrl;

    // 2. 4 RPS Reconstruction
    const rot4Idx = Math.max(0, Math.min(3, Math.floor(t / 250)));
    const url4 = `/static/precomputed/motion_clean/views_${state.views}/clock_${state.clock}/method_${state.method}/rps_4/rot_${rot4Idx}.png`;
    elements.recon4.src = url4;

    // 3. 20 RPS Reconstruction
    const rot20Idx = Math.max(0, Math.min(19, Math.floor(t / 50)));
    const url20 = `/static/precomputed/motion_clean/views_${state.views}/clock_${state.clock}/method_${state.method}/rps_20/rot_${rot20Idx}.png`;
    elements.recon20.src = url20;

    // 4. 80 RPS Reconstruction
    const rot80Idx = Math.max(0, Math.min(79, Math.floor(t / 12.5)));
    const url80 = `/static/precomputed/motion_clean/views_${state.views}/clock_${state.clock}/method_${state.method}/rps_80/rot_${rot80Idx}.png`;
    elements.recon80.src = url80;

    // 5. Update slider and text labels
    if (elements.timeSlider) {
        elements.timeSlider.value = String(t);
    }
    if (elements.timeLabel) {
        elements.timeLabel.textContent = `${t} ms`;
    }
}

// Global Animation Play/Pause loop using requestAnimationFrame
function animate(now) {
    if (!state.lastTick) {
        state.lastTick = now;
    }
    const elapsed = now - state.lastTick;

    if (state.isPlaying) {
        // Smoothly increment time across the 1.0s cardiac cycle (1000ms)
        state.time = (state.time + elapsed) % 1000;
        
        // Quantize to closest step of 20 ms
        const stepTime = Math.floor(state.time / 20) * 20;
        updateImages(stepTime);
    }

    state.lastTick = now;
    requestAnimationFrame(animate);
}

// Toggle Play / Pause state
function togglePlayPause() {
    state.isPlaying = !state.isPlaying;
    if (elements.playPauseBtn) {
        if (state.isPlaying) {
            elements.playPauseBtn.textContent = 'Pause';
            elements.playPauseBtn.classList.remove('paused');
            if (elements.generalStatus) {
                elements.generalStatus.textContent = 'Playback online. Rendering precomputed physics matrix.';
            }
        } else {
            elements.playPauseBtn.textContent = 'Play';
            elements.playPauseBtn.classList.add('paused');
            if (elements.generalStatus) {
                elements.generalStatus.textContent = 'Playback paused. Ready to step.';
            }
        }
    }
}

// Set reconstructed method and toggles classes on buttons
function selectMethod(methodVal) {
    state.method = methodVal;
    if (elements.btnFlashfbp && elements.btnNeuralspark) {
        elements.btnFlashfbp.classList.toggle('active', methodVal === 'flashfbp');
        elements.btnNeuralspark.classList.toggle('active', methodVal === 'neuralspark');
    }
    const stepTime = Math.floor(state.time / 20) * 20;
    updateImages(stepTime);
}

// Bind all interactive listeners
function bindEvents() {
    // 1. Time Slider control
    if (elements.timeSlider) {
        elements.timeSlider.addEventListener('input', (e) => {
            // Dragging suspends auto-playback for clean frame scrubbing
            state.isPlaying = false;
            if (elements.playPauseBtn) {
                elements.playPauseBtn.textContent = 'Play';
                elements.playPauseBtn.classList.add('paused');
            }
            if (elements.generalStatus) {
                elements.generalStatus.textContent = 'Scrubbing clinical frames.';
            }
            state.time = Number(e.target.value);
            updateImages(state.time);
        });
    }

    // 2. Play/Pause Button
    if (elements.playPauseBtn) {
        elements.playPauseBtn.addEventListener('click', () => {
            togglePlayPause();
        });
    }

    // 3. Views Radio Selections
    document.querySelectorAll('input[name="motion-views-radio"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            state.views = e.target.value;
            const stepTime = Math.floor(state.time / 20) * 20;
            updateImages(stepTime);
        });
    });

    // 4. Clock Radio Selections
    document.querySelectorAll('input[name="motion-clock-radio"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            state.clock = e.target.value;
            const stepTime = Math.floor(state.time / 20) * 20;
            updateImages(stepTime);
        });
    });

    // 5. Square Reconstruction Algorithm Selection Buttons
    if (elements.btnFlashfbp) {
        elements.btnFlashfbp.addEventListener('click', () => {
            selectMethod('flashfbp');
        });
    }
    if (elements.btnNeuralspark) {
        elements.btnNeuralspark.addEventListener('click', () => {
            selectMethod('neuralspark');
        });
    }
}

// Initializer block
function init() {
    bindEvents();
    
    // Default to FlashFBP
    selectMethod('flashfbp');
    
    // Set initial view paths
    updateImages(0);
    
    // Start playback loop
    requestAnimationFrame(animate);
}

document.addEventListener('DOMContentLoaded', init);









