// ─────────────────────────────────────────────────────────────────────────────
// AI Mock Interviewer — Frontend Logic
// Camera proctoring runs entirely in the BROWSER via getUserMedia + face-api.js
// No camera frames are ever sent to the server.
// ─────────────────────────────────────────────────────────────────────────────

let timerInterval;
const MAX_TIME = 90;
let timeLeft = MAX_TIME;

// DOM — screens
const uploadScreen   = document.getElementById('upload-screen');
const interviewScreen = document.getElementById('interview-screen');
const reportScreen   = document.getElementById('report-screen');

// DOM — upload
const uploadForm      = document.getElementById('upload-form');
const resumeUpload    = document.getElementById('resume-upload');
const fileNameDisplay = document.getElementById('file-name');
const startBtn        = document.getElementById('start-btn');
const loader          = document.getElementById('upload-loader');

// DOM — interview
const timeDisplay     = document.getElementById('time-left');
const topicsTimeline  = document.getElementById('topics-timeline');
const currentTopicTag = document.getElementById('current-topic-tag');
const questionText    = document.getElementById('question-text');
const answerBox       = document.getElementById('answer-box');
const submitBtn       = document.getElementById('submit-btn');
const skipBtn         = document.getElementById('skip-btn');
const quitBtn         = document.getElementById('quit-btn');
const feedbackDisplay = document.getElementById('feedback-display');

// DOM — proctor
const proctorDot         = document.getElementById('proctor-dot');
const proctorLabel       = document.getElementById('proctor-label');
const proctorVideoImg    = document.getElementById('proctor-video-stream');
const violationCount     = document.getElementById('violation-count');
const gazeStatus         = document.getElementById('gaze-status');
const violBanner         = document.getElementById('violation-banner');

// ─────────────────────────────────────────────────────────────────────────────
// SERVER-SIDE AI PROCTOR  (polling and stream controller)
// ─────────────────────────────────────────────────────────────────────────────

let proctorStatusInterval = null;

async function startProctor() {
    // Start streaming the annotated server webcam feed
    proctorVideoImg.src = '/api/proctor/video_feed';
    
    proctorDot.className     = 'proctor-dot active';
    proctorLabel.textContent = 'AI Monitor Active';
    
    // Poll the status every 1000ms
    startProctorStatusPolling();
}

function stopProctor() {
    stopProctorStatusPolling();
}

function startProctorStatusPolling() {
    if (proctorStatusInterval) clearInterval(proctorStatusInterval);
    
    proctorStatusInterval = setInterval(async () => {
        try {
            const res = await fetch('/api/proctor/status');
            const data = await res.json();
            
            if (data.active) {
                violationCount.textContent = data.violations;
                
                if (data.last_violation) {
                    violBanner.textContent = `⚠️ ${data.last_violation}`;
                    violBanner.classList.remove('hidden');
                    gazeStatus.textContent = '👁 Warning';
                    gazeStatus.className = 'gaze-warn';
                    
                    if (data.violations >= data.limit) {
                        violBanner.textContent = `⛔ Limit exceeded (${data.violations}/${data.limit}). Flagged.`;
                        violBanner.classList.add('critical');
                    } else {
                        violBanner.classList.remove('critical');
                    }
                } else {
                    violBanner.classList.add('hidden');
                    gazeStatus.textContent = '👁 On Screen';
                    gazeStatus.className = 'gaze-ok';
                }
            }
        } catch (err) {
            console.warn('[Proctor] Failed to poll status:', err);
        }
    }, 1000);
}

function stopProctorStatusPolling() {
    if (proctorStatusInterval) {
        clearInterval(proctorStatusInterval);
        proctorStatusInterval = null;
    }
    proctorVideoImg.src = '';
    proctorDot.className = 'proctor-dot';
    proctorLabel.textContent = 'Camera Off';
}



// ─────────────────────────────────────────────────────────────────────────────
// FILE INPUT
// ─────────────────────────────────────────────────────────────────────────────

resumeUpload.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        fileNameDisplay.textContent = e.target.files[0].name;
        fileNameDisplay.style.color = 'var(--primary)';
    }
});

// ── Interactive Skill Badges Selection ────────────────────────────────────────
const selectedSkills = new Set();
const badgeOptions = document.querySelectorAll('.skill-badge-option');
const hiddenSkillsInput = document.getElementById('extra-skills');

badgeOptions.forEach(badge => {
    badge.addEventListener('click', () => {
        const skill = badge.getAttribute('data-skill');
        if (selectedSkills.has(skill)) {
            selectedSkills.delete(skill);
            badge.classList.remove('selected');
        } else {
            selectedSkills.add(skill);
            badge.classList.add('selected');
        }
        // Update the hidden input value as a comma-separated list
        hiddenSkillsInput.value = Array.from(selectedSkills).join(', ');
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// UPLOAD & SESSION START
// ─────────────────────────────────────────────────────────────────────────────

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = resumeUpload.files[0];
    if (!file) return;

    const extraSkills = document.getElementById('extra-skills').value;

    startBtn.classList.add('hidden');
    loader.classList.remove('hidden');

    const formData = new FormData();
    formData.append('resume', file);
    formData.append('skills', extraSkills);

    try {
        const res  = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            switchToInterview();
            updateTimeline(data.topics_to_cover, [], data.topics_to_cover[0]);
            fetchNextQuestion();
        } else {
            alert(data.error);
        }
    } catch (err) {
        console.error(err);
        alert('Server connection failed.');
    } finally {
        startBtn.classList.remove('hidden');
        loader.classList.add('hidden');
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// SCREEN TRANSITIONS
// ─────────────────────────────────────────────────────────────────────────────

function switchToInterview() {
    uploadScreen.classList.remove('active');
    setTimeout(() => {
        uploadScreen.classList.add('hidden');
        interviewScreen.classList.remove('hidden');
        void interviewScreen.offsetWidth;
        interviewScreen.classList.add('active');
    }, 500);

    // Start browser-side proctor (not the server-side one)
    startProctor();
}

function parseMarkdownToHtml(markdown) {
    let escaped = markdown
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    escaped = escaped.replace(/^#\s+(.+)$/gm, '<h1 class="report-h1">$1</h1>');
    escaped = escaped.replace(/^##\s+(.+)$/gm, '<h2 class="report-h2">$1</h2>');
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    escaped = escaped.replace(/^\s*-\s+(.+)$/gm, '<li class="report-li">$1</li>');
    escaped = escaped.replace(/^={15,}$/gm, '<hr class="report-hr-double">');
    escaped = escaped.replace(/^-{15,}$/gm, '<hr class="report-hr-single">');

    escaped = escaped.replace(/!\[(.*?)\]\((.*?)\)/g, 
        '<div class="violation-img-card">' +
        '  <div class="violation-img-header"><span class="infraction-badge">⚠️ INFRACTION EVIDENCE</span></div>' +
        '  <img src="$2" alt="$1" class="violation-screenshot" onclick="window.open(\'$2\', \'_blank\')">' +
        '  <div class="violation-img-footer">$1 (Click to enlarge)</div>' +
        '</div>'
    );

    escaped = escaped.replace(/(🛡️\s+EXAM INTEGRITY &amp; AI PROCTORING REPORT)/g, '<span class="integrity-title">$1</span>');
    escaped = escaped.replace(/(⚠️\s+Integrity Warning:.*? Session flagged for manual review\.)/g, '<div class="report-alert-critical">$1</div>');

    return escaped;
}

function switchToReport(reportText) {
    document.getElementById('report-text').innerHTML = parseMarkdownToHtml(reportText);
    clearInterval(timerInterval);
    stopProctor();

    interviewScreen.classList.remove('active');
    setTimeout(() => {
        interviewScreen.classList.add('hidden');
        reportScreen.classList.remove('hidden');
        void reportScreen.offsetWidth;
        reportScreen.classList.add('active');
        loadSessionHistory();
    }, 500);
}

// ─────────────────────────────────────────────────────────────────────────────
// INTERVIEW LOGIC
// ─────────────────────────────────────────────────────────────────────────────

async function fetchNextQuestion() {
    resetTimer();
    answerBox.disabled  = true;
    submitBtn.disabled  = true;
    skipBtn.disabled    = true;
    questionText.textContent = 'Analyzing context and retrieving next question...';

    // Refresh topic timeline
    try {
        const stateRes = await fetch('/api/state');
        const state    = await stateRes.json();
        updateTimeline(state.topics_to_cover, state.topics_covered, state.current_topic);
    } catch (e) {}

    try {
        const res  = await fetch('/api/next');
        const data = await res.json();

        if (data.finished || data.error) {
            endInterview();
            return;
        }

        currentTopicTag.textContent = data.topic.replace(/_/g, ' ').toUpperCase();
        questionText.textContent    = data.question;
        answerBox.value             = '';
        answerBox.disabled          = false;
        submitBtn.disabled          = false;
        skipBtn.disabled            = false;
        answerBox.focus();
        startTimer();

    } catch (err) {
        console.error(err);
        questionText.textContent = 'Error fetching question.';
    }
}

async function submitAnswer(isSkip = false) {
    clearInterval(timerInterval);
    answerBox.disabled  = true;
    submitBtn.disabled  = true;
    skipBtn.disabled    = true;

    const answer = answerBox.value.trim();
    if (!isSkip && !answer) {
        alert('Please enter an answer or skip.');
        answerBox.disabled  = false;
        submitBtn.disabled  = false;
        skipBtn.disabled    = false;
        return;
    }

    try {
        const res  = await fetch('/api/answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ answer, skip: isSkip })
        });
        const data = await res.json();

        feedbackDisplay.textContent = '';
        feedbackDisplay.className   = '';

        if (isSkip) {
            feedbackDisplay.textContent = 'Question Skipped. Penalty applied.';
            feedbackDisplay.classList.add('score-weak');
        } else {
            const score = data.score;
            if (score >= 0.7) {
                feedbackDisplay.textContent = `Excellent Answer! Score: ${(score * 10).toFixed(1)}/10`;
                feedbackDisplay.classList.add('score-strong');
            } else if (score >= 0.4) {
                feedbackDisplay.textContent = `Mediocre Answer. Score: ${(score * 10).toFixed(1)}/10`;
                feedbackDisplay.classList.add('score-mediocre');
            } else {
                feedbackDisplay.textContent = `Weak Answer. Score: ${(score * 10).toFixed(1)}/10`;
                feedbackDisplay.classList.add('score-weak');
            }
        }

        setTimeout(() => {
            feedbackDisplay.textContent = '';
            fetchNextQuestion();
        }, 2000);

    } catch (err) {
        console.error(err);
    }
}

async function endInterview() {
    clearInterval(timerInterval);
    questionText.textContent = 'Generating final report...';
    try {
        const res  = await fetch('/api/report');
        const data = await res.json();
        switchToReport(data.report_content);
    } catch (err) {
        console.error(err);
        switchToReport('Failed to generate report.');
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// UI HELPERS
// ─────────────────────────────────────────────────────────────────────────────

function updateTimeline(upcoming, done, current) {
    topicsTimeline.innerHTML = '';

    done.forEach(t => {
        const span = document.createElement('span');
        span.className   = 'topic-badge done';
        span.textContent = t.replace(/_/g, ' ');
        topicsTimeline.appendChild(span);
    });

    if (current) {
        const span = document.createElement('span');
        span.className   = 'topic-badge current';
        span.textContent = current.replace(/_/g, ' ');
        topicsTimeline.appendChild(span);
    }

    upcoming.forEach(t => {
        if (t === current || done.includes(t)) return;
        const span = document.createElement('span');
        span.className   = 'topic-badge';
        span.textContent = t.replace(/_/g, ' ');
        topicsTimeline.appendChild(span);
    });
}

function resetTimer() {
    clearInterval(timerInterval);
    timeLeft = MAX_TIME;
    timeDisplay.textContent = `${timeLeft}s`;
    timeDisplay.style.color = 'var(--warning)';
}

function startTimer() {
    timerInterval = setInterval(() => {
        timeLeft--;
        timeDisplay.textContent = `${timeLeft}s`;
        if (timeLeft <= 10) timeDisplay.style.color = 'var(--danger)';
        else                timeDisplay.style.color = 'var(--warning)';
        if (timeLeft <= 0) {
            clearInterval(timerInterval);
            submitAnswer(true); // timeout → auto skip
        }
    }, 1000);
}

// ─────────────────────────────────────────────────────────────────────────────
// BUTTON EVENTS
// ─────────────────────────────────────────────────────────────────────────────

submitBtn.addEventListener('click', () => submitAnswer(false));
skipBtn.addEventListener('click',   () => submitAnswer(true));
quitBtn.addEventListener('click',   () => endInterview());

answerBox.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') submitAnswer(false);
});

// ─────────────────────────────────────────────────────────────────────────────
// PAST SESSIONS
// ─────────────────────────────────────────────────────────────────────────────

async function loadSessionHistory() {
    try {
        const res  = await fetch('/api/history');
        const data = await res.json();
        const container = document.getElementById('sessions-table-body');
        if (!container || !data.sessions) return;
        container.innerHTML = '';
        data.sessions.forEach(s => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${s.timestamp ? s.timestamp.split('T')[0] : '—'}</td>
                <td>${s.candidate_name  || '—'}</td>
                <td>${s.readiness_level || '—'}</td>
                <td>${s.overall_score !== undefined ? (s.overall_score * 10).toFixed(1) + '/10' : '—'}</td>
            `;
            container.appendChild(tr);
        });
    } catch (e) {}
}

// ── Auto-resume active session on page load/refresh ──────────────────────────
async function checkActiveSession() {
    try {
        const res = await fetch('/api/state');
        if (res.ok) {
            const state = await res.json();
            // Switch screen to interview
            uploadScreen.classList.add('hidden');
            interviewScreen.classList.remove('hidden');
            void interviewScreen.offsetWidth;
            interviewScreen.classList.add('active');
            
            // Re-establish webcam stream and status polling loop
            startProctor();
            
            // Resume fetching questions
            fetchNextQuestion();
        } else {
            loadSessionHistory();
        }
    } catch (e) {
        loadSessionHistory();
    }
}

checkActiveSession();
