const statusDot = document.getElementById('status-dot');
const statusLabel = document.getElementById('status-label');
const captionText = document.getElementById('caption-text');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceValue = document.getElementById('confidence-value');
const lockedSentence = document.getElementById('locked-sentence');
const tokenCount = document.getElementById('token-count');
const clearSentenceButton = document.getElementById('clear-sentence');
const muteTtsButton = document.getElementById('mute-tts');
const historyEmpty = document.getElementById('history-empty');
const historyList = document.getElementById('history-list');
const sessionCountdownFill = document.getElementById('session-countdown-fill');

let tokenCursor = 0;
let lockedPhrases = [];
let ttsMuted = false;
let polling = true;
let displayHoldUntil = 0;
let sessionResetTimer = null;
let sessionCountdownTimer = null;
const DISPLAY_HOLD_MS = 3000;
const SESSION_INACTIVITY_MS = 5000;

function setStatus(status) {
    const mode = status.toLowerCase().includes('locked') ? 'confident' : status.toLowerCase().includes('error') ? 'low' : 'watching';
    statusDot.className = `status-dot is-${mode}`;
    statusLabel.textContent = status;
}

function setPrediction(text, confidence) {
    if (Date.now() < displayHoldUntil) return;
    captionText.textContent = text || 'Watching for your next phrase...';
    const percentage = Math.max(0, Math.min(1, confidence || 0)) * 100;
    confidenceFill.style.width = `${percentage}%`;
    confidenceValue.textContent = `${percentage.toFixed(1)}%`;
}

function showLockedPrediction(text, confidence) {
    displayHoldUntil = Date.now() + DISPLAY_HOLD_MS;
    captionText.textContent = text;
    captionText.className = 'caption-text is-confident';
    const percentage = Math.max(0, Math.min(1, confidence || 0)) * 100;
    confidenceFill.style.width = `${percentage}%`;
    confidenceFill.className = 'confidence-fill is-confident';
    confidenceValue.textContent = `${percentage.toFixed(1)}%`;
    window.setTimeout(() => {
        if (Date.now() >= displayHoldUntil) {
            captionText.textContent = 'Waiting for next gesture...';
            captionText.className = 'caption-text';
        }
    }, DISPLAY_HOLD_MS + 20);
}

function addHistoryItem(phrase, confidence) {
    if (!historyList) return;
    if (historyEmpty) historyEmpty.hidden = true;
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = '<span></span><strong></strong><span></span>';
    item.children[0].textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    item.children[1].textContent = phrase;
    item.children[2].textContent = `${(confidence * 100).toFixed(1)}%`;
    historyList.prepend(item);
}

function speakPhrase(phrase) {
    if (ttsMuted || !phrase || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(phrase);
    const indianVoice = window.speechSynthesis.getVoices().find(voice => voice.lang.toLowerCase() === 'en-in');
    if (indianVoice) utterance.voice = indianVoice;
    utterance.lang = indianVoice ? 'en-IN' : 'en-US';
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
}

function renderLockedSentence() {
    lockedSentence.textContent = lockedPhrases.length ? lockedPhrases.join(' ') : 'Waiting for gesture...';
    tokenCount.textContent = `${lockedPhrases.length} phrase${lockedPhrases.length === 1 ? '' : 's'}`;
}

function startSessionInactivityTimer() {
    window.clearTimeout(sessionResetTimer);
    window.clearInterval(sessionCountdownTimer);
    const deadline = Date.now() + SESSION_INACTIVITY_MS;
    sessionCountdownTimer = window.setInterval(() => {
        const remaining = Math.max(0, deadline - Date.now());
        if (sessionCountdownFill) sessionCountdownFill.style.width = `${(remaining / SESSION_INACTIVITY_MS) * 100}%`;
    }, 80);
    sessionResetTimer = window.setTimeout(resetSentenceSession, SESSION_INACTIVITY_MS);
}

async function resetSentenceSession() {
    window.clearInterval(sessionCountdownTimer);
    sessionCountdownTimer = null;
    if (historyList) historyList.classList.add('is-clearing');
    await fetch('/reset_sentence', { method: 'POST' }).catch(() => {});
    window.setTimeout(() => {
        lockedPhrases = [];
        tokenCursor = 0;
        if (historyList) {
            historyList.replaceChildren();
            historyList.classList.remove('is-clearing');
        }
        if (historyEmpty) historyEmpty.hidden = false;
        renderLockedSentence();
        if (sessionCountdownFill) sessionCountdownFill.style.width = '0%';
    }, 260);
}

function handleNewTokens(tokens) {
    tokens.forEach(token => {
        if (!token || !token.text) return;
        lockedPhrases.push(token.text);
        addHistoryItem(token.text, token.confidence || 0);
        speakPhrase(token.text);
        showLockedPrediction(token.text, token.confidence || 0);
        startSessionInactivityTimer();
    });
    renderLockedSentence();
}

async function pollPrediction() {
    if (!polling) return;
    try {
        const response = await fetch(`/get_prediction?cursor=${tokenCursor}`, { cache: 'no-store' });
        const data = await response.json();
        if (!data.success) throw new Error(data.message || 'Prediction unavailable');
        setStatus(data.status || 'Idle');
        setPrediction(data.prediction, data.confidence);
        handleNewTokens(data.locked_tokens || []);
        tokenCursor = data.token_count || tokenCursor;
    } catch (error) {
        setStatus('Connection issue');
        setPrediction(error.message, 0);
    } finally {
        if (polling) window.setTimeout(pollPrediction, 180);
    }
}

clearSentenceButton?.addEventListener('click', async () => {
    await resetSentenceSession();
    window.speechSynthesis?.cancel();
});

muteTtsButton?.addEventListener('click', () => {
    ttsMuted = !ttsMuted;
    muteTtsButton.textContent = ttsMuted ? 'Unmute TTS' : 'Mute TTS';
    muteTtsButton.setAttribute('aria-pressed', String(ttsMuted));
    if (ttsMuted) window.speechSynthesis?.cancel();
});

renderLockedSentence();
pollPrediction();
