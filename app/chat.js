const chatHistory = document.getElementById('chat-history');
const chatTextarea = document.getElementById('chat-textarea');
const chatSendButton = document.getElementById('chat-send-button');

const socket = new WebSocket('ws://localhost:8000/ws');

// React to bot responses by putting them in the chat history section
socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    addBotResponseToChatHistory(message.sentence, message.relevant, message.score);
    scrollToBottom();
}

// Send message when send button is clicked
chatSendButton.addEventListener('click', sendMessage);

// Send message when Enter key is pressed
chatTextarea.addEventListener('keydown', event => {
    if (event.keyCode === 13 && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

// Send message when textarea reaches a certain number of characters
chatTextarea.addEventListener("input", function () {
    let words = chatTextarea.value.split(/\s+/);
    if (words.length > 30) {
        sendMessage();
    }
});

function sendMessage() {
    const message = chatTextarea.value.trim();
    chatTextarea.value = '';
    if (message) {
        socket.send(message);
    }
}

function addBotResponseToChatHistory(message, relevant, score) {
    // Create bot answer message bubble
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', relevant ? 'relevant' : 'irrelevant');
    messageElement.appendChild(createBotTextAnswer(message, score));


    // messageElement.appendChild(createFeedbackButtons(messageElement));



    // ... and append to chat history
    chatHistory.appendChild(messageElement);
}

function createBotTextAnswer(message, score) {
    const messageText = document.createElement('div');
    messageText.classList.add('chat-message-text');
    messageText.innerText = message;
    let scoreText = document.createElement('span');
    scoreText.classList.add('score');
    // Round a number to integer
    scoreText.innerText = 'Score: ' + Math.round(score);
    messageText.appendChild(scoreText)
    scoreText.appendChild(createFeedbackButton(null, 'positive'));
    scoreText.appendChild(createFeedbackButton(null, 'negative'));
    return messageText;
}

function createFeedbackButtons(messageElement) {
    const messageFeedback = document.createElement('div');
    messageFeedback.classList.add('chat-message-feedback');
    messageFeedback.appendChild(createFeedbackButton(messageElement, 'positive'));
    messageFeedback.appendChild(createFeedbackButton(messageElement, 'negative'));
    return messageFeedback;
}

function createFeedbackButton(messageElement, type) {
    const feedbackPositiveButton = document.createElement('button');
    feedbackPositiveButton.classList.add('chat-message-feedback-button', 'feedback-' + type);
    const icon = document.createElement('i');
    icon.classList.add('fa', type === 'positive' ? 'fa-thumbs-up' : 'fa-thumbs-down');
    feedbackPositiveButton.appendChild(icon);
    feedbackPositiveButton.addEventListener('click', () => sendFeedback(messageElement, type));
    return feedbackPositiveButton;
}

function scrollToBottom() {
    window.scrollTo(0, document.body.scrollHeight);
}


