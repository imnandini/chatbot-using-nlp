function sendMessage(message) {
    fetch('http://127.0.0.1:5000/chat', {
        // ... (rest of the code)
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.response);
        // Handle the chatbot's response
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
