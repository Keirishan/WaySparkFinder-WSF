function openCamera() {
    window.location.href = "https://sltc.ac.lk/"; // Replace with the actual URL for accessing the camera
}

function trackLocation() {
    window.location.href = "https://www.google.com/maps"; // Replace with the actual URL for tracking location
}

function sendMessage() {
    window.location.href = "sendmessage.html"; // Replace with the actual URL for sending messages
}

function signOut() {
    window.location.href = "login.html"; // Redirect to the login page
}
function backtodashboard() {
    window.location.href = "dashboard.html"; // Redirect to the login page
}
function checkLogin() {
    var username = document.getElementById("username").value;
    var password = document.getElementById("password").value;

    // Check the credentials (you can replace this with your authentication logic)
    if (username === "VM1" && password === "123456") {
        // Redirect to the next page or perform any other action
        window.location.href = "index.html";
    } else {
        alert("Invalid credentials. Please try again.");
    }
}