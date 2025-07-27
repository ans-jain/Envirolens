let map = null;
let marker = null;

document.getElementById("uploadForm").addEventListener("submit", function (e) {
    e.preventDefault();
    
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        const { weather_condition, latitude, longitude, image_url } = data;

        // Show result
        const resultText = `Weather: ${weather_condition}`;
        document.getElementById("result").innerText = resultText;

        // Show uploaded image
        const uploadedImage = document.getElementById("uploadedImage");
        uploadedImage.src = image_url;
        uploadedImage.style.display = "block";

        // If valid coordinates exist, update the map
        if (latitude !== null && longitude !== null) {
            updateMap(latitude, longitude);
        } else {
            // Hide map if coordinates are missing
            if (map) {
                map.remove(); // fully remove existing map
                map = null;
            }
            document.getElementById("map").innerHTML = ""; // clear map div
        }
    })
    .catch(error => {
        console.error("Error:", error);
    });
});

function updateMap(lat, lon) {
    // Clear map if it already exists
    if (map) {
        map.remove();
    }

    map = L.map('map').setView([lat, lon], 13);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    marker = L.marker([lat, lon]).addTo(map)
        .bindPopup(`Latitude: ${lat}<br>Longitude: ${lon}`)
        .openPopup();
}
