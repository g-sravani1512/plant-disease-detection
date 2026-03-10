// ===========================
// Global Variables
// ===========================
let currentStream = null;
let capturedImageData = null;
let predictionResult = null;

// ===========================
// File Upload Handler
// ===========================
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        handleImageUpload(file);
    }
});

// ===========================
// Handle Image Upload
// ===========================
function handleImageUpload(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (PNG, JPG, JPEG)');
        return;
    }
    
    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        capturedImageData = e.target.result;
        showScanning(e.target.result);
        performPrediction(file);
    };
    reader.readAsDataURL(file);
}

// ===========================
// Camera Functions
// ===========================
function openCamera() {
    const modal = document.getElementById('cameraModal');
    const video = document.getElementById('cameraVideo');
    
    modal.classList.add('active');
    
    // Access camera
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
        } 
    })
    .then(stream => {
        currentStream = stream;
        video.srcObject = stream;
    })
    .catch(err => {
        console.error('Camera access error:', err);
        alert('Unable to access camera. Please check permissions.');
        closeCamera();
    });
}

function closeCamera() {
    const modal = document.getElementById('cameraModal');
    const video = document.getElementById('cameraVideo');
    
    // Stop camera stream
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    
    video.srcObject = null;
    modal.classList.remove('active');
}

function captureImage() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to data URL
    capturedImageData = canvas.toDataURL('image/jpeg', 0.95);
    
    // Close camera
    closeCamera();
    
    // Show scanning animation
    showScanning(capturedImageData);
    
    // Perform prediction
    performPredictionFromCamera(capturedImageData);
}

// ===========================
// Show Scanning Animation
// ===========================
function showScanning(imageSrc) {
    // Hide upload section
    const uploadSection = document.querySelector('.upload-container');
    if (uploadSection) {
        uploadSection.style.display = 'none';
    }
    
    // Show preview container
    const previewContainer = document.getElementById('imagePreviewContainer');
    previewContainer.style.display = 'block';
    
    // Set preview image
    const previewImage = document.getElementById('previewImage');
    previewImage.src = imageSrc;
    
    // Scroll to preview
    previewContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ===========================
// Perform Prediction (File Upload)
// ===========================
function performPrediction(file) {
    const formData = new FormData();
    formData.append('image', file);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            predictionResult = data;
            // Simulate scanning duration
            setTimeout(() => {
                displayResults(data);
            }, 3000); // 3 seconds scanning animation
        } else {
            alert('Error: ' + data.error);
            resetScan();
        }
    })
    .catch(error => {
        console.error('Prediction error:', error);
        alert('Failed to get prediction. Please try again.');
        resetScan();
    });
}

// ===========================
// Perform Prediction (Camera)
// ===========================
function performPredictionFromCamera(imageData) {
    const formData = new FormData();
    formData.append('image_data', imageData);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            predictionResult = data;
            // Simulate scanning duration
            setTimeout(() => {
                displayResults(data);
            }, 3000);
        } else {
            alert('Error: ' + data.error);
            resetScan();
        }
    })
    .catch(error => {
        console.error('Prediction error:', error);
        alert('Failed to get prediction. Please try again.');
        resetScan();
    });
}

// ===========================
// Display Results
// ===========================
function displayResults(data) {
    // Hide scanning
    const previewContainer = document.getElementById('imagePreviewContainer');
    previewContainer.style.display = 'none';
    
    // Show results container
    const resultsContainer = document.getElementById('resultsContainer');
    resultsContainer.style.display = 'block';
    
    // Set disease name
    document.getElementById('diseaseName').textContent = data.disease;
    
    // Set confidence
    const confidence = Math.round(data.confidence);
    document.getElementById('confidenceValue').textContent = confidence;
    
    // Update donut chart
    updateDonutChart(confidence);
    
    // Set description
    document.getElementById('diseaseDescription').textContent = data.description;
    
    // Set symptoms
    const symptomsList = document.getElementById('diseaseSymptoms');
    symptomsList.innerHTML = '';
    data.symptoms.forEach(symptom => {
        const li = document.createElement('li');
        li.textContent = symptom;
        symptomsList.appendChild(li);
    });
    
    // Set remedies
    const remediesList = document.getElementById('diseaseRemedies');
    remediesList.innerHTML = '';
    data.remedies.forEach(remedy => {
        const li = document.createElement('li');
        li.textContent = remedy;
        remediesList.appendChild(li);
    });
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ===========================
// Update Donut Chart
// ===========================
function updateDonutChart(confidence) {
    const circle = document.getElementById('donutProgress');
    const radius = 80;
    const circumference = 2 * Math.PI * radius;
    
    // Calculate offset based on confidence
    const offset = circumference - (confidence / 100) * circumference;
    
    // Set stroke color based on confidence level
    let strokeColor = '#2ecc71'; // Green
    if (confidence < 70) {
        strokeColor = '#f39c12'; // Orange
    }
    if (confidence < 40) {
        strokeColor = '#e74c3c'; // Red
    }
    
    circle.style.strokeDashoffset = offset;
    circle.style.stroke = strokeColor;
}

// ===========================
// Download PDF Report
// ===========================
function downloadPDF() {
    if (!predictionResult || !capturedImageData) {
        alert('No prediction data available');
        return;
    }
    
    // Prepare data for PDF generation
    const pdfData = {
        image: capturedImageData,
        disease: predictionResult.disease,
        confidence: predictionResult.confidence,
        description: predictionResult.description,
        symptoms: predictionResult.symptoms,
        remedies: predictionResult.remedies
    };
    
    // Show loading state
    const downloadBtn = event.target;
    const originalText = downloadBtn.innerHTML;
    downloadBtn.innerHTML = '⏳ Generating PDF...';
    downloadBtn.disabled = true;
    
    fetch('/generate-pdf', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(pdfData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('PDF generation failed');
        }
        return response.blob();
    })
    .then(blob => {
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'plant_disease_report.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        // Reset button
        downloadBtn.innerHTML = originalText;
        downloadBtn.disabled = false;
    })
    .catch(error => {
        console.error('PDF download error:', error);
        alert('Failed to generate PDF. Please try again.');
        downloadBtn.innerHTML = originalText;
        downloadBtn.disabled = false;
    });
}

// ===========================
// Reset Scan
// ===========================
function resetScan() {
    // Hide preview and results
    document.getElementById('imagePreviewContainer').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'none';
    
    // Show upload section
    const uploadSection = document.querySelector('.upload-container');
    if (uploadSection) {
        uploadSection.style.display = 'grid';
    }
    
    // Reset file input
    document.getElementById('fileInput').value = '';
    
    // Clear data
    capturedImageData = null;
    predictionResult = null;
    
    // Scroll to upload section
    document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
}

// ===========================
// Smooth Scroll for Navigation
// ===========================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        if (targetId === '#') return;
        
        const targetElement = document.querySelector(targetId);
        if (targetElement) {
            targetElement.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ===========================
// Active Navigation Link
// ===========================
window.addEventListener('scroll', function() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (pageYOffset >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// ===========================
// Close Modal on Outside Click
// ===========================
document.getElementById('cameraModal').addEventListener('click', function(e) {
    if (e.target === this) {
        closeCamera();
    }
});

// ===========================
// Prevent Form Submission on Enter
// ===========================
document.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.target.tagName !== 'TEXTAREA') {
        e.preventDefault();
    }
});

// ===========================
// Initialize on Page Load
// ===========================
document.addEventListener('DOMContentLoaded', function() {
    console.log('Plant Disease Detection System Initialized');
    
    // Reset donut chart
    const circle = document.getElementById('donutProgress');
    if (circle) {
        const circumference = 2 * Math.PI * 80;
        circle.style.strokeDasharray = circumference;
        circle.style.strokeDashoffset = circumference;
    }
});
