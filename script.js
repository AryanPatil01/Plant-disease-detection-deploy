document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.querySelector('.browse-btn');
    
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const predictBtn = document.getElementById('predict-btn');
    
    const loadingState = document.getElementById('loading-state');
    const resultSection = document.getElementById('result-section');
    const resetBtn = document.getElementById('reset-btn');

    const plantNameInput = document.getElementById('plant-name');
    
    const diseaseName = document.getElementById('disease-name');
    const confidenceScore = document.getElementById('confidence-score');
    const progressBar = document.getElementById('progress-bar');
    
    let currentFile = null;

    // --- Event Listeners for Drag and Drop ---
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // --- Click to Upload ---
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // --- Remove Image ---
    removeBtn.addEventListener('click', () => {
        resetUI();
    });

    // --- Handle File ---
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultSection.classList.add('hidden');
        };
        
        reader.readAsDataURL(file);
    }

    // --- Predict Button ---
    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Update UI to loading state
        previewContainer.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            const formData = new FormData();
            formData.append('image', currentFile);
            formData.append('plant', plantNameInput.value.trim());

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`);
            }

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            showResult(data.prediction, data.confidence);

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to predict. Is the backend running?');
            resetUI();
        }
    });

    // --- Show Result ---
    function showResult(prediction, confidence) {
        loadingState.classList.add('hidden');
        resultSection.classList.remove('hidden');
        
        diseaseName.textContent = prediction;
        
        const percent = (confidence * 100).toFixed(1);
        confidenceScore.textContent = `${percent}% Match`;
        
        // Animate progress bar
        setTimeout(() => {
            progressBar.style.width = `${percent}%`;
            
            // Color based on confidence
            if (percent > 90) {
                progressBar.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
            } else if (percent > 70) {
                progressBar.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
            } else {
                progressBar.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
            }
        }, 100);
    }

    // --- Reset UI ---
    resetBtn.addEventListener('click', resetUI);

    function resetUI() {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        progressBar.style.width = '0%';
        
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        loadingState.classList.add('hidden');
        resultSection.classList.add('hidden');
    }
});
