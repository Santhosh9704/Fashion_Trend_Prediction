class FashionStylePredictor {
    constructor() {
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.previewSection = document.getElementById('previewSection');
        this.previewImage = document.getElementById('previewImage');
        this.predictBtn = document.getElementById('predictBtn');
        this.resultsSection = document.getElementById('resultsSection');
        this.loading = document.getElementById('loading');
        this.predictedStyle = document.getElementById('predictedStyle');
        this.confidenceBar = document.getElementById('confidenceBar');
        this.confidenceText = document.getElementById('confidenceText');
    }

    bindEvents() {
        // Upload area events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        // File input change
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Predict button
        this.predictBtn.addEventListener('click', this.predictStyle.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB.');
            return;
        }

        this.currentFile = file;
        this.displayPreview(file);
    }

    displayPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.previewSection.style.display = 'block';
            this.resultsSection.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    async predictStyle() {
        if (!this.currentFile) return;

        this.showLoading(true);
        this.predictBtn.disabled = true;

        try {
            // Simulate API call for demo purposes
            // In a real implementation, you would send the image to your Python backend
            await this.simulateStylePrediction();
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Error predicting style. Please try again.');
        } finally {
            this.showLoading(false);
            this.predictBtn.disabled = false;
        }
    }

    async simulateStylePrediction() {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Mock prediction results
        const styles = ['casual', 'formal', 'streetwear', 'bohemian', 'vintage', 'minimalist'];
        const randomStyle = styles[Math.floor(Math.random() * styles.length)];
        const confidence = Math.floor(Math.random() * 30) + 70; // 70-99%

        this.displayResults(randomStyle, confidence);
    }

    displayResults(style, confidence) {
        this.predictedStyle.textContent = style;
        this.confidenceBar.style.width = `${confidence}%`;
        this.confidenceText.textContent = `${confidence}%`;
        
        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    showLoading(show) {
        this.loading.style.display = show ? 'block' : 'none';
        if (show) {
            this.resultsSection.style.display = 'none';
        }
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new FashionStylePredictor();
});