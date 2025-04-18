document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');
    const queryInput = document.getElementById('queryInput');
    const queryMode = document.getElementById('queryMode');
    const queryButton = document.getElementById('queryButton');
    const answer = document.getElementById('answer');
    const answerContent = answer.querySelector('.answer-content');
    const loadingSpinner = answer.querySelector('.loading-spinner');
    const sources = document.getElementById('sources');
    const querySection = document.querySelector('.query-section');

    let files = [];
    let hasUploadedFiles = false;

    // Initialize UI state
    updateUIState();

    // File upload handling
    dropzone.addEventListener('click', () => fileInput.click());
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--primary-color)';
        dropzone.style.background = 'rgba(79, 70, 229, 0.05)';
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = 'var(--border-color)';
        dropzone.style.background = 'var(--background-color)';
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--border-color)';
        dropzone.style.background = 'var(--background-color)';
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function updateUIState() {
        if (hasUploadedFiles) {
            querySection.style.opacity = '1';
            querySection.style.pointerEvents = 'auto';
            queryInput.disabled = false;
            queryButton.disabled = false;
        } else {
            querySection.style.opacity = '0.5';
            querySection.style.pointerEvents = 'none';
            queryInput.disabled = true;
            queryButton.disabled = true;
        }
    }

    function handleFiles(files) {
        const formData = new FormData();
        let validFiles = 0;
        
        Array.from(files).forEach(file => {
            if (file.size > 10 * 1024 * 1024) { // 10MB limit
                showMessage(`File ${file.name} exceeds 10MB limit`, 'error');
                return;
            }
            formData.append('files', file);
            validFiles++;
        });

        if (validFiles === 0) return;

        // Clear previous file list
        fileList.innerHTML = '';
        
        // Show selected files
        Array.from(files).forEach(file => {
            if (file.size <= 10 * 1024 * 1024) {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <div class="file-info">
                        <i class="fas fa-file"></i>
                        <span class="file-name">${file.name}</span>
                        <span class="file-size">(${formatFileSize(file.size)})</span>
                    </div>
                    <div class="file-status">
                        <i class="fas fa-spinner fa-spin"></i>
                    </div>
                `;
                fileList.appendChild(fileItem);
            }
        });

        // Upload files
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Upload failed');
            }
            return response.json();
        })
        .then(data => {
            // Update file status to success
            const fileItems = fileList.querySelectorAll('.file-item');
            fileItems.forEach(item => {
                const statusIcon = item.querySelector('.file-status i');
                statusIcon.className = 'fas fa-check-circle';
                statusIcon.style.color = 'var(--success-color)';
            });
            
            hasUploadedFiles = true;
            updateUIState();
            showMessage('Files uploaded and processed successfully!', 'success');
            // Clear the file input
            fileInput.value = '';
        })
        .catch(error => {
            // Update file status to error
            const fileItems = fileList.querySelectorAll('.file-item');
            fileItems.forEach(item => {
                const statusIcon = item.querySelector('.file-status i');
                statusIcon.className = 'fas fa-times-circle';
                statusIcon.style.color = 'var(--error-color)';
            });
            
            showMessage('Error uploading files: ' + error.message, 'error');
        });
    }

    // Query handling
    queryButton.addEventListener('click', async () => {
        const question = queryInput.value.trim();
        if (!question) {
            showMessage('Please enter a question', 'error');
            return;
        }

        if (!hasUploadedFiles) {
            showMessage('Please upload files before asking questions', 'error');
            return;
        }

        // Show loading state
        loadingSpinner.style.display = 'flex';
        answerContent.style.display = 'none';
        sources.innerHTML = '';

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    mode: queryMode.value
                })
            });

            if (!response.ok) {
                throw new Error('Query failed');
            }

            const data = await response.json();
            
            // Hide loading state
            loadingSpinner.style.display = 'none';
            answerContent.style.display = 'block';
            
            // Display answer with animation
            answerContent.innerHTML = `
                <h3>Answer</h3>
                <div class="answer-text">${data.answer}</div>
            `;
            answerContent.style.opacity = '0';
            answerContent.style.transform = 'translateY(20px)';
            setTimeout(() => {
                answerContent.style.opacity = '1';
                answerContent.style.transform = 'translateY(0)';
            }, 100);
            
            // Display sources if available
            if (data.sources && data.sources.length > 0) {
                sources.innerHTML = `
                    <h3>Sources</h3>
                    <ul class="sources-list">
                        ${data.sources.map(source => `
                            <li class="source-item">
                                <i class="fas fa-file-alt"></i>
                                <span>${source}</span>
                            </li>
                        `).join('')}
                    </ul>
                `;
                sources.style.opacity = '0';
                sources.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    sources.style.opacity = '1';
                    sources.style.transform = 'translateY(0)';
                }, 200);
            }

            // Display references if available
            if (data.references && data.references.length > 0) {
                const referencesDiv = document.createElement('div');
                referencesDiv.className = 'references-box';
                referencesDiv.innerHTML = `
                    <h3>References</h3>
                    <ul class="references-list">
                        ${data.references.map(ref => `
                            <li class="reference-item">
                                <i class="fas fa-quote-left"></i>
                                <p>${ref}</p>
                            </li>
                        `).join('')}
                    </ul>
                `;
                sources.appendChild(referencesDiv);
                referencesDiv.style.opacity = '0';
                referencesDiv.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    referencesDiv.style.opacity = '1';
                    referencesDiv.style.transform = 'translateY(0)';
                }, 300);
            }
        } catch (error) {
            loadingSpinner.style.display = 'none';
            answerContent.style.display = 'block';
            answerContent.innerHTML = `
                <h3>Error</h3>
                <div class="error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>${error.message}</p>
                </div>
            `;
            showMessage('Error processing query: ' + error.message, 'error');
        }
    });

    // Utility functions
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function showMessage(message, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
            <span>${message}</span>
        `;
        
        // Remove any existing messages
        const existingMessage = document.querySelector('.message');
        if (existingMessage) {
            existingMessage.remove();
        }
        
        document.querySelector('.container').insertBefore(messageDiv, document.querySelector('.upload-section'));
        
        // Add animation
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(-20px)';
        setTimeout(() => {
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        }, 100);
        
        // Remove message after 5 seconds
        setTimeout(() => {
            messageDiv.style.opacity = '0';
            messageDiv.style.transform = 'translateY(-20px)';
            setTimeout(() => messageDiv.remove(), 300);
        }, 5000);
    }
}); 