document.addEventListener('DOMContentLoaded', function() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const clearBtn = document.getElementById('clearBtn');
    const videoFeed = document.getElementById('videoFeed');
    const detectedWords = document.getElementById('detectedWords');
    const fullSentence = document.getElementById('fullSentence');
    const translation = document.getElementById('translation');
    const showBoxes = document.getElementById('showBoxes');
    const hospitalMode = document.getElementById('hospitalMode');
    const loadingOverlay = document.getElementById('videoLoading');

    let isDetectionRunning = false;
    let detectedWordsList = [];
    let videoFeedInterval = null;

    // Function to show loading state
    function showLoading() {
        loadingOverlay.style.display = 'flex';
        videoFeed.style.display = 'none';
    }

    // Function to hide loading state
    function hideLoading() {
        loadingOverlay.style.display = 'none';
        videoFeed.style.display = 'block';
    }

    // Function to update video feed
    function updateVideoFeed() {
        if (isDetectionRunning) {
            const timestamp = new Date().getTime();
            let videoUrl = '/video_feed';
            videoUrl += `?show_boxes=${showBoxes.checked}`;
            videoUrl += `&hospital_mode=${hospitalMode.checked}`;
            videoUrl += `&t=${timestamp}`;
            
            // Add error handling for video feed
            videoFeed.onerror = function() {
                console.error('Error loading video feed');
                const statusIndicator = videoFeed.parentElement.querySelector('.status-indicator');
                if (statusIndicator) {
                    statusIndicator.className = 'status-indicator inactive';
                    statusIndicator.innerHTML = '<i class="fas fa-exclamation-circle"></i> Camera Error';
                }
                
                // Show error message in loading overlay
                const loadingText = document.createElement('div');
                loadingText.className = 'loading-text';
                loadingText.textContent = 'Error loading video feed. Please check your camera connection.';
                loadingOverlay.appendChild(loadingText);
                
                // Try to reconnect after 2 seconds
                setTimeout(() => {
                    if (isDetectionRunning) {
                        updateVideoFeed();
                    }
                }, 2000);
            };
            
            videoFeed.onload = function() {
                const statusIndicator = videoFeed.parentElement.querySelector('.status-indicator');
                if (statusIndicator) {
                    statusIndicator.className = 'status-indicator active';
                    statusIndicator.innerHTML = '<i class="fas fa-circle"></i> Camera Active';
                }
                hideLoading();
            };
            
            videoFeed.src = videoUrl;
        }
    }

    // Function to update button states
    function updateButtonStates() {
        startBtn.disabled = isDetectionRunning;
        stopBtn.disabled = !isDetectionRunning;
        
        if (isDetectionRunning) {
            startBtn.classList.remove('btn-success');
            startBtn.classList.add('btn-secondary');
            stopBtn.classList.remove('btn-secondary');
            stopBtn.classList.add('btn-danger');
        } else {
            startBtn.classList.remove('btn-secondary');
            startBtn.classList.add('btn-success');
            stopBtn.classList.remove('btn-danger');
            stopBtn.classList.add('btn-secondary');
        }
    }

    // Start detection
    startBtn.addEventListener('click', async function() {
        if (!isDetectionRunning) {
            showLoading();
            startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Initializing Camera...';
            
            try {
                const response = await fetch('/start_detection', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        hospital_mode: hospitalMode.checked,
                        show_boxes: showBoxes.checked
                    })
                });

                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.message || 'Failed to start detection');
                }

                isDetectionRunning = true;
                updateButtonStates();
                
                // Show camera initialization message
                startBtn.innerHTML = '<i class="fas fa-camera"></i> Camera Initialized';
                setTimeout(() => {
                    startBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
                }, 2000);
                
                // Start video feed updates
                updateVideoFeed();
                videoFeedInterval = setInterval(updateVideoFeed, 1000);
                
                // Add status indicator
                const statusIndicator = document.createElement('div');
                statusIndicator.className = 'status-indicator active';
                statusIndicator.innerHTML = '<i class="fas fa-circle"></i> Camera Active';
                videoFeed.parentElement.appendChild(statusIndicator);
                
                // Add camera properties display
                const cameraProps = document.createElement('div');
                cameraProps.className = 'camera-properties';
                cameraProps.innerHTML = `
                    <div class="camera-prop"><i class="fas fa-video"></i> Resolution: 1280x720</div>
                    <div class="camera-prop"><i class="fas fa-clock"></i> FPS: 30</div>
                `;
                videoFeed.parentElement.appendChild(cameraProps);
                
            } catch (error) {
                console.error('Camera initialization error:', error);
                alert('Error starting detection: ' + error.message);
                startBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
                hideLoading();
                
                // Add error status indicator
                const statusIndicator = document.createElement('div');
                statusIndicator.className = 'status-indicator inactive';
                statusIndicator.innerHTML = '<i class="fas fa-exclamation-circle"></i> Camera Error';
                videoFeed.parentElement.appendChild(statusIndicator);
                
                // Show error message in loading overlay
                const loadingText = document.createElement('div');
                loadingText.className = 'loading-text';
                loadingText.textContent = 'Camera initialization failed. Please check your camera connection.';
                loadingOverlay.appendChild(loadingText);
            }
        }
    });

    // Stop detection
    stopBtn.addEventListener('click', async function() {
        if (isDetectionRunning) {
            showLoading();
            stopBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Stopping Camera...';
            
            try {
                const response = await fetch('/stop_detection', {
                    method: 'POST'
                });

                if (!response.ok) {
                    throw new Error('Failed to stop detection');
                }

                isDetectionRunning = false;
                updateButtonStates();
                videoFeed.src = '';
                
                // Clear the video feed update interval
                if (videoFeedInterval) {
                    clearInterval(videoFeedInterval);
                    videoFeedInterval = null;
                }
                
                // Remove status indicator
                const statusIndicator = videoFeed.parentElement.querySelector('.status-indicator');
                if (statusIndicator) {
                    statusIndicator.remove();
                }
                
                // Show success message
                stopBtn.innerHTML = '<i class="fas fa-check"></i> Camera Stopped';
                setTimeout(() => {
                    stopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Detection';
                }, 2000);
            } catch (error) {
                alert('Error stopping detection: ' + error.message);
                stopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Detection';
            } finally {
                hideLoading();
            }
        }
    });

    // Clear words
    clearBtn.addEventListener('click', async function() {
        clearBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Clearing...';
        
        try {
            const response = await fetch('/clear_words', {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Failed to clear words');
            }

            detectedWordsList = [];
            detectedWords.innerHTML = '';
            fullSentence.innerHTML = '';
            translation.innerHTML = '';
            
            // Show success message
            clearBtn.innerHTML = '<i class="fas fa-check"></i> Cleared';
            setTimeout(() => {
                clearBtn.innerHTML = '<i class="fas fa-eraser"></i> Clear Words';
            }, 2000);
        } catch (error) {
            alert('Error clearing words: ' + error.message);
            clearBtn.innerHTML = '<i class="fas fa-eraser"></i> Clear Words';
        }
    });

    // Settings change handlers
    showBoxes.addEventListener('change', function() {
        if (isDetectionRunning) {
            updateVideoFeed();
        }
    });

    hospitalMode.addEventListener('change', function() {
        if (isDetectionRunning) {
            updateVideoFeed();
        }
    });

    // Update detection output periodically
    function updateDetectionOutput() {
        if (isDetectionRunning) {
            fetch('/get_detected_words')
                .then(response => response.json())
                .then(data => {
                    // Update detected words
                    detectedWords.innerHTML = '';
                    data.words.forEach(word => {
                        if (!detectedWordsList.includes(word)) {
                            detectedWordsList.push(word);
                            const wordElement = document.createElement('span');
                            wordElement.className = 'badge bg-primary me-1 mb-1';
                            wordElement.textContent = word;
                            detectedWords.appendChild(wordElement);
                            
                            // Add animation class
                            wordElement.classList.add('fade-in');
                            
                            // Remove animation class after animation completes
                            setTimeout(() => {
                                wordElement.classList.remove('fade-in');
                            }, 500);
                        }
                    });

                    // Update full sentence and translation
                    fullSentence.textContent = data.full_sentence;
                    translation.textContent = data.translation;
                })
                .catch(error => console.error('Error updating detection output:', error));
        }
    }

    // Initialize button states
    updateButtonStates();
    
    // Update detection output every 1 second
    setInterval(updateDetectionOutput, 1000);
}); 