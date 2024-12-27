// script.js

let mediaRecorder;
let recordedChunks = [];
let recordedBlob = null;
let isRecording = false;
const MAX_RECORD_TIME = 30000; // 30秒
let recordTimeout;

const recordBtn = document.getElementById('recordBtn');
const recordingStatus = document.getElementById('recordingStatus');
const audioFileInput = document.getElementById('audioFile');
const playback = document.getElementById('playback');
const audioPlayback = document.getElementById('audioPlayback');
const clearUploadBtn = document.getElementById('clearUploadBtn'); // 清除上传按钮
const clearRecordingBtn = document.getElementById('clearRecordingBtn'); // 清除录音按钮

// 用于激活 MFCC 热力图和预测结果展示
const resultTabContent = document.getElementById('resultTabContent');
const mfccTab = document.getElementById('mfcc-tab');
const predictionTab = document.getElementById('prediction-tab');

/**
 * 禁用或启用文件上传和录音功能
 * @param {boolean} disableUpload - 是否禁用文件上传
 * @param {boolean} disableRecord - 是否禁用录音
 */
function toggleUploadAndRecord(disableUpload, disableRecord) {
    audioFileInput.disabled = disableUpload;
    recordBtn.disabled = disableRecord;
}

/**
 * 更新录音按钮的显示状态
 */
function updateRecordButton() {
    if (isRecording) {
        recordBtn.innerHTML = '<i class="fas fa-square"></i> Stop Recording';
        recordBtn.classList.remove('btn-danger');
        recordBtn.classList.add('btn-secondary');
        recordingStatus.textContent = 'Recording...';
        recordingStatus.classList.add('recording');
    } else {
        recordBtn.innerHTML = '<i class="fas fa-circle"></i> Start Recording';
        recordBtn.classList.remove('btn-secondary');
        recordBtn.classList.add('btn-danger');
        recordingStatus.textContent = recordedBlob ? 'Recording stopped.' : '';
        recordingStatus.classList.remove('recording');
    }
}

/**
 * 录音按钮点击事件处理
 */
recordBtn.addEventListener('click', async () => {
    if (!isRecording) {
        // 开始录音
        // 检查浏览器是否支持 MediaRecorder
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            Swal.fire('Error', 'Your browser does not support audio recording.', 'error');
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            isRecording = true;
            updateRecordButton();

            // 禁用文件上传功能
            toggleUploadAndRecord(true, false);

            mediaRecorder.ondataavailable = function(e) {
                if (e.data.size > 0) {
                    recordedChunks.push(e.data);
                }
            };

            mediaRecorder.onstop = function() {
                recordedBlob = new Blob(recordedChunks, { type: 'audio/webm' });
                recordedChunks = [];
                isRecording = false;
                updateRecordButton();

                // 显示播放控件
                audioPlayback.src = URL.createObjectURL(recordedBlob);
                playback.style.display = 'block';

                // 显示清除录音按钮
                clearRecordingBtn.style.display = 'inline-block';

                // 禁用文件上传功能，因为已经有录音
                toggleUploadAndRecord(true, false);
            };

            // 设置定时器自动停止录音
            recordTimeout = setTimeout(() => {
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                    Swal.fire('Info', 'Maximum recording time reached.', 'info');
                }
            }, MAX_RECORD_TIME);
        } catch (err) {
            console.error('The following error occurred: ' + err);
            Swal.fire('Error', 'Unable to access your microphone.', 'error');
        }
    } else {
        // 停止录音
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            clearTimeout(recordTimeout); // 清除定时器
        }
    }
});

/**
 * 表单提交处理
 */
document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData();
    const audioFile = audioFileInput.files[0];
    const modelSelection = document.getElementById('modelSelection').value;  // 获取用户选择的模型类型

    // 检查是否选择了文件或录制了音频
    if (!audioFile && !recordedBlob) {
        Swal.fire('Error', 'Please select an audio file or record audio!', 'error');
        return;
    }

    if (audioFile) {
        formData.append('audio', audioFile);  // 添加音频文件到 FormData
    } else if (recordedBlob) {
        // 将 Blob 转换为 File 对象
        const recordedFile = new File([recordedBlob], 'recorded_audio.webm', { type: 'audio/webm' });
        formData.append('audio', recordedFile);
    }

    formData.append('model', modelSelection);  // 添加用户选择的模型类型

    // 更新状态：上传中
    document.getElementById('result').textContent = 'Uploading file...';

    // 显示处理中的弹窗
    Swal.fire({
        title: 'Processing',
        text: 'Your audio is being analyzed...',
        icon: 'info',
        allowOutsideClick: false,
        didOpen: () => {
            Swal.showLoading();
        }
    });

    // 使用 fetch 发送 POST 请求
    fetch('/predict', { // 替换为你的后端接口
        method: 'POST',
        body: formData  // 发送包含音频文件和选择模型的 FormData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok.');
        }
        return response.json();
    })
    .then(data => {
        Swal.close();
        // 分别接收 predicted_category 和 mfcc_heatmap
        const predictedCategory = data.predicted_category;
        const mfccHeatmap = data.mfcc_heatmap;

        // 显示预测结果
        document.getElementById('result').textContent =
            `Prediction Result: ${predictedCategory}`;

        // 显示 MFCC 热力图
        if (mfccHeatmap) {
            const mfccContainer = document.getElementById('mfcc-container');
            const mfccImage = document.getElementById('mfcc-heatmap');
            mfccImage.src = `data:image/png;base64,${mfccHeatmap}`;  // 设置图片的 src 属性为 Base64 数据
            mfccContainer.style.display = 'block';  // 显示 MFCC 特征的区域
        } else {
            console.warn('No MFCC heatmap data received.');
            document.getElementById('result').textContent += '\n(MFCC heatmap not available)';
        }

        // 将预测结果填充到标签内容中
        document.getElementById('predictedCategory').textContent = predictedCategory;

        // 切换到预测结果标签页
        $(predictionTab).tab('show'); // 使用 jQuery 激活预测结果标签
    })
    .catch(error => {
        console.error('Error:', error);
        Swal.close();
        Swal.fire('Error', 'An error occurred.', 'error');
    });
});

/**
 * 清除上传文件功能
 */
clearUploadBtn.addEventListener('click', () => {
    audioFileInput.value = ''; // 清除文件输入
    document.getElementById('result').textContent = 'File upload cleared.';

    // 隐藏清除上传按钮
    clearUploadBtn.style.display = 'none';

    // 启用录音功能
    toggleUploadAndRecord(false, false);
});

/**
 * 清除录音功能
 */
clearRecordingBtn.addEventListener('click', () => {
    clearRecording();
});

/**
 * 清除录音状态和录音文件
 */
function clearRecording() {
    recordedBlob = null;
    recordingStatus.textContent = 'No recording.';
    playback.style.display = 'none';
    audioPlayback.src = '';
    document.getElementById('result').textContent = 'Recording cleared.';

    // 隐藏清除录音按钮
    clearRecordingBtn.style.display = 'none';

    // 启用文件上传功能
    toggleUploadAndRecord(false, false);
}

/**
 * 监控文件上传变化
 */
audioFileInput.addEventListener('change', () => {
    if (audioFileInput.files.length > 0) {
        // 显示清除上传按钮
        clearUploadBtn.style.display = 'inline-block';

        // 禁用录音功能
        toggleUploadAndRecord(false, true);
    } else {
        // 隐藏清除上传按钮
        clearUploadBtn.style.display = 'none';

        // 启用录音功能
        toggleUploadAndRecord(false, false);
    }
});
