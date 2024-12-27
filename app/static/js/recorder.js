// recorder.js

let mediaRecorder;
let recordedChunks = [];
let recordedBlob = null;
let isRecording = false;
const MAX_RECORD_TIME = 30000; // 最大录音时间 30秒
let recordTimeout;

/**
 * 禁用或启用文件上传和录音按钮
 * @param {boolean} disableUpload - 是否禁用文件上传
 * @param {boolean} disableRecord - 是否禁用录音
 */
function toggleUploadAndRecord(disableUpload, disableRecord) {
    const audioFileInput = document.getElementById('audioFile');
    const recordBtn = document.getElementById('recordBtn');
    audioFileInput.disabled = disableUpload;
    recordBtn.disabled = disableRecord;
}

/**
 * 更新录音按钮的显示状态
 */
function updateRecordButton() {
    const recordBtn = document.getElementById('recordBtn');
    const recordingStatus = document.getElementById('recordingStatus');

    if (isRecording) {
        recordBtn.innerHTML = '<i class="fas fa-square"></i> 停止录音';
        recordBtn.classList.remove('btn-danger');
        recordBtn.classList.add('btn-secondary');
        recordingStatus.textContent = '正在录音...';
        recordingStatus.classList.add('recording');
    } else {
        recordBtn.innerHTML = '<i class="fas fa-circle"></i> 开始录音';
        recordBtn.classList.remove('btn-secondary');
        recordBtn.classList.add('btn-danger');
        recordingStatus.textContent = recordedBlob ? '录音已停止。' : '';
        recordingStatus.classList.remove('recording');
    }
}

/**
 * 初始化录音功能
 */
export function initializeRecorder() {
    const recordBtn = document.getElementById('recordBtn');
    const clearRecordingBtn = document.getElementById('clearRecordingBtn');
    const playback = document.getElementById('playback');
    const audioPlayback = document.getElementById('audioPlayback');

    // 初始状态更新
    updateRecordButton();
    clearRecordingBtn.style.display = 'none';

    // 录音按钮点击事件
    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            // 开始录音
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                Swal.fire('错误', '您的浏览器不支持音频录制。', 'error');
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

                    // 禁用文件上传功能，因为已有录音
                    toggleUploadAndRecord(true, false);
                };

                // 设置定时器自动停止录音
                recordTimeout = setTimeout(() => {
                    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                        mediaRecorder.stop();
                        Swal.fire('信息', '已达到最大录音时间。', 'info');
                    }
                }, MAX_RECORD_TIME);
            } catch (err) {
                console.error('发生以下错误: ' + err);
                Swal.fire('错误', '无法访问您的麦克风。', 'error');
            }
        } else {
            // 停止录音
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                clearTimeout(recordTimeout); // 清除定时器
            }
        }
    });

    // 清除录音按钮点击事件
    clearRecordingBtn.addEventListener('click', () => {
        clearRecording();
    });

    /**
     * 清除录音状态和录音文件
     */
    function clearRecording() {
        recordedBlob = null;
        isRecording = false;
        updateRecordButton();
        const recordingStatus = document.getElementById('recordingStatus');
        const playback = document.getElementById('playback');
        const audioPlayback = document.getElementById('audioPlayback');

        recordingStatus.textContent = '无录音。';
        playback.style.display = 'none';
        audioPlayback.src = '';
        document.getElementById('result').textContent = '录音已清除。';

        // 隐藏清除录音按钮
        clearRecordingBtn.style.display = 'none';

        // 启用文件上传功能
        toggleUploadAndRecord(false, false);
    }

    // 将录音 Blob 暴露给上传模块
    window.getRecordedBlob = () => recordedBlob;
}
