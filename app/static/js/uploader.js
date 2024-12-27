// uploader.js

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
 * 初始化上传和表单提交功能
 */
export function initializeUploader() {
    const audioFileInput = document.getElementById('audioFile');
    const clearUploadBtn = document.getElementById('clearUploadBtn');
    const uploadForm = document.getElementById('uploadForm');
    const playback = document.getElementById('playback');
    const audioPlayback = document.getElementById('audioPlayback');

    // 初始状态隐藏清除上传按钮
    clearUploadBtn.style.display = 'none';

    // 监控文件上传变化
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

    // 清除上传文件功能
    clearUploadBtn.addEventListener('click', () => {
        audioFileInput.value = ''; // 清除文件输入
        document.getElementById('result').textContent = '文件上传已清除。';

        // 隐藏清除上传按钮
        clearUploadBtn.style.display = 'none';

        // 启用录音功能
        toggleUploadAndRecord(false, false);
    });

    // 表单提交处理
    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();

        const formData = new FormData();
        const audioFile = audioFileInput.files[0];
        const recordedBlob = window.getRecordedBlob(); // 获取录音 Blob
        const modelSelection = document.getElementById('modelSelection').value; // 获取选择的模型

        // 检查是否选择了文件或录制了音频
        if (!audioFile && !recordedBlob) {
            Swal.fire('错误', '请选择一个音频文件或录制音频！', 'error');
            return;
        }

        if (audioFile) {
            formData.append('audio', audioFile); // 添加音频文件到 FormData
        } else if (recordedBlob) {
            // 将 Blob 转换为 File 对象
            const recordedFile = new File([recordedBlob], 'recorded_audio.webm', { type: 'audio/webm' });
            formData.append('audio', recordedFile);
        }

        formData.append('model', modelSelection); // 添加用户选择的模型类型

        // 更新状态：上传中
        document.getElementById('result').textContent = '正在上传文件...';

        // 显示处理中的弹窗
        Swal.fire({
            title: '处理中',
            text: '您的音频正在分析中...',
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
                throw new Error('网络响应不正常。');
            }
            return response.json();
        })
        .then(data => {
            Swal.close();
            handleResponse(data);
        })
        .catch(error => {
            console.error('错误:', error);
            Swal.close();
            Swal.fire('错误', '发生了一个错误。', 'error');
        });
    });

    /**
     * 处理后端响应
     * @param {Object} data - 后端返回的数据
     */
    function handleResponse(data) {
        // 检查是否有错误
        if (data.error) {
            Swal.fire('错误', data.error, 'error');
            return;
        }

        const predictedCategory = data.predicted_category;
        const featureName = data.feature_name;
        const base64 = data.base64;

        // 显示预测结果
        document.getElementById('result').textContent = `预测结果: ${predictedCategory}`;

        // 显示特征展示区域
        if (base64) {
            const featureContainer = document.getElementById('feature-container');
            const resultTab = document.getElementById('resultTab');
            const resultTabContent = document.getElementById('resultTabContent');

            // 清空之前的内容
            resultTab.innerHTML = '';
            resultTabContent.innerHTML = '';

            // 创建标签页
            const featureTabId = `feature-${featureName.replace(/\s+/g, '-').toLowerCase()}`;
            const predictionTabId = 'prediction-tab';

            // 特征标签
            const featureTab = document.createElement('li');
            featureTab.classList.add('nav-item');

            const featureTabLink = document.createElement('a');
            featureTabLink.classList.add('nav-link', 'active');
            featureTabLink.id = `${featureTabId}-tab`;
            featureTabLink.setAttribute('data-toggle', 'tab');
            featureTabLink.href = `#${featureTabId}`;
            featureTabLink.setAttribute('role', 'tab');
            featureTabLink.setAttribute('aria-controls', featureTabId);
            featureTabLink.setAttribute('aria-selected', 'true');
            featureTabLink.textContent = featureName;

            featureTab.appendChild(featureTabLink);
            resultTab.appendChild(featureTab);

            // 预测结果标签
            const predictionTab = document.createElement('li');
            predictionTab.classList.add('nav-item');

            const predictionTabLink = document.createElement('a');
            predictionTabLink.classList.add('nav-link');
            predictionTabLink.id = `${predictionTabId}-tab`;
            predictionTabLink.setAttribute('data-toggle', 'tab');
            predictionTabLink.href = `#prediction`;
            predictionTabLink.setAttribute('role', 'tab');
            predictionTabLink.setAttribute('aria-controls', 'prediction');
            predictionTabLink.setAttribute('aria-selected', 'false');
            predictionTabLink.textContent = 'Prediction Result';

            predictionTab.appendChild(predictionTabLink);
            resultTab.appendChild(predictionTab);

            // 创建特征内容
            const featureContent = document.createElement('div');
            featureContent.classList.add('tab-pane', 'fade', 'show', 'active');
            featureContent.id = featureTabId;
            featureContent.setAttribute('role', 'tabpanel');
            featureContent.setAttribute('aria-labelledby', `${featureTabId}-tab`);

            const featureCard = document.createElement('div');
            featureCard.classList.add('card', 'mt-4');

            const featureCardBody = document.createElement('div');
            featureCardBody.classList.add('card-body');

            const featureImg = document.createElement('img');
            featureImg.id = `heatmap-${featureTabId}`;
            featureImg.src = `data:image/png;base64,${base64}`;
            featureImg.alt = `${featureName} Heatmap`;
            featureImg.classList.add('img-fluid');

            featureCardBody.appendChild(featureImg);
            featureCard.appendChild(featureCardBody);
            featureContent.appendChild(featureCard);
            resultTabContent.appendChild(featureContent);

            // 创建预测结果内容
            const predictionContent = document.createElement('div');
            predictionContent.classList.add('tab-pane', 'fade');
            predictionContent.id = 'prediction';
            predictionContent.setAttribute('role', 'tabpanel');
            predictionContent.setAttribute('aria-labelledby', 'prediction-tab');

            const predictionCard = document.createElement('div');
            predictionCard.classList.add('card', 'mt-4');

            const predictionCardBody = document.createElement('div');
            predictionCardBody.classList.add('card-body');

            const predictionTitle = document.createElement('h3');
            predictionTitle.classList.add('card-title');
            predictionTitle.textContent = '预测结果';

            const predictionText = document.createElement('p');
            predictionText.id = 'predictedCategory';
            predictionText.textContent = predictedCategory;

            predictionCardBody.appendChild(predictionTitle);
            predictionCardBody.appendChild(predictionText);
            predictionCard.appendChild(predictionCardBody);
            predictionContent.appendChild(predictionCard);
            resultTabContent.appendChild(predictionContent);

            // 显示特征展示区域
            featureContainer.classList.remove('d-none');

            // 切换到预测结果标签页
            $('#prediction-tab').tab('show'); // 使用 jQuery 激活预测结果标签
        } else {
            console.warn('未收到特征热力图数据。');
            document.getElementById('result').textContent += '\n(特征热力图不可用)';
        }

        // 将预测结果填充到标签内容中
        document.getElementById('predictedCategory').textContent = predictedCategory;
    }
}
