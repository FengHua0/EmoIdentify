document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault();

    const formData = new FormData();
    const audioFile = document.getElementById('audioFile').files[0];
    const modelSelection = document.getElementById('modelSelection').value;  // 获取用户选择的模型类型

    // 检查是否选择了文件
    if (!audioFile) {
        document.getElementById('result').textContent = 'Please select an audio file!';
        return;
    }

    formData.append('audio', audioFile);  // 添加音频文件到 FormData
    formData.append('model', modelSelection);  // 添加用户选择的模型类型

    // 更新状态：上传中
    document.getElementById('result').textContent = 'Uploading file...';

    // 使用 fetch 发送 POST 请求
    fetch('/predict', {
        method: 'POST',
        body: formData  // 发送包含音频文件和选择模型的 FormData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok.');
            }
            document.getElementById('result').textContent = 'Processing audio...';
            return response.json();
        })
        .then(data => {
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
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').textContent = 'An error occurred.';
        });
});
