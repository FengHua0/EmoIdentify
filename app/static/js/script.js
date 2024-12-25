document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault(); // 阻止表单的默认提交行为

    const formData = new FormData();
    const audioFile = document.getElementById('audioFile').files[0];

    // 检查是否选择了文件
    if (!audioFile) {
        document.getElementById('result').textContent = 'Please select an audio file!';
        return;
    }

    formData.append('audio', audioFile);

    // 更新状态：上传中
    document.getElementById('result').textContent = 'Uploading file...';

    // 使用 fetch 发送 POST 请求
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok.');
            }
            document.getElementById('result').textContent = 'Processing audio...';
            return response.json();
        })
        .then(data => {
            // 更新状态：显示预测结果
            document.getElementById('result').textContent =
                `Prediction Result: ${data.predicted_category}`;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').textContent = 'An error occurred.';
        });
});
