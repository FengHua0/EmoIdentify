// main.js

import { initializeRecorder } from './recorder.js';
import { initializeUploader } from './uploader.js';

/**
 * 定时检测后端心跳
 */
function checkBackendStatus() {
    fetch('/ping')
        .then(response => response.json())
        .then(data => {
            if (data.status !== 'ok') {
                Swal.fire('警告', '后端服务异常，请检查服务器！', 'warning');
            }
        })
        .catch(() => {
            Swal.fire('错误', '无法连接后端服务！', 'error');
        });
}

// 每隔5秒检测一次
setInterval(checkBackendStatus, 5000);

/**
 * 初始化所有模块
 */
document.addEventListener('DOMContentLoaded', () => {
    initializeRecorder();
    initializeUploader();
});
