// main.js

import { initializeRecorder } from './recorder.js';
import { initializeUploader } from './uploader.js';

/**
 * 初始化所有模块
 */
document.addEventListener('DOMContentLoaded', () => {
    initializeRecorder();
    initializeUploader();
});
