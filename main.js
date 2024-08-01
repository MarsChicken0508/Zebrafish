document.getElementById('yolo-button').addEventListener('click', function () {
    const fileInput = document.getElementById('file');
    const weightFileInput = document.getElementById('weight-file');

    if (fileInput.files.length === 0 || weightFileInput.files.length === 0) {
        alert('請選擇圖片和權重值檔案');
        return;
    }

    const file = fileInput.files[0];
    const weightFile = weightFileInput.files[0];

    // 假設這裡有 YOLO 計算邏輯
    // 目前只展示圖片
    const reader = new FileReader();
    reader.onload = function (e) {
        const imageContainer = document.getElementById('image-container');
        const img = document.createElement('img');
        img.src = e.target.result;
        imageContainer.innerHTML = '';
        imageContainer.appendChild(img);
    };
    reader.readAsDataURL(file);

    // 假設這裡有處理權重值的邏輯
    // ...
});
