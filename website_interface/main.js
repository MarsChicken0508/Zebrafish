document.getElementById('yolo-button').addEventListener('click', function () {
    const fileInput = document.getElementById('file');

    if (fileInput.files.length === 0) {
        // 如果沒有選擇圖片文件，仍然顯示功能按鈕
        document.getElementById('buttons-container').style.display = 'block';
        return;
    }

    const file = fileInput.files[0];

    // 假設這裡有 YOLO 計算邏輯
    // 目前只展示圖片
    const reader = new FileReader();
    reader.onload = function (e) {
        const imageContainer = document.getElementById('image-container');
        const img = document.createElement('img');
        img.src = e.target.result;
        imageContainer.innerHTML = '';
        imageContainer.appendChild(img);

        // 顯示四個功能按鈕
        document.getElementById('buttons-container').style.display = 'block';
    };
    reader.readAsDataURL(file);

    // 顯示功能按鈕
    document.getElementById('buttons-container').style.display = 'block';
});

document.getElementById('zebrafish-video').addEventListener('click', function () {
    // 播放斑馬魚運動軌跡影片的邏輯
    console.log('斑馬魚運動軌跡影片');
});

document.getElementById('normal-behavior').addEventListener('click', function () {
    // 顯示正常佔比的邏輯
    console.log('斑馬魚行為 (正常佔比)');
});

document.getElementById('fear-behavior').addEventListener('click', function () {
    // 顯示害怕佔比的邏輯
    console.log('斑馬魚行為 (害怕佔比)');
});

document.getElementById('anxiety-behavior').addEventListener('click', function () {
    // 顯示焦慮佔比的邏輯
    console.log('斑馬魚行為 (焦慮佔比)');
});

document.getElementById('depression-behavior').addEventListener('click', function () {
    // 顯示壓抑佔比的邏輯
    console.log('斑馬魚行為 (壓抑佔比)');
});

document.getElementById('download-video').addEventListener('click', function () {
    // 下載影片的邏輯
    console.log('下載影片');
    // 這裡假設有一個已處理好的影片文件
    const link = document.createElement('a');
    link.href = 'path/to/your/video.mp4'; // 替換為實際的影片文件路徑
    link.download = 'video.mp4';
    link.click();
});
