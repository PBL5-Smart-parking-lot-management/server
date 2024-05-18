const videoStream = document.getElementById('video-stream');
const serverUrl = 'http://192.168.1.22:5000/video_feed';

function updateVideoStream() {
    fetch(serverUrl)
        .then(response => response.blob())
        .then(blob => {
            const objectURL = URL.createObjectURL(blob);
            videoStream.src = objectURL;
        })
        .catch(error => console.error(error));
}

// Cập nhật luồng video mỗi 30ms
setInterval(updateVideoStream, 30);