// Ajax form submission for image form
$('#image-form').submit(function(e) {
    e.preventDefault();
    var formData = new FormData(this);
    
    $.ajax({
        url: '/',
        type: 'POST',
        data: formData,
        success: function(data) {
            updateResultView(data.result_path);
        },
        cache: false,
        contentType: false,
        processData: false
    });
});

// Ajax form submission for video form
$('#video-form').submit(function(e) {
    e.preventDefault();
    var formData = new FormData(this);

    $.ajax({
        url: '/',
        type: 'POST',
        data: formData,
        success: function(data) {
            updateResultView(data.result_video_path);
        },
        cache: false,
        contentType: false,
        processData: false
    });
});

// Fungsi untuk memulai webcam
    function startWebcam() {
        // Memeriksa apakah browser mendukung getUserMedia
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            // Mengambil elemen video
            const video = document.getElementById('webcam-video');

            // Mengakses webcam dengan getUserMedia
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(stream) {
                    // Menyetel src elemen video dengan stream webcam
                    video.srcObject = stream;
                })
                .catch(function(error) {
                    console.log('Error starting webcam: ', error);
                });
        } else {
            console.log('Browser tidak mendukung getUserMedia');
        }
    }
