from flask import Flask, render_template, request,Response
from werkzeug.utils import secure_filename
from image_detection import process_image
from video_detection import video_detection
import os
import cv2



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
UPLOAD_FOLDER = 'static/files'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = 'static/results'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/deteksi',methods=['GET', 'POST'])
def deteksi():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)
                img = cv2.imread(image_path)
                result_img = process_image(img)
                result_filename = 'result.jpg'
                result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
                cv2.imwrite(result_path, result_img)
                return render_template('deteksi.html', image_path=image_path, result_path=result_filename)
    return render_template('deteksi.html')
    
@app.route('/deteksi_video',methods=['GET', 'POST'])
def deteksi_video():
    if 'video' in request.files:
            video = request.files['video']
            if video and allowed_file(video.filename):
                filename = secure_filename(video.filename)
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                video.save(video_path)
                result_video_filename = 'result.mp4'
                result_video_path = os.path.join(app.config['RESULTS_FOLDER'], result_video_filename)
                video_generator = video_detection(video_path)

                def generate():
                    for frame in video_generator:
                        ret, jpeg = cv2.imencode('.jpg', frame)
                        frame_bytes = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

                return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template('deteksi_video.html')

@app.route('/deteksi_webcam')
def deteksi_webcam():
    return render_template('deteksi_webcam.html')

# Route for streaming webcam
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to generate webcam frames
def generate_frames():
    camera = cv2.VideoCapture(1)  # Open webcam
    while True:
        success, frame = camera.read()  # Read frame from webcam
        if not success:
            print("tidak berhasil read frame from webcam")
            break
        else:
            # Perform object detection on the frame here
            # You can use the loaded model to detect objects in the frame

            # Convert the frame to JPEG format
            print("berhasil read frame from webcam")
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# [Routing untuk API]		
@app.route("/get/chatbot")
def apiDeteksi():
    # NLP
    prediction_input = request.args.get('prediction_input')
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)

    input_text_tokenized = bert_tokenizer.encode(prediction_input,
                                             truncation=True,
                                             padding='max_length',
                                             return_tensors='tf')
  
    bert_predict = bert_load_model(input_text_tokenized)          # Lakukan prediksi
    bert_predict = tf.nn.softmax(bert_predict[0], axis=-1)         # Softmax function untuk mendapatkan hasil klasifikasi
    output = tf.argmax(bert_predict, axis=1)

    response_tag = le.inverse_transform([output])[0]
    res = random.choice(responses[response_tag])

    return res

if __name__ == '__main__':

    #Pretrained Model
    PRE_TRAINED_MODEL = 'indobenchmark/indobert-base-p2'

    #Load tokenizer dari pretrained model
    bert_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)

    # Load hasil fine-tuning
    bert_load_model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=27)

    #Load Model
    bert_load_model.load_weights('bert-model.h5')

    #Deploy di localhost
    app.run(debug=True)
