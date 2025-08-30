# @title Face Verification
!pip install opencv-python face_recognition
import cv2
import face_recognition
import numpy as np
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def video_stream():
  js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var pendingResolve = null;
    var shutdown = false;

    function removeDom() {
      stream.getTracks().forEach(function(track) {
        track.stop();
      });
      video.remove();
      div.remove();
      captureCanvas.remove();
      imgElement.remove();
    }

    async function setup() {
      div = document.createElement('div');
      document.body.appendChild(div);

      video = document.createElement('video');
      video.style.display = 'block';
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.display = 'none';
      div.appendChild(imgElement);

      stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: "user"}});
      video.srcObject = stream;

      await new Promise(function(resolve) {
        video.onloadedmetadata = function() {
          resolve(video);
        };
      });

      video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      captureCanvas.getContext('2d').drawImage(video, 0, 0);

      pendingResolve = null;
    }

    async function MtakePhoto(quality) {
      if (shutdown) {
        return;
      }

      if (!stream) {
        await setup();
      }

      captureCanvas.getContext('2d').drawImage(video, 0, 0);
      var data = captureCanvas.toDataURL('image/jpeg', quality);
      return data;
    }

    function Mshutdown() {
      shutdown = true;
      removeDom();
    }
    ''')

  display(js)
  def take_photo(quality=0.8):
    data = eval_js('MtakePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    return binary

  return take_photo


try:
    card_image = face_recognition.load_image_file("omani_card.jpg")
    card_face_encoding = face_recognition.face_encodings(card_image)[0]
    known_face_encodings = [card_face_encoding]
    known_face_names = ["Card Holder"]
except (IndexError, FileNotFoundError) as e:
    print(f"❌ Error: Could not process 'omani_card.jpg'. Make sure the file is uploaded and contains a clear face.")
    exit()


print("🚀 Starting camera... Look at the camera for verification.")
take_photo = video_stream()

while True:
    try:
        photo_data = take_photo()
        frame = cv2.imdecode(np.frombuffer(photo_data, np.uint8), -1)

        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            color = (0, 0, 255)

            if True in matches:
                name = "Match Found"
                color = (0, 255, 0)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2_imshow(frame)

    except Exception as e:
        eval_js('Mshutdown()')
        print("✅ Verification stopped.")
        break
