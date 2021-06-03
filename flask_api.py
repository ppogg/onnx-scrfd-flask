from flask import Flask, render_template, Response
import argparse
from tools.scrfd import *
import datetime
import cv2

class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        success, image = self.video.read()
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()
        return image

app = Flask(__name__)

@app.route('/')  # 主页
def index():
    # 具体格式保存在index.html文件中
    return render_template('index.html')

def scrfd(camera):
    detector = SCRFD(model_file='onnx/scrfd_500m_bnkps_shape160x160.onnx')
    while True:
        frame = camera.get_frame()
        # cv2.imshow('fourcc', frame)
        # img = cv2.imread(frame)

        for _ in range(1):
            ta = datetime.datetime.now()
            bboxes, kpss = detector.detect(frame, 0.5, input_size = (160, 160))
            tb = datetime.datetime.now()
            print('all cost:', (tb - ta).total_seconds() * 1000)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1, y1, x2, y2, score = bbox.astype(np.int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if kpss is not None:
                kps = kpss[i]
                for kp in kps:
                    kp = kp.astype(np.int)
                    cv2.circle(frame, tuple(kp), 1, (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    if model == 'scrfd':
        return Response(scrfd(VideoCamera()),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection using YOLO-Fastest in OPENCV')
    parser.add_argument('--model', type=str, default='scrfd')
    args = parser.parse_args()
    model = args.model
    app.run(host='0.0.0.0', debug=True, port=1938)