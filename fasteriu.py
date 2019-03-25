from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from google.cloud import storage
import os
import sys
import numpy as np
import os
import cv2
import tensorflow as tf
from io import StringIO
from PIL import Image
import numpy as np
import subprocess as sp
import sys
from object_detection.utils import ops as utils_ops
sys.path.append("..")

# them tien ich
from utils import label_map_util
from utils import visualization_utils as vis_util

"""tên của folder chứa dữ liệu training trước"""
MODEL_NAME = 'mo-hinh-faster'

"""link đến folder chứa folder graph"""
CWD_PATH = os.getcwd()

"""Path đến file graph .pb"""
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

"""Link đến file label_map"""
PATH_TO_LABELS = os.path.join(CWD_PATH,'label','mscoco_label_map.pbtxt')

"""Số lớp nhận dạng"""
NUM_CLASSES = 90

"""Load label map và danh mục các lớp nhận dạng"""
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

"""load tensorflow vào bộ nhớ"""
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


"""Định nghĩa input và output 
   Đầu vào của tensor là 1 image"""
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

"""Đầu ra là boxes, scores, và classes
   Mỗi hộp đại diện cho một phần của hình ảnh đối tượng cụ thể được phát hiện"""
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

"""Mỗi điểm đại diện cho mức độ tin cậy cho từng đối tượng.
   Điểm số được hiển thị trên hình ảnh kết quả, cùng với nhãn lớp"""
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

"""số lượng"""
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

class fasteriu(QDialog):
    def __init__(self):
        super (fasteriu,self).__init__()
        loadUi('faster.ui',self)
        self.image=None
        self.frame=None
        self.startButton.clicked.connect(self.capture)
        self.stopButton.clicked.connect(self.process)        
    def capture(self):
        mytext = self.textEdit.toPlainText()
        self.timer=QTimer(self)
        self.video = cv2.VideoCapture(mytext)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        self.timer=QTimer(self)
        self.timer.timeout.connect(self.loadhinh)
        self.timer.start(2)
		
    def process(self):
        self.timer.stop()		
    def loadhinh(self):
        ret, self.frame = self.video.read()
        self.frame = cv2.flip(self.frame,1)
        frame = self.frame
        frame_expanded = np.expand_dims(frame, axis=0)
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
	        frame, np.squeeze(boxes), 
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
		    category_index,
		    use_normalized_coordinates=True,
		    line_thickness=8,
		    min_score_thresh=0.63
		    )
        cv2.imshow("Faster-RCNN", frame)

if __name__=='__main__':
    app=QApplication(sys.argv)
    window=fasteriu()
    window.setWindowTitle('Faster-RCNN')
    window.show()
    sys.exit(app.exec_())