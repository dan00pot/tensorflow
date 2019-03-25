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
MODEL_NAME = 'mo-hinh-mask'

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


#dinh nghia ham load_image_into_numpy_array cho viec chuyen doi numpy_array
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

#dinh nghia ham kiem tra voi mot hinh anh
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # nhap va su ly anh dau vao
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                #tao yeu cau de ket thuc ham
                if 'detection_masks' in tensor_dict:
                    # su ly cho 1 hinh anh don
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe là cần thiết để dịch mặt nạ từ hộp tọa độ để tọa độ hình ảnh và phù hợp với kích thước hình ảnh.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # thay doi lai masks
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                #ket thuc neu xd duoc mask
            
            #dinh nghia image cho tensorflow
                
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # chay inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

            # tat ca output la float32 numpy arrays, cover lai cho phu hop
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict



class maskiu(QDialog):
    def __init__(self):
        super (maskiu,self).__init__()
        loadUi('mask.ui',self)
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
        image = Image.fromarray(frame, 'RGB')
        image = image.convert('RGB')
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
	
# Ve ket qua box va masks.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
	        output_dict['detection_boxes'],
	        output_dict['detection_classes'],
	        output_dict['detection_scores'],
	        category_index,
	        instance_masks=output_dict.get('detection_masks'),
	        use_normalized_coordinates=True,
	        line_thickness=8, min_score_thresh=0.5
			)
        cv2.imshow("Mask R-CNN", image_np)
if __name__=='__main__':
    app=QApplication(sys.argv)
    window=maskiu()
    window.setWindowTitle('Mask design by Nguyen Ngoc Duy')
    window.show()
    sys.exit(app.exec_())