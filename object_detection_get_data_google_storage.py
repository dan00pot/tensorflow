"""thêm thư viện và các tiện ích cần thiết"""
from google.cloud import storage
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
sys.path.append("..")
import matplotlib

from utils import label_map_util
from utils import visualization_utils as vis_util
import imageio

"""tên của folder chứa dữ liệu training trước"""
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'

"""link đến folder chứa folder graph"""
CWD_PATH = 'C:/Anaconda3/envs/myNewEnv/models/research/object_detection'

"""Path đến file graph .pb"""
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

"""Link đến file label_map"""
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

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

"""Kết nối bucket thông qua khóa API"""
storage_client = storage.Client.from_service_account_json(
        'xxxxxxxxxxxxxxxxxxxxxxx.json')
bucket_name=('xxxxxxxxxxxxxxxxxxxx.appspot.com')
bucket = storage_client.get_bucket(bucket_name)

"""dinh nghia ham tai xuong khung hinh"""
def download_bucket(blob_name, path_to_file):
    """ dowload data to a bucket"""
    blob = bucket.blob(blob_name)
    blob.download_to_filename(path_to_file)

"""Định nghĩa hàm upload từng frame đến google strorage"""
def upload_to_bucket(blob_name1, path_to_file1):
    blob = bucket.blob(blob_name1)
    blob.upload_from_filename(path_to_file1)

i=0
j=0
# video =cv2.VideoCapture('a.mp4')

while True:
    # ret, frame = video.read()
    # frame_expanded = np.expand_dims(frame, axis=0)
    if i==0:
        path_to_file=('image_input/chan.jpg')
        blob_name=('abc/a.jpg')
        download_bucket(path_to_file, blob_name)
        # image=Image.open('abc/a.jpg')
        frame = cv2.imread('abc/a.jpg')
        frame_expanded = np.expand_dims(frame, axis=0)

    if i==1:
        path_to_file=('image_input/le.jpg')
        blob_name=('abc/a1.jpg')
        download_bucket(path_to_file, blob_name)
        # image=Image.open('abc/a1.jpg')
        frame = cv2.imread('abc/a1.jpg')		
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
        min_score_thresh=0.6
        )
    cv2.imwrite('abc/a0.jpg', frame)
    path_to_file1= 'abc/a0.jpg'
    blob_name1='image_output/chan.jpg'
    upload_to_bucket(blob_name1, path_to_file1)
    #cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    j += 1
    i = j% 2

cv2.destroyAllWindows()