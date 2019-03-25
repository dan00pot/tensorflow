Huong dan chay chuong trinh:
2. Training mot model bao gom:
   -Tao file tfrecord tu file hinh anh
       +danh nhan co the dung app labelme(thu duoc file xml chua thong tin danh nhan): http://labelme.csail.mit.edu/Release3.0/ hoac https://github.com/wkentaro/labelme hoac https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1
       +config xml to csv or json su dung file xml_to_csv.py (mot so file tham khao them json2csv.py) 
       +tao tf record tu file tfrecord.py va create_coco_tf_record.py cho mo hinh mask
       +luu y neu su dung tfrecord.py tao tu file csv (cau truc file csv trong luanvan) 
             # TO-DO replace this with label map
             def class_text_to_int(row_label):
             if row_label == 'nine':
                 return 1
             elif row_label == 'ten':
                 return 2
             elif row_label == 'jack':
                 return 3
             elif row_label == 'queen':
                 return 4
             elif row_label == 'king':
                 return 5
             elif row_label == 'ace':
                 return 6
             else:
        return None
       +luu y voi mo hinh mask su dung file create_coco_tf_record.py cau hinh file json xem trong luanvan 
       
   -config cau hinh cho mo hinh (mau trong traning/faster/faster_rcnn_inception_v2_coco)
   -Khoi tao training tren tensorflow
3. Chay chuong trinh