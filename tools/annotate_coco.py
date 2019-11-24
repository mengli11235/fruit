## This file is used to annotate xml files to coco format

import json
import os
import cv2
from xml.dom.minidom import parse
import xml.dom.minidom
 
if __name__ == '__main__':
    path = '../dataset_AgrilFruit_forCounting/exp1/'     
    # fruit classes
    folders = ['lemon', 'custardapple', 'apple', 'pear', 'persimmon']
    train_dataset = {'categories':[], 'images':[], 'annotations':[]}
    test_dataset = {'categories':[], 'images':[], 'annotations':[]}
    # create coco categories
    for i, j in enumerate(folders, 0):
      train_dataset['categories'].append({'id': i, 'name': j, 'supercategory': 'mark'})
      test_dataset['categories'].append({'id': i, 'name': j, 'supercategory': 'mark'})
    # train annotations
    for i, f in enumerate(os.listdir(os.path.join(path, 'train_xml'))):
        # read through xml tree
        DOMTree = xml.dom.minidom.parse(os.path.join(path, 'train_xml', f))
        collection = DOMTree.documentElement
        filename = collection.getElementsByTagName("filename")[0]
        folder = collection.getElementsByTagName("folder")[0]
        size = collection.getElementsByTagName("size")[0]
        width = size.getElementsByTagName("width")[0]
        height = size.getElementsByTagName("height")[0]
        objects = collection.getElementsByTagName("object")
        train_dataset['images'].append({'file_name': filename.childNodes[0].data,
                                  'id': i,
                                  'width': int(width.childNodes[0].data),
                                  'height': int(height.childNodes[0].data)})
        for k,object in enumerate(objects):
            bndbox = object.getElementsByTagName('bndbox')[0]
            name = object.getElementsByTagName('name')[0]
            xmin = bndbox.getElementsByTagName('xmin')[0]
            ymin = bndbox.getElementsByTagName('ymin')[0]
            xmax = bndbox.getElementsByTagName('xmax')[0]
            ymax = bndbox.getElementsByTagName('ymax')[0]
            if name.childNodes[0].data.lower() == folder.childNodes[0].data.lower():
                x1 = float(xmin.childNodes[0].data)
                y1 = float(ymin.childNodes[0].data)
                x2 = float(xmax.childNodes[0].data)
                y2 = float(ymax.childNodes[0].data)
                width_object = max(0, x2 - x1)
                height_object = max(0, y2 - y1)
                train_dataset['annotations'].append({'area': width_object * height_object,
                                              'bbox': [x1, y1, width_object, height_object],
                                              'category_id': folders.index(folder.childNodes[0].data.lower()),
                                              'id': i*100+k,
                                              'image_id': i,
                                              'iscrowd': 0,
                                              'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]})
    #test annotations
    for i, f in enumerate(os.listdir(os.path.join(path, 'test_xml'))):
        DOMTree = xml.dom.minidom.parse(os.path.join(path, 'test_xml', f))
        collection = DOMTree.documentElement
        filename = collection.getElementsByTagName("filename")[0]
        folder = collection.getElementsByTagName("folder")[0]
        size = collection.getElementsByTagName("size")[0]
        width = size.getElementsByTagName("width")[0]
        height = size.getElementsByTagName("height")[0]
        objects = collection.getElementsByTagName("object")
        test_dataset['images'].append({'file_name': filename.childNodes[0].data,
                                  'id': i,
                                  'width': int(width.childNodes[0].data),
                                  'height': int(height.childNodes[0].data)})
        for k,object in enumerate(objects):
            bndbox = object.getElementsByTagName('bndbox')[0]
            name = object.getElementsByTagName('name')[0]
            xmin = bndbox.getElementsByTagName('xmin')[0]
            ymin = bndbox.getElementsByTagName('ymin')[0]
            xmax = bndbox.getElementsByTagName('xmax')[0]
            ymax = bndbox.getElementsByTagName('ymax')[0]
            if name.childNodes[0].data.lower() == folder.childNodes[0].data.lower():
                x1 = float(xmin.childNodes[0].data)
                y1 = float(ymin.childNodes[0].data)
                x2 = float(xmax.childNodes[0].data)
                y2 = float(ymax.childNodes[0].data)
                width_object = max(0, x2 - x1)
                height_object = max(0, y2 - y1)
                test_dataset['annotations'].append({'area': width_object * height_object,
                                              'bbox': [x1, y1, width_object, height_object],
                                              'category_id': folders.index(folder.childNodes[0].data.lower()),
                                              'id': i*100+k,
                                              'image_id': i,
                                              'iscrowd': 0,
                                              'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]})
    # save the json file                                          
    save_folder = os.path.join(path, 'annotations')
    if not os.path.exists(save_folder):
      os.makedirs(save_folder)
    json_name = os.path.join(save_folder, 'annotations_train.json')
    with open(json_name, 'w') as f:
      json.dump(train_dataset, f)
    json_name = os.path.join(save_folder, 'annotations_test.json')
    with open(json_name, 'w') as f:
      json.dump(test_dataset, f)