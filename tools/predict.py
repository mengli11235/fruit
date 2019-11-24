"""
predicted results using trained model and get statistics out of the results

how to look at training curves in tensorboard: 
put event files in output folder first, then:
load_ext tensorboard
tensorboard --logdir output
"""
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import random, cv2
import os
import numpy as np
import json
import itertools
import matplotlib.pyplot as plt
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor, default_argument_parser
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # set configs
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 
    # register our fruit test dataset
    register_coco_instances("fruit_test", {}, args.test_annotations, args.test_images)
    fruit_metadata=MetadataCatalog.get("fruit_test")
    # load the trained weights
    cfg.MODEL.WEIGHTS = os.path.join( cfg.OUTPUT_DIR, args.config_file, "model_final.pth")
    # set the testing threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.DATASETS.TEST = ("fruit_test", )
    NUM_PROPOSALS = 1000*300
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get("fruit_test")
    # fruit count per image
    label_counts = []
    predict_counts = []
    correct_counts = []
    # tps: true positives, fns: false negatives, fps: false positives
    fruit_tps = [0,0,0,0,0]
    fruit_fns = [0,0,0,0,0]
    fruit_fps = [0,0,0,0,0]
    folders = ['lemon', 'custardapple', 'apple', 'pear', 'persimmon']
    if not os.path.exists("example_outputs"):
        os.makedirs("example_outputs")
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        # predicted outputs
        outputs = predictor(im)
        predictions = outputs["instances"].to("cpu")
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        classes = classes.tolist()
        # count numbers
        predict_counts.append(len(boxes))
        label_counts.append(len(d['annotations']))
        labeled_classes = [x['category_id'] for x in d['annotations']]
        tp = 0
        for k in range(len(labeled_classes)):
            # find correctly classified samples
            if labeled_classes[k] in classes:
                tp += 1
                fruit_tps[labeled_classes[k]] += 1
                classes.remove(labeled_classes[k])
            else:
                fruit_fns[labeled_classes[k]] += 1
            if len(classes) == 0:
                break
        for item in classes:
            fruit_fps[item] += 1
        correct_counts.append(tp)
        # save the visualized image
        v = Visualizer(im[:, :, ::-1],
                       metadata=fruit_metadata, 
                       scale=0.8, 
                       #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )                          
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("example_outputs/image"+str(d['image_id'])+".jpg", v.get_image()[:, :, ::-1]) 
    # true negatives out of total proposals       
    tns = NUM_PROPOSALS - sum(label_counts) -sum(predict_counts) + sum(correct_counts)
    # the MSE between labeled counts and predicted counts
    print("Count Error:")
    print(sum((np.array(label_counts)-np.array(predict_counts))**2)/len(label_counts))
    # the correct percentage of predicted counts
    print("Count Accuracy:")
    print(sum(np.array(label_counts)==np.array(predict_counts))/len(label_counts))
    # accuracy out of all proposals
    print("Total Accuracy:")
    print((sum(correct_counts)+tns)/NUM_PROPOSALS)
    # recall of all fruit classes
    print("Total Recall:")
    print(sum(correct_counts)/sum(label_counts))
    # precision of all fruit classes
    print("Total Precision:")
    print(sum(correct_counts)/sum(predict_counts))
    # the total number of fruits
    fruit_totals = sum(fruit_tps) + sum(fruit_fns)
    for i in range(len(folders)):
        # calculate accuracy, recall and precision for each fruit class respectively
        fruit_tns = fruit_totals - fruit_tps[i] - fruit_fps[i] - fruit_fns[i]
        print('{}:  Accuracy: {} Recall: {} Precision:{}'.format(folders[i], (fruit_tps[i]+fruit_tns)/fruit_totals, fruit_tps[i]/(fruit_tps[i]+fruit_fns[i]),fruit_tps[i]/(fruit_tps[i]+fruit_fps[i])))
    # write counting results
    with open('example_outputs/test.txt', 'w+') as f:
        f.write("".join(list(map(str,[label_counts,predict_counts]))))
    f.closed