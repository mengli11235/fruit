from detectron2.data.datasets import register_coco_instances
register_coco_instances("fruit_train", {}, "/data/s2651513/fruit/dataset_AgrilFruit_forCounting/exp1/annotations/annotations_train.json", "/data/s2651513/fruit/dataset_AgrilFruit_forCounting/exp1/train_images_coco")
register_coco_instances("fruit_test", {}, "/data/s2651513/fruit/dataset_AgrilFruit_forCounting/exp1/annotations/annotations_test.json", "/data/s2651513/fruit/dataset_AgrilFruit_forCounting/exp1/test_images_coco")

import random,cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
fruits_nuts_metadata=MetadataCatalog.get("fruit_train")
dataset_dicts = DatasetCatalog.get("fruit_train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite("image.jpg", vis.get_image()[:, :, ::-1])