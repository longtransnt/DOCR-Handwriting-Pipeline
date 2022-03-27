from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


if __name__ == "__main__":
   
    # TODO:change the path
    default_instance = "trainval.json"
    images_path = "./Subset/" 
    register_coco_instances('paper', {}, default_instance, images_path)
    dataset_dicts = DatasetCatalog.get('paper')
    bills_metadata = MetadataCatalog.get('paper')   

    cfg = get_cfg()
    cfg.merge_from_file(
        './configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    )
    cfg.DATASETS.TRAIN = ('paper')
    cfg.DATASETS.TEST = ('paper')  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = (
        300
    )  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
