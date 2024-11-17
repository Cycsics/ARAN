import warnings
warnings.filterwarnings('ignore')
from aran import ARAN
import torch
import sys
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(f"Python Version: {sys.version}")
    print(f"CUDA Version: {torch.version.cuda}")
    dataset_list = [
        # f'REEF.yaml',
        f'URPC.yaml',
        # f'Brackish_1920.yaml',
        # f'DUO.yaml',
        # f'DUO_COCO.yaml',
        # f'coco.yaml',
    ]
    model_list = [
        "ARAN.yaml",
    ]
    # model_size = 'yolov9c.pt'
    model_size = 'aran.pt'
    # optimizer = ['SGD', 'Adam'] # using SGD
    # optimizers = ['SGD'] # using SGD
    optimizer = 'SGD'
    iou_list = [0.6]
    iou = 0.6
    batch_size = 16
    # imgszs = [640, 720, 1024, 1280]
    # moedel_name = "yolov8-AFPN.yaml"
    for dataset in dataset_list:
        for model_name in model_list:
            # for optimizer in optimizers:
            # for iou in iou_list:
                for i in range(1):
                    # if dataset == 'REEF.yaml' and model_name == 'yolov8-iRMB_EMA-HAT3-AFPN.yaml':
                    #     continue
                    # model = RTDETR(model_name)
                    model = ARAN(model_name)
                    # model.load(model_size) 
                    # logging.info(f"GFLOPS: {gflops_calculated_value}")
                    model.train(data=dataset,
                                project=f'runs/{dataset.split(".")[0]}',
                                name=f"{(model_name.split('.')[0])}-{optimizer}-1280-{iou}-{batch_size}",
                                # name=f"{model_name.split('-')[0]}-{(model_name.split('.')[0]).split('yolov')[1][1:]}-{optimizer}-1280-{iou}-{batch_size}",
                                cache=False,
                                imgsz=[1920, 1080],
                                epochs=100,
                                batch=batch_size,
                                close_mosaic=10,
                                workers=8,
                                device=0,
                                optimizer=optimizer, # using SGD
                                amp=True,# close amp
                                single_cls=False,
                                deterministic=True,
                                verbose=True,
                                # cache=True,
                                patience=100,
                                iou=iou,
                                lr0=0.01
                                # lr0=0.001
                                )
    
    