####FASTER_RCNN
python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py\
    /workspace/mmdetection/models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/faster_rcnn/R-50-FPN \
    --show \
    --device cuda:0



python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py  \
    /workspace/mmdetection/models/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/faster_rcnn/R-101-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py \
    /workspace/mmdetection/models/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/faster_rcnn/X-101-32x4d-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py\
    /workspace/mmdetection/models/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/faster_rcnn/X-101-64x4d-FPN \
    --show \
    --device cuda:0




###### RetinaNet

python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py\
    /workspace/mmdetection/models/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/RetinaNet/R-50-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/retinanet/retinanet_r101_fpn_1x_coco.py \
    /workspace/mmdetection/models/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/RetinaNet/R-101-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/retinanet/retinanet_x101_32x4d_fpn_1x_coco.py\
    /workspace/mmdetection/models/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/RetinaNet/X-101-32x4d-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py \
    /workspace/mmdetection/models/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/RetinaNet/X-101-64x4d-FPN \
    --show \
    --device cuda:0






###### Cascade R-CNN


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py\
    /workspace/mmdetection/models/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/Cascade_R-CNN/R-50-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py \
    /workspace/mmdetection/models/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/Cascade_R-CNN/R-101-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco.py\
    /workspace/mmdetection/models/cascade_rcnn_x101_32x4d_fpn_1x_coco_20200316-95c2deb6.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/Cascade_R-CNN/X-101-32x4d-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco.py \
    /workspace/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/Cascade_R-CNN/X-101-64x4d-FPN \
    --show \
    --device cuda:0





###### Cascade Mask R-CNN


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py\
    /workspace/mmdetection/models/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/Cascade_Mask_R-CNN/R-50-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco.py \
    /workspace/mmdetection/models/cascade_mask_rcnn_r101_fpn_1x_coco_20200203-befdf6ee.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/Cascade_Mask_R-CNN/R-101-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco.py\
    /workspace/mmdetection/models/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco_20200201-0f411b1f.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/Cascade_Mask_R-CNN/X-101-32x4d-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco.py \
    /workspace/mmdetection/models/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/Cascade_Mask_R-CNN/X-101-64x4d-FPN \
    --show \
    --device cuda:0



###### YOLOv3

python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/yolo/yolov3_d53_320_273e_coco.py \
    /workspace/mmdetection/models/yolov3_d53_320_273e_coco-421362b6.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/YOLOv3/DarkNet-53_scale320 \
    --show \
    --device cuda:0

python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/yolo/yolov3_d53_mstrain-416_273e_coco.py \
    /workspace/mmdetection/models/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/YOLOv3/DarkNet-53_scale416 \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    /workspace/mmdetection/models/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/YOLOv3/DarkNet-53_scale608 \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco.py \
    /workspace/mmdetection/models/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/YOLOv3/MobileNetV2_scale608 \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/yolo/yolov3_mobilenetv2_320_300e_coco.py \
    /workspace/mmdetection/models/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/YOLOv3/MobileNetV2_scale608 \
    --show \
    --device cuda:0



###### FSAF


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/fsaf/fsaf_r50_fpn_1x_coco.py\
    /workspace/mmdetection/models/fsaf_r50_fpn_1x_coco-94ccc51f.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/FSAF/R-50-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/fsaf/fsaf_r101_fpn_1x_coco.py \
    /workspace/mmdetection/models/fsaf_r101_fpn_1x_coco-9e71098f.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/FSAF/R-101-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/fsaf/fsaf_x101_64x4d_fpn_1x_coco.py\
    /workspace/mmdetection/models/fsaf_x101_64x4d_fpn_1x_coco-e3f6e6fd.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/FSAF/X-101-FPN \
    --show \
    --device cuda:0



###### FreeAnchor


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py\
    /workspace/mmdetection/models/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/FreeAnchor/R-50 \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/free_anchor/retinanet_free_anchor_r101_fpn_1x_coco.py \
    /workspace/mmdetection/models/retinanet_free_anchor_r101_fpn_1x_coco_20200130-358324e6.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/FreeAnchor/R-101 \
    --show \
    --device cuda:0



python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco.py \
    /workspace/mmdetection/models/retinanet_free_anchor_x101_32x4d_fpn_1x_coco_20200130-d4846968.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/FreeAnchor/X-101-32x4d \
    --show \
    --device cuda:0



###### ATSS



python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/atss/atss_r50_fpn_1x_coco.py \
    /workspace/mmdetection/models/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/ATSS/R-50 \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/atss/atss_r101_fpn_1x_coco.py \
    /workspace/mmdetection/models/atss_r101_fpn_1x_20200825-dfcadd6f.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/ATSS/R-101 \
    --show \
    --device cuda:0




###### Dynamic R-CNN


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py \
    /workspace/mmdetection/models/dynamic_rcnn_r50_fpn_1x-62a3f276.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/Dynamic_R-CNN/R-50 \
    --show \
    --device cuda:0



###### PAA



python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/paa/paa_r50_fpn_mstrain_3x_coco.py \
    /workspace/mmdetection/models/paa_r50_fpn_mstrain_3x_coco_20210121_145722-06a6880b.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/PAA/R-50-FPN \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/paa/paa_r101_fpn_mstrain_3x_coco.py \
    /workspace/mmdetection/models/dynamic_rcnn_r50_fpn_1x-62a3f276.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/PAA/R-101-FPN \
    --show \
    --device cuda:0





###### YOLOX


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/yolox/yolox_tiny_8x8_300e_coco.py\
    /workspace/mmdetection/models/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/YOLOX/YOLOX-tiny \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py \
    /workspace/mmdetection/models/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/YOLOX/YOLOX-s \
    --show \
    --device cuda:0



python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/yolox/yolox_l_8x8_300e_coco.py \
    /workspace/mmdetection/models/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/YOLOX/YOLOX-l \
    --show \
    --device cuda:0



python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py \
    /workspace/mmdetection/models/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/YOLOX/YOLOX-x \
    --show \
    --device cuda:0


#######convnext

python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_convnext_tiny_fpn_1x_coco.py\
    /workspace/mmdetection/work_dirs/cascade_mask_rcnn_convnext/tiny/latest.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/ConvNeXt/tiny/ \
    --show \
    --device cuda:0


python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_convnext_small_fpn_1x_coco.py\
    /workspace/mmdetection/work_dirs/cascade_mask_rcnn_convnext/small/latest.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/ConvNeXt/small/ \
    --show \
    --device cuda:0

python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_convnext_base_fpn_1x_coco.py\
    /workspace/mmdetection/work_dirs/cascade_mask_rcnn_convnext/base/latest.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/ConvNeXt/base/ \
    --show \
    --device cuda:0

python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_convnext_large_fpn_1x_coco.py\
    /workspace/mmdetection/work_dirs/cascade_mask_rcnn_convnext/large/latest.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/ConvNeXt/large/ \
    --show \
    --device cuda:0

python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /workspace/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_convnext_xlarge_fpn_1x_coco.py\
    /workspace/mmdetection/work_dirs/cascade_mask_rcnn_convnext/xlarge/latest.pth \
    /workspace/mmdetection/demo/demo.jpg \
    --work-dir tensorrt/ConvNeXt/xlarge/ \
    --show \
    --device cuda:0

