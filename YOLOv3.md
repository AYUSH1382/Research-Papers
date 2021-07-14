# YOLOv3



### Introduction

YOLO is an algorithm that uses neural networks to provide real-time object detection. This algorithm is popular because of its speed and accuracy. The model is YOLOv3,they have trained a new classifier network that’s in this model better than the other ones.YOLOv3 runs significantly faster than other detection methods with comparable performance.

### Working of model

In the bounding box prediction the system predicts bounding boxes using dimension clusters as anchor boxes. During training they have used sum of squared error loss and calculated the ground truth value using which they have predicted the objectness score. In class prediction each box predicts the classes the bounding box may contain using independent logistic classifiers instead of SoftMax.

 YOLOv3 predicts boxes at 3 different scales. They have used COCO dataset. They add several convolutional layers to process the combined feature map, and eventually predict tensors. They also use K-mean clustering to determine bounding box priors.

 They have used a new network for performing feature extraction. The network is a hybrid approach between the network used in YOLOv2 and Darknet-19 which they call Darknet-53 since it has 53 convolutional layers. Darknet-53 is better than ResNet-101 and ResNet-152. Darknet-53 achieves the highest measured floating point operations per second compared to ResNets. They have used the Darknet neural network framework for training and testing.

### Improvements in the Model

The YOLOv3 performs pretty good. Looking at the “old” detection metric of mAP at IOU= .5 (or AP~50~) YOLOv3 is very strong. It is still quite a bit behind to other models like Retina-Net in this metric though. It is almost on par with Retina-Net indicating that YOLOv3 is a very strong detector that excels at producing decent boxes for objects. The models performance drops significantly as the IOU threshold increases. The past YOLO models have struggled with aligning small objects but YOLOv3 has relatively high APS performance for that.

On the contrary, it has comparatively worse performance on medium and larger size objects.  When a plot of accuracy vs speed on the AP~50~ metric is done it is seen that YOLOv3 has significant benefits over other detection systems. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster.

###### Things they tried that didn’t work well:

1.  Anchor box x, y offset predictions - They tried using the normal anchor box prediction mechanism using a linear activation but that formulation decreased model stability.
2.  Linear activation x, y predictions instead of logistic activation - It led to a couple point drop in mAP.
3.  Focal loss - It dropped mAP about 2 points.
4.  Dual IOU thresholds and truth assignment - Faster R-CNN uses two IOU thresholds during training but in this case couldn’t get good results.

### Conclusion

YOLOv3 is a good detector. It’s fast, it’s accurate. It’s not as great on the COCO average AP between .5 and .95 IOU metric but it’s very good on the old detection metric of .5 IOU.

