# Detecting People in Artwork with CNNs

###  Introduction

Object detection has improved a lot in recent years due to a lot of factors one of them being CNNs but the cross depiction problem i.e. detecting objects regardless of how they are depicted be it in photos, painted or drawing has still been a practical challenge.

 Any model premised on a single depictive style e.g. photos will lack sufficient descriptive power for cross-depiction recognition. Therefore, an image search using methods will limit its results to photos and photo-like depictions. In they paper, they have taken into consideration natural images (photos) and non-natural images (artwork) as a whole. Since the distribution of styles is not uniform across the dataset it creates a problem for generalization. In this paper they have used a new dataset, People-Art, which contains photos, cartoons and images from 41 different artwork movements. This dataset has a single class: people. Detecting people within this dataset is a challenging task because of the huge range of ways artists depict people.

The best performance on a pre-release of the dataset was 45% average precision (AP), from a CNN that was neither trained nor fine-tuned for this task. By fine-tuning a state-of-the-art CNN for this task  they achieved 58% AP, a substantial improvement.

 The contributions made:

1. It is shown that a simple tweak for the “Fast Region-based Convolutional Network” method (Fast R-CNN), changing the criteria for negative training exemplars compared to default configuration, is key to higher performance on artwork.

2. Fine-tuning a CNN on artwork improves the performance in detecting to certain extent but it’s not the only solution since fine-tuning also give a performance of less than 60% AP.

3. The lower convolutional layers of a CNN generalize to artwork: others benefit from fine-tuning.

 They have used a state-of-the-art CNN to improve performance on a cross-depiction dataset, thereby contributing towards cross-depiction object recognition. They first explored work on deep learning for object detection and localization (largely in photos), followed work on the cross-depiction problem.

#####  Deep Learning for Object Detection and Localization

Early CNN based approaches for object localization used the sliding-window approach used by previous state-of-the-art detection systems. As CNNs became larger, and with an increased number of layers, this approach became intractable. As the size of their receptive fields increases, obtaining a precise bounding box using sliding window and non-maximal suppression became difficult. A lot of other researchers have tried to work on improving it. Szegedy et al. replaced the final layer of the CNN with a regression layer which produces a binary mask indicating whether a given pixel lies within the bounding box of an object. Girshick et al. introduced “regions with CNN features” (R-CNN), over there they have used selective search to generate possible object locations within an image. A support vector machine (SVM) classifies each region and have used a regression model to improve the accuracy of the bounding box output. One of them have improved the run-time performance by introducing SPP-net. The convolutional layers operate on the whole image, while the SPP layer pools based on the region proposal to obtain a fixed length feature vector for the fully connected layers. Girshick later introduced Fast R-CNN which improves upon R-CNN and SPP-net, thus replacing the SVM. Ren et al. used the output of the existing convolutional layers plus additional convolutional layers to predict regions, resulting in a further increase in accuracy and efficiency. Redmon et al. proposed YOLO, which operates quicker though with less accuracy than approaches.

Huang et al. proposed a system, introducing up-sampling layers to ensure the model performs better with very small and overlapping objects.

##### Cross-Depiction Detection and Matching

People have used many approaches for Cross-Depiction Detection and Matching to allow matching between colors or the model projections for that they have used wavelet decomposition of image color, Fourier analysis of the distance transforms, optimizing the CNN to minimize the distances between sketches and 3D model projections, using CNN generated features to match faces between photos and artwork. One of them used self-similarity to detect patterns but

this approach is not suitable for identifying (most) objects as a whole. Deformable Part-based Model (DPM) has been tried to perform cross-depiction matching between photographs and “artwork”this has improved the performance. This dataset is more challenging than the one used for testing, leading to a low accuracy using DPM and hence this is approach is not suitable. Zissermann et al. evaluated the performance of CNNs learnt on photos for classifying objects in paintings, showing strong performance in spite of the different domain. The same approach is used in the People-Art dataset, furthermore, it show the performance improvement when a CNN is fine-tuned for this dataset rather than simply fine-tuned on photos.

### The People-Art Dataset and its Challenges

The People-Art dataset contains images divided into 43 depiction styles. Images from 41 of these styles came from WikiArt.org while the photos came from PASCAL VOC 2012 and the cartoons from google searches.In the dataset they have labelled people which increases the total number of individual instances and thus the range of depictive styles represented.

The challeneges in the dataset :

**Range of denotational styles** is the style with which primitive marks are made (brush strokes, pencil lines, etc.), **range of projective style** includes linear camera projection, orthogonal projection, inverse perspective, and in fact a range of ad-hoc projections, range of poses is a challenge even though pose is handled by previous computer vision algorithms, it is observed that artwork, in general, exhibits a wider variety of poses than photos, overlapping, occluded and truncated people occurs in artwork as in photos, and perhaps to a greater extent.

### CNN architecture

The architecture of Fast R-CNN is used where the CNN has two inputs: an image and a set of class-agnostic rectangular region proposals. For generating region proposals they have used selective search.

The first stage of the CNN operates on the entire image. This stage consists of convolutional layers, rectified linear units (ReLUs), max-pooling layers and, in some cases, local response normalization layers. The final layer is a region of interest (ROI) pooling layer which is novel to Fast R-CNN. The output is a fixed-length feature vector formed by max-pooling of the convolution features. The feature vector is the input to the second stage of the CNN, which is fully connected. It consists of inner product and ReLU layers, as well as dropout layers (training only) aimed at preventing overfitting. The final layer is modified to output a score and bounding box prediction for only one class: person. Same approach is applied for training as Fast R-CNN, which uses stochastic gradient descent (SGD) with momentum, initializing the network with weights from the pre-trained models, in our case, trained on ImageNet. The models are fine-tuned using People-Art dataset. Three different models were tested CaffeNet, Oxford VGG’s “CNN M 1024” (VGG1024) and Oxford VGG’s “Net D” (VGG16). Each CNN’s fully connected network structure consists of two inner product layers, each followed by ReLU and dropout layers.

### Experiments

Their benchmark for validation and testing,is AP(average precision). The default configuration of Fast-RCNN defines positive ROI be region proposals whose IoU overlap with a ground-truth bounding box is at least 0.5, and defines negative ROI to be those whose overlap lies in the interval [0.1, 0.5). The cutoff between positive and negative ROI matches the definition of positive detection according the VOC detection task. They have experimented two alternative configurations for fine tuning:

a) gap - They hypothesized that ROI lying in interval [0.4, 0.6) are ambiguous and hamper training performance.

b) all-neg - They removed the lower bound for negative ROI which improves the performance on the People-art dataset. This results in the inclusion of ROI containing classes and also permits the inclusion of more artwork examples.

 Removing the lower bound on negative ROI (all-neg) results in a significant increase in performance, around a 9 percentage point increase in average precision in the best performing case. The optimal number of convolutional layers for which to fix weights to the pre-trained model, F, varies across the different training configurations, even for the same CNN. The performance falls rapidly for F ≥ 5,which suggests the first three or four convolutional layers are more transferable than later layers. The best performing CNN, VGG16, scores 58% AP, an improvement of 13 percentage points on the best previous result 45%. Training and fine-tuning a CNN on photos yields a model which overfits to photographic images. Selective Search achieves a recall rate of 98% on the People-Art test set. They fine-tuned YOLO on People-Art which results in an exploding gradient.

 There are three types of detections based on the IoU with a ground truth labelling: **Cor**, **Loc** ,**BG**. At higher thresholds, the majority of incorrect detections are caused by poor localization; at lower thresholds, background regions dominate and with perfect detection, there would be no false positives or false negatives. In some of the cases, the poor localization is caused by the presence of more than one person, which leads to the bounding box covering multiple people. In other cases, the bounding box does not cover the full extent of the person, this shows the extent to which the range of poses makes detecting people in artwork a challenging problem.

 Along with People-Art they have also worked on the Picasso Dataset. CNNs fine-tuned on photos overfit to photo, in addition, it is seen that fine-tuning results in a model which is not just better for People-Art but a dataset containing artwork which they did not train on. The best performing CNN is the smallest, suggesting that the CNNs may still be overfitting to less abstract artwork. Furthermore, the best performing method is YOLO despite being fine-tuned on photos therefore YOLO’s design is more robust to abstract forms of art.

 The ROI pooling layer captures the global structure of the person, while earlier convolutional layers only pick up the local structure. In all cases, replacing the default ROI pooling layer with a single cell max-pooling layer results in worse performance also the performance is worse than when fine-tuned on VOC 2007 with the default configuration.

### Conclusion

They have demonstrated state-of-the-art cross-depiction detection performance on the dataset, People-Art, by fine-tuning a CNN. It is seen that a CNN trained on photograph alone overfits to photos, while fine-turning on artwork allows the CNN to better generalize to other styles of artwork. The performance on the People-Art dataset, best so far, is still less than 60% AP. It is seen that the CNN often detects other mammals instead of people or makes other spurious detections and often fails to localize people correctly. The dataset contains only a subset of possible images containing people and there is a huge scope for further research.