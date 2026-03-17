# project goal
Develop a robust machine learning model to accurately detect and classify two specific mouse behaviors from our annotated video dataset

# Framework
In modern computer vision, analyzing animal behavior generally falls into two main paradigms for feature representation and classification:

- Direct Video Features (End-to-End).  This involves feeding raw video frames directly into spatial-temporal models (like 3D CNNs or Video Transformers) to let the network learn the visual and motion features to classify the action.
- Pose-based Classification.  This involves first locating and tracking key body parts (pose estimation) to create a time-series of coordinates, and then training a classifier on these spatial-temporal coordinate features.

# Dataset
https://drive.google.com/drive/folders/1d9R3DDxqY05nrouCFwhWC4YXDM7jWQsU?usp=drive_link

# Task
Supervised Behavior Classification. For this project, our core focus will be on the Supervised Behavior Classification task, utilizing the annotations we already have for the two target behaviors. Whether we choose to work directly from raw video features or utilize pose estimation data, we must keep in mind that existing, off-the-shelf methods might not perfectly fit the unique characteristics of our dataset or project requirements. We will need to think critically about the architecture. For this classification task, we have two main paths forward:

Establish a Baseline: We can adapt existing frameworks or standard models (like standard 3D-CNNs for video, or 1D-CNNs/LSTMs/Random Forests for coordinate data) to set an initial performance benchmark.
Build from Scratch: If we find that standard models fail to capture the specific temporal dynamics of our two actions, we should be prepared to design and train a custom architecture from the ground up.


# Literature summary
- Supervised Behavior Classification
    - Applying Deep Learning Models to Mouse Behavior Recognition (Chen et al., 2019): This paper explores the direct video feature approach, applying classic 3D CNNs (like I3D and R(2+1)D) directly to video frames for end-to-end classification.
    - SimBA (Simple Behavioral Analysis) (Nilsson et al., 2020): An excellent open-source toolkit that applies machine learning classifiers to pose-estimation data. This is a highly recommended read for understanding baseline feature engineering and classification if we go the coordinate route.
- Mouse Pose Estimation
    - DeepLabCut (Mathis et al., 2018): The foundational paper on markerless animal pose estimation using deep learning.
    - SLEAP (Pereira et al., 2022): A more recent, highly optimized framework designed for multi-animal tracking and faster inference.
- Unsupervised Behavior Discovery
    - Keypoint-MoSeq (Weinreb et al., 2023): This represents the cutting edge of computational neuroscience, using generative models (like HMMs) to automatically parse behavior into sub-second "syllables" without human labels.
