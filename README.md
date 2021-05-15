# Semantic Segmentation on Aerial Drone Images using U-Net
Spring 2021 Machine Learning Final Project  
By Muyang Xu, Chengyu Zhang, Sida Chen

Our project addresses the developing automation in aerial drone piloting and image capturing. While current automation is not yet ripe, with the manual operation still needed, we decided to run image semantic segmentation on aerial drone-captured images to automate and refine the object detection in drone operation. This project first uses U-Net and then advances to Mobile-Unet as the primary machine learning approach towards solving the problem. Our model yields satisfactory results with the implementation of the high-performance model (in both computational time and accuracy) and by fine-tuning the model and self-creating testing datasets.

**Dataset:** Semantic Drone Datasets, Institute of Computer Graphics and Vision, https://www.tugraz.at/index.php?id=22387.  
**Evaluation Data Generator:** Semantic Segmentation Editor https://github.com/Hitachi-Automotive-And-Industry-Lab/semanticsegmentation-editor.  
**Reference:**  
- Aerial Semantic Segmentation Drone Dataset. Kaggle, https://www.kaggle.com/bulentsiyah/semantic-drone-dataset
- U-Net. Olaf Ronneberger, Philipp Fischer, and Thomas Brox "U-Net: Convolutional Networks for Biomedical Image Segmentation." arXiv:1505.04597v1, 2015.https://arxiv.org/abs/1505.04597
- Mobile-Unet. Junfeng Jing, Zhen Wang, Matthias Ratsch and Huanhuan Zhang. "Mobile-Unet: An efficient convolutional neural network for fabric defect detection." In Textile Research Journal [DOI:10.1177/0040517520928604](https://journals.sagepub.com/doi/full/10.1177/0040517520928604), 2020.
- MobileNetV2. Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), arXiv:1801.04381v4, 2019.https://arxiv.org/abs/1801.04381  

## Path to our documents  
See our **Project Proprosal** [here](document/Project%20Proposal.pdf)  
See our **Project Presentatoin Slides** [here](https://docs.google.com/presentation/d/1X5dVDS3FbJU6Vf7eZQmayWUq979J6oYpvzezia4IyJo/edit?usp=sharing)  
See our **Final Paper** [here](document/Final_paper_draft.pdf)  

## Several Clarification on the files:  
Processed Dataset: [dataset](dataset)  
Self-implemented U-Net: [unet.py](unet.py)  
Main training and visualization code: [main.py](main.py)  
Trained Model Pytorch .pt file: [trained_model.pt](trained_model.pt)  
Testing Results using images found on the Internet: [val_result](val_result)  



