这里写一下我是如何得到文件夹内的所有名称的，通过python中的os模块，直接看代码吧。

```python
import os
filePath = '/Users/leizheng/Documents/GitHub/desktop-tutorial/breast_cancer_diagnosis/Paper_with_Code'
ans=(os.listdir(filePath))
ans.sort()
cnt=1
for item in ans:
    if '.pdf' in item:
        print(cnt,'.',item[:-4],end='')
        print()
        cnt+=1
```

```python
# 1 . An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization
# 2 . BCI Breast Cancer Immunohistochemical Image Generation through Pyramid Pix2pix
# 3 . BreastScreening  On the Use of Multi-Modality in Medical Imaging Diagnosis
# 4 . Conditional Infilling GANs for Data Augmentation in Mammogram Classification
# 5 . Curvature-based Feature Selection with Application in Classifying Electronic Health Records
# 6 . Deep Convolutional Neural Networks for Breast Cancer Histology Image Analysis
# 7 . Deep Learning to Improve Breast Cancer Early Detection on Screening Mammography
# 8 . Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening
# 9 . Detecting and classifying lesions in mammograms with Deep Learning
# 10 . Differences between human and machine perception in medical diagnosis
# 11 . High-Resolution Breast Cancer Screening with Multi-View Deep Convolutional Neural Networks
# 12 . Magnification Generalization for Histopathology Image Embedding
# 13 . O2PF  Oversampling via Optimum-Path Forest for Breast Cancer Detection
# 14 . On Breast Cancer Detection  An Application of Machine Learning Algorithms on the Wisconsin Diagnostic Dataset
# 15 . Regression Concept Vectors for Bidirectional Explanations in Histopathology
# 16 . Two-Stage Convolutional Neural Network for Breast Cancer Histology Image Classification
# 17 . Utilizing Automated Breast Cancer Detection to Identify Spatial Distributions of Tumor Infiltrating Lymphocytes in Invasive Breast Cancer

```

