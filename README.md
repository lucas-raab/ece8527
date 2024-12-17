# ece8527  
Repository for my final project from the course '[Introduction to Machine Learning and Pattern Recognition](https://isip.piconepress.com/courses/temple/ece_8527/)'

## Problem Statement  
The goal of this project is to develop a machine learning model capable of classifying electrocardiogram (EKG) signals into one of seven distinct classes, each representing a heart condition. The dataset includes the following classes:  

1. **1dAVb** (First-degree atrioventricular block)  
2. **RBBB** (Right bundle branch block)  
3. **LBBB** (Left bundle branch block)  
4. **SB** (Sinus bradycardia)  
5. **AF** (Atrial fibrillation)  
6. **ST** (Sinus tachycardia)  
7. **Healthy** (represented as a vector of zeroes)  

The challenge involved building a classification model to accurately predict the class of an input EKG signal. The primary evaluation metric for this task was **macro accuracy / macro F1 score**, although other measures such as micro F1 were also considered.  

## Results  
| Name                    | Model | Train (Acc / F1) |  Dev (Acc / F1)  | Eval (Acc / F1) |
|-------------------------|-------|------------------|------------------------|-----------------|
| Raab, Lucas (2024 Spring) | KNN   | 0.9099 / 0.2862 | 0.9009 / 0.2243        | 0.9013 / 0.2276 |
| Raab, Lucas (2024 Spring) | CNN   | 0.8860 / 0.7031 | 0.8518 / 0.6491        | 0.9044 / 0.6060 |


For more details about the project and evaluation criteria, refer to the [course exam page](https://isip.piconepress.com/courses/temple/ece_8527/exams/2024_00_spring/exam_04/).  
