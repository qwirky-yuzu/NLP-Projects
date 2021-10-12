# The Problem

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

In Toxic Comment Challenge competition held on Kaggle, the challenge was to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate.

*Credits: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge*

# The Solution

## 1. Dataset
The dataset given contains a mixture of toxic and clean comments but as expected, there was a severe class imbalance with toxic comments making up approximately 1/8 of the entire dataset. 
There are many approaches that we can apply but they generally fall within 2 major school-of-thoughts:
1. Sampling (ROSE, SMOTE, etc)
2. Class weighting

### Bigger problems to solve.....
While it is nice if we can try out all the techniques, it is unlikely that most of us have sufficient computation resources to do it. However, if you have sufficient resources and time, please do give other methods a shot!

### The choosen one:
Due to limited computation time on Colab's free tier, I will only use the simplest method of sampling which is to balance out the number of clean and toxic comments relatively evenly manually. Surprisingly, this is sufficient to give us a reasonable baseline model! Below are just some methods which you can try out (there are a lot more ways):

|S/N:| Method | Description |
|--- | --- | --- |
|1| Simple Sampling | Sample 20,000 clean comments but extract **ALL** toxic comments. |
|2|ROSE| Naive strategy is to generate new samples by randomly sampling with replacement the current available samples
|3|SMOTE| Generate new samples in by interpolation
|4|Class Weights Adjustments (Manual)| Create a new column to classify between Toxic VS Clean comments and assign weights with formula: `(1/class) * (total/2.0)`|
|5|Class Weights Adjustments (Sklearn)| Use Sklearn class weight computation to automatically assign weights|

## 2. Model
The model used here was a BERT base (uncased) model. This is a pretrained model on an enormous corpus of English words using a masked language modeling objective. You can find more of the model's description here: https://huggingface.co/bert-base-uncased 

While training on a GPU is fast, training on TPU is even faster especially for models like BERT whose parameters can easily over a hundred million! Placing the model on the GPU would take approximate 25 minutes per epoch while on a TPU, an epoch will take less than 5 minutes to train, reducing training time by almost 80%. The only issue is that training on a TPU is scary process. However, huge thanks to Abid Ali Awan's article (check out the references section below) which provided an excellent example on training a BERT model on TPU, I was able to create the model with relative ease.

## 3. Results
- Validation Accuracy: 93.652%
- Test Accuracy: 98.341%

## 4. Reference:
1. https://www.analyticsvidhya.com/blog/2021/08/training-bert-text-classifier-on-tensor-processing-unit-tpu/
