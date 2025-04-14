# Group34_NLP_CS6120_Final_Group_Project

## Drive Link To Datasets
https://drive.google.com/drive/folders/1YZy-aY4fPOJ18skEo4UwyuUvkaQzon6d?usp=sharing

## Description

### This project analyzed spam and sentiment classification in emails using the Enron dataset. Our objective was to reliably identify spam, non-spam, positive, and negative emails to minimize harmful impacts on corporate environments. We applied unsupervised clustering (K-Means, Hierarchical) and supervised deep learning techniques (Bi-LSTM, BERT), evaluating 13 model configurations across spam detection and sentiment analysis tasks. Results showed strong spam detection accuracy, while sentiment analysis varied: VADER-based models performed strongly overall, and BART-based models provided more balanced yet less accurate sentiment classifications. In addition, we leveraged transformer-based models such as FinBERT, Twitter-RoBERTa, and DeBERTa-v3 — applied in both zero-shot and fine-tuned configurations — to improve sentiment and emotion classification. Fine-tuned ensembles demonstrated stronger generalization and better class balance, while zero-shot models offered baseline comparisons. DistilRoBERTa was also used for zero-shot emotion detection, providing qualitative insights into emotional tone across corporate emails.


## Getting Started

### Dependencies
#### This project uses the following external Python libraries:
#### * pandas – Data manipulation and analysis
#### * numpy – Numerical computations
#### * scikit-learn – Machine learning tools (preprocessing, model selection, metrics, feature extraction, decomposition, clustering, etc.)
#### * tensorflow – Deep learning framework (includes Keras for building models)
#### * keras-tuner – Hyperparameter tuning for TensorFlow/Keras models
#### * matplotlib – Data visualization and plotting
#### * seaborn – Statistical data visualization (built on top of Matplotlib)
#### * nltk – Natural language processing tools (stopword list, tokenization, sentiment analysis, etc.)
#### * transformers – State-of-the-art natural language processing models and pipelines

### Executing program
#### 1. Download [Repository](https://github.com/Yashb0299/NLP_CS6120_Final_Group_Project) to Local Environment
#### 2. Update and Install any Missing Dependencies to Environment 
#### 3. Unzip any data files required
#### * [group34_email_sentiment_results.csv](https://github.com/Yashb0299/NLP_CS6120_Final_Group_Project/blob/main/group34_email_sentiment_results.csv.zip)
#### * [group34_test_data_with_clusters.csv](https://github.com/Yashb0299/NLP_CS6120_Final_Group_Project/blob/main/group34_test_data_with_clusters.csv)
#### 4. Run Python File Associated with Each Module
# Spam and Sentiment Email Analysis: Spam Bi-LSTM Supervised Learning

# Spam Email Classification: Unsupervised Clustering and Supervised Bi-LSTM Analysis
## How to Run the Project Files
This project involves unsupervised and supervised spam detection using the Enron-Spam dataset.

## Prerequisites:
Python 3.8+

Required libraries: pandas, numpy, scikit-learn, nltk, tensorflow, matplotlib, seaborn

Jupyter Notebook or Google Colab

## Dataset Setup:
Download the dataset from this Kaggle link: https://www.kaggle.com/datasets/wanderfj/enron-spam.

Extract the archive.zip and place the archive folder in the same directory as the notebooks.

## Run Order
1. group34_unsupervised_spam.ipynb
  * This notebook performs:

  * Email loading and preprocessing (spam/ham emails from Enron 1–6)

  * TF-IDF vectorization & dimensionality reduction (SVD)

  * Clustering using K-Means and Hierarchical Clustering

  * Visualization with t-SNE

  * Saves two files:

    * group34_train_data_with_clusters.csv

    * group34_test_data_with_clusters.csv

  * Make sure these two CSVs are saved before running the next notebook.

2. group34_supervised_spam.ipynb
  * This notebook performs:

  * Loads the saved CSVs

  * Tokenizes and pads email text for LSTM input

  * Builds and evaluates a Bi-LSTM model:

    * Baseline (no cluster info)

    * With K-Means clusters

    * With Hierarchical clusters

    * With combined cluster features

  * Uses 5-fold cross-validation and performs final testing

* Includes error analysis for all model types

## Notes
* Ensure nltk.download('stopwords') and nltk.download('punkt') are run at least once. If using Google Colab, upload the archive folder or mount Google Drive. The output CSV files are reused across the two notebooks—do not delete them between runs.
  
# Sentiment Email Classification: Unsupervised Clustering and Supervised Bi-LSTM Analysis
## How to Run the Project Files
### Prerequisites
#### Python: Version 3.8 or later
#### Required Packages:
 * numpy
 * pandas
 * matplotlib
 * seaborn
 * nltk
 * transformers
 * scikit-learn
 * tensorflo
 * keras-tuner
 * ```pip install numpy pandas matplotlib seaborn nltk transformers scikit-learn tensorflow keras-tuner```
#### Data Requirements
#### Email Data:Ensure your email data is stored in an enron_email_data/ directory with subdirectories such as enron1/, enron2/, etc.
#### Labeled File: Unzip and place the email_sentiment_results.csv file (produced by running the sentiment analysis) in the repository root or update the path in the code accordingly.
### How to Run
#### To run the entire pipeline (all in one file), navigate to the repository directory in your terminal and execute:
```python group34_unsupervised_sentiment.py```
#### When executed, the script will:
#### Load and Clean Data: Scan the specified folders to load each email (.txt file) and clean it by removing headers, stopwords, numbers, punctuation, and custom words.
#### TF-IDF Vectorization and Clustering: Convert the cleaned text into TF-IDF features (up to 5000 features), normalize the data, reduce the dimensionality using PCA (to 2 dimensions), and apply KMeans clustering. The Elbow Method plot helps to choose the optimal number of clusters (set to 4 in the code). Cluster assignments and representative features for each cluster are visualized.
#### Sentiment Labeling: Use NLTK’s VADER to calculate sentiment polarity scores and use the Transformers pipeline with the facebook/bart-large-mnli model for zero-shot sentiment classification.
#### Data Preparation for Modeling: Tokenize the cleaned text, pad the sequences to a uniform length (200 tokens), and one-hot encode the sentiment labels for model training.
#### Model Building, Tuning, and Training: Define a BiLSTM model with tunable hyperparameters. Use Keras Tuner’s RandomSearch to explore 10 hyperparameter configurations based on the VADER-labeled data. Train final BiLSTM models for both the VADER-based and BART-based sentiment labels using the best configuration.
#### Model Evaluation: Evaluate the models on test sets and output final loss, accuracy, and classification metrics (precision, recall, F1-score) for each sentiment labeling method. Additionally, cross-validation is performed to assess model robustness.
### Output
#### After running the script, you will see printed outputs for:
 * TF-IDF Features Sample: A preview of the feature matrix.
 * Elbow Plot: Helps determine the optimal number of clusters.
 * Cluster Visualization: A 2D scatter plot of the PCA-reduced data colored by cluster assignments.
 * Bar Plots: Top representative features for each cluster.
 * Model Evaluation Metrics: Loss, accuracy, and detailed classification reports (using scikit-learn’s classification_report) for both the VADER-based and BART-based models.
 * Cross-Validation Results: Mean validation accuracy for both models.

# Enron Email Sentiment & Emotion Analysis 

This project performs sentiment and emotion analysis on the Enron email dataset using:

- VADER (rule-based sentiment)
- FinBERT (raw)
- Twitter RoBERTa (raw)
- Fine-tuned 5-fold DeBERTa-v3 ensemble
- Fine-tuned 5-fold Twitter-RoBERTa ensemble
- Hybrid ensemble (soft-voting: DeBERTa + Twitter-RoBERTa)
- Emotion classification (DistilRoBERTa)

## Required: Execution Order

1. Run `Fine-Tune_BERT_Model.py` to train and save the fine-tuned models  
2. Run `Inference_And_Evaluation.py` to perform sentiment and emotion analysis and generate final predictions  
3. Optionally, run `Visualization.py` to generate evaluation plots

### Required Packages

Make sure you have the following Python libraries installed:

- pandas
- numpy
- torch
- scikit-learn
- transformers
- datasets
- nltk
- matplotlib
- seaborn
- wordcloud

## NLTK Resource Setup

The required NLTK resources (`punkt`, `stopwords`, `wordnet`, `vader_lexicon`) are automatically downloaded in the code. No manual setup is needed.

Note: This project only uses the `ham/` emails (non-spam).


## How to Run

### Step 1: Fine-Tune Models (`Fine-Tune_BERT_Model.py`)

This script will:

- Load all `ham/` emails  
- Clean and use manually labeled sentiment from `Labels.csv`  
- Fine-tune:
  - microsoft/deberta-v3-base
  - cardiffnlp/twitter-roberta-base-sentiment
- Use 5-fold stratified cross-validation
- Save models to:
  - `./finetuned_deberta/fold1/`, `fold2/`, etc.
  - `./finetuned_twitter/fold1/`, etc.

---

### Step 2: Run Analysis Pipeline (`Inference_And_Evaluation.py`)

This script will:

- Load `Labels.csv` with manually labeled sentiment
- Run sentiment analysis using:
  - VADER (rule-based)
  - Raw FinBERT and Twitter-RoBERTa (zero-shot)
  - Fine-tuned 5-fold ensembles (DeBERTa, Twitter-RoBERTa)
  - Hybrid ensemble (soft-voting)
- Perform emotion classification using DistilRoBERTa
- Save results to `Results.csv`
- Print predictions for a random email example

---

### Step 3: Visualizations (`Visualization.py`)

This script automatically generates:

- Confusion matrices for each model
- Confidence vs accuracy calibration plots
- Per-class F1 score bar charts
- Word clouds for each sentiment
- Misclassified word clouds
- t-SNE projection of embeddings

All visual outputs are saved to the `figures/` directory.

## Output Example

Sample prediction output for a randomly selected email:

True Label: positive
VADER: positive | Score: 0.556
FinBERT: neutral | (precomputed)
Twitter: neutral | Conf: 0.913
Fine-tuned DeBERTa: positive | Avg Conf: 0.895
Fine-tuned Twitter: positive | Avg Conf: 0.834
Hybrid: positive | Avg Conf: 0.846
Emotion: neutral | Conf: 0.56

Full predictions saved to `Results.csv`

---

## Authors
#### [Yash Vipul Bhuptani](https://github.com/Yashb0299)
#### [Kevin Lee](https://github.com/kevinleee21)
#### [Wilson Neira](https://github.com/wilsonnexus)
#### [Matt Ray](https://github.com/MatthewjRay)

## Version History
#### V_0.1
  #### * Initial Release
