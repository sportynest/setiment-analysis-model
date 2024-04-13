# Sentiment Analysis of Movie Reviews

This project delves into the application of natural language processing (NLP) techniques for classifying sentiment (positive or negative) within movie reviews.

## Key Results

* **Best Models (Tied):** 
    * Support Vector Machine (SVM) with a linear kernel
    * Support Vector Machine (SVM) with an RBF kernel found via GridSearchCV
* **Accuracy:** 89.9% (Both models achieved this accuracy) 

## Dataset

* This repository includes a small sample of the IMDB Movie Reviews Dataset (`data/sample_reviews.csv`) for demonstration purposes. 
* **To obtain the full dataset and for usage guidelines, please refer to the original source (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download).**
* **Citation:** When using this dataset (even the provided sample), please cite the following ACL 2011 paper: 
    https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib

## Dependencies

* scikit-learn
* pandas
* nltk
* numpy

Install required dependencies with: `pip install -r requirements.txt`

## Instructions

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Obtain the full IMDB dataset (if not using a sample)
4. Run `sentiment_analysis.ipynb` 

## Technical Approach

* **Preprocessing:** The text data was cleaned, tokenized, and stemmed. Stopwords were removed, and I used TF-IDF weighting for feature representation.
* **Modeling:** I experimented with Naive Bayes and Support Vector Machines with different kernels. GridSearchCV was used for hyperparameter tuning the SVM.
* **Evaluation:** The model was evaluated using accuracy and confusion matrices. After fine-tuning, an SVM with an RBF kernel achieved the highest performance.  A slight bias towards positive classifications (more false positives than false negatives) suggests an area for potential refinement. 


## Future Work

* Explore incorporating sentiment lexicons as additional features.
* Address the slight bias towards positive classifications by exploring techniques to handle class imbalance. 

## Contact
* paulsoyewo@gmail.com
* sportynest.github.io
* https://www.linkedin.com/in/lekan-soyewo/

Feel free to reach out with any questions or feedback! 