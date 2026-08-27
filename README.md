# RajmohanKarthikeyan_NLPE_Project — Text Emotion Classifier

An end-to-end Natural Language Processing (NLP) project that classifies text inputs into six distinct emotional categories. The project implements and evaluates two deep learning methodologies:
1. **Part A**: A baseline Feed-Forward Neural Network using TF-IDF representation.
2. **Part B**: An advanced Recurrent Neural Network using Word Embeddings and a Bidirectional Long Short-Term Memory (BiLSTM) architecture.

---

## Project Structure

```text
RajmohanKarthikeyan_NLPE_Project/
├── RajmohanKarthikeyan_27_NLPE_Project/
│   ├── data/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
│   ├── NLP_Project_Brief.pdf
│   ├── RajmohanKarthikeyan_PartA.ipynb
│   ├── RajmohanKarthikeyan_PartB_BiLSTM.ipynb
│   └── Rajmohan_LearningLog.ipynb
```

---

## Dataset Overview

The dataset consists of **20,000 text emotion records** split into training, validation, and testing partitions. Each entry is structured as a semicolon-separated pair: `text;emotion`.

### Dataset Splits
- **Training Set**: 16,000 samples (80%)
- **Validation Set**: 2,000 samples (10%)
- **Test Set**: 2,000 samples (10%)

### Emotion Classes & Distribution
There are **six (6) emotional classes** labeled in the dataset: `joy`, `sadness`, `anger`, `fear`, `love`, and `surprise`.

| Emotion Class | Train Count | Val Count | Test Count | Total Count |
| :--- | :---: | :---: | :---: | :---: |
| **joy** | 5,362 | 704 | 695 | **6,761** |
| **sadness** | 4,666 | 550 | 581 | **5,797** |
| **anger** | 2,159 | 275 | 275 | **2,709** |
| **fear** | 1,937 | 212 | 224 | **2,373** |
| **love** | 1,304 | 178 | 159 | **1,641** |
| **surprise** | 572 | 81 | 66 | **719** |
| **Total** | **16,000** | **2,000** | **2,000** | **20,000** |

---

## Part A: Baseline Model (TF-IDF + Dense NN)

### 1. Preprocessing Pipeline
For each text sample, the following preprocessing steps are applied:
*   **Lowercasing**: All characters are converted to lowercase.
*   **Punctuation Removal**: Non-alphabetical characters (except spaces) are discarded.
*   **Tokenization**: Sentences are split into lists of words.
*   **Stopwords Removal**: Common grammatical words (e.g., "the", "is", "at") are removed using `nltk.corpus.stopwords`.
*   **Lemmatization**: Words are reduced to their dictionary root form (e.g., "feeling" $\rightarrow$ "feel") using `nltk.stem.WordNetLemmatizer`.

### 2. Feature Representation
*   **TF-IDF Vectorization**: Text is represented as a sparse matrix using `TfidfVectorizer` from `scikit-learn`. 
*   **Feature Limit**: `max_features` is limited to **5,000** (reduced from 10,000 to minimize feature noise and combat heavy overfitting).

### 3. Model Architecture
A sequential neural network is built with:
*   `Dense` Input Layer (128 units, ReLU activation, input shape matching the 5,000 TF-IDF features)
*   `Dropout` (rate: 0.5) to mitigate overfitting
*   `Dense` Hidden Layer (64 units, ReLU activation)
*   `Dropout` (rate: 0.5)
*   `Dense` Output Layer (6 units, Softmax activation)

### 4. Training Configuration
*   **Optimizer**: Adam
*   **Loss Function**: Categorical Crossentropy
*   **Batch Size**: 64
*   **Regularization**: Early stopping monitoring validation loss (`val_loss`) with a `patience=3` threshold. It stopped training early around epoch 6–7.

---

## Part B: Advanced Model (Bidirectional LSTM)

### 1. Preprocessing Pipeline
*   Utilizes the same cleaning pipeline (lowercasing, punctuation removal, stopword removal, lemmatization) as Part A.

### 2. Feature Representation
*   **Tokenizer**: Sentences are converted to sequences of unique integer tokens mapping to a maximum vocabulary size of **10,000 words**. Out-Of-Vocabulary (OOV) tokens are represented by `<OOV>`.
*   **Padding**: Sequences are padded/truncated to a fixed sequence length (`MAX_LEN = 50`) using post-padding.
*   **Word Embedding Layer**: An `Embedding` layer maps word integer IDs to **64-dimensional dense vectors**. These vectors are learned from scratch during training to capture context-specific semantics.

### 3. Model Architecture
A Recurrent Neural Network (RNN) designed for sequential data:
*   `Embedding` (Input: 10,000 vocabulary size, Output: 64 embedding dimensions, Sequence Length: 50)
*   `SpatialDropout1D` (rate: 0.3) - randomly drops 30% of embedding channels across all 50 token positions to prevent embedding channel memorization.
*   `Bidirectional` wrapping an `LSTM` layer (64 hidden units, recurrent dropout of 0.2, input dropout of 0.2). This produces a merged 128-dimensional output vector.
*   `Dropout` (rate: 0.4)
*   `Dense` Output Layer (6 units, Softmax activation)

### 4. Training Configuration
*   **Optimizer**: Adam
*   **Loss Function**: Categorical Crossentropy
*   **Batch Size**: 64
*   **Early Stopping**: Monitored `val_loss` with a `patience=3` threshold. The model finished training at epoch 9 and utilized `restore_best_weights=True` to automatically rewind and restore the model weights to the best epoch (epoch 6), where `val_loss` was at its minimum (~0.286).

---

## Results & Performance Comparison

### Model Performance Metrics

| Metric | Part A (TF-IDF + Dense NN) | Part B (Bidirectional LSTM) |
| :--- | :---: | :---: |
| **Test Accuracy** | **88.00%** | **90.00%** |
| **Macro Average F1-score** | 0.84 | 0.85 |
| **Weighted Average F1-score** | 0.88 | 0.90 |
| **Overfitting Severity** | Heavy (diverges after epoch 2) | Mild (train and validation lines are tightly aligned) |
| **Epochs Trained** | Stopped early (~6-7 epochs) | Stopped early (9 epochs, rewound to epoch 6) |

### Class-by-Class F1-score Comparison

| Emotion Class | Part A F1-score | Part B F1-score | Test Support |
| :--- | :---: | :---: | :---: |
| **anger** | 0.88 | **0.89** | 275 |
| **fear** | 0.83 | **0.88** | 224 |
| **joy** | 0.91 | **0.93** | 695 |
| **love** | 0.76 | **0.78** | 159 |
| **sadness** | 0.92 | **0.94** | 581 |
| **surprise** | **0.72** | 0.68 | 66 |

---

## Why Part B Performed Better

Part B achieves a higher classification accuracy and generalizes much better to unseen test data because of four core enhancements:

1.  **Word Embeddings vs. TF-IDF**: Word embeddings map words to a dense, continuous semantic space. TF-IDF represents text as orthogonal, independent dimensions, treating words like "happy" and "joyful" as completely unrelated. Word embeddings allow the model to learn that these words share a high degree of semantic overlap.
2.  **Sequential Memory vs. Bag-of-Words**: TF-IDF calculates overall term frequencies, discarding the relative order of words. The Bidirectional LSTM reads the text sequence word-by-word. This allows the model to process contextual modifiers, maintaining a crucial distinction between expressions like *"not happy"* and *"happy, not..."*.
3.  **Bidirectional Contextual Processing**: Running two LSTM chains simultaneously (one forward from start-to-end, and one backward from end-to-start) ensures that every word receives context from both past and future words. This assists the network in identifying negations, structural contrasts, and late emotional twists in sentences.
4.  **Advanced Regularization**: Part B implements three layers of regularization: `SpatialDropout1D` on embeddings, `recurrent_dropout` on LSTM state-to-state transfers, and standard `Dropout` before classification. Coupled with Early Stopping weight restoration, this dramatically reduced training memorization, resulting in a tight gap between training and validation accuracy.

---

## Setup & Installation

Follow these steps to set up the environment and run the project notebooks.

### 1. Prerequisites
Ensure you have Python 3.10+ (or Miniconda/Anaconda) installed on your system.

### 2. Create and Activate a Virtual Environment
Navigate to the project root directory in your command line:

**Using Python `venv`:**
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (Command Prompt)
venv\Scripts\activate

# Activate on Windows (PowerShell)
venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

**Using Conda:**
```bash
# Create conda environment
conda create -n nlp_env python=3.10 -y

# Activate conda environment
conda activate nlp_env
```

### 3. Install Dependencies
Install all the required python packages using the provided `requirements.txt` file:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Running the Jupyter Notebooks
Start Jupyter Lab or Notebook to execute the training files:
```bash
# Launch Jupyter Notebook
jupyter notebook
```
Open your browser and run the following notebooks in order:
1.  **`RajmohanKarthikeyan_PartA.ipynb`**: Baseline TF-IDF Neural Network.
2.  **`RajmohanKarthikeyan_PartB_BiLSTM.ipynb`**: Advanced Bidirectional LSTM Model.
3.  **`Rajmohan_LearningLog.ipynb`**: Detailed academic learning diary and pipeline analysis.

*Note: The notebooks are preconfigured to automatically check for and download the required NLTK resources (`stopwords`, `wordnet`, and `omw-1.4`) upon cell execution.*
