## 🚢 Titanic Survival Prediction using Machine Learning

### Project Overview
This project focuses on predicting the survival of passengers aboard the **Titanic** using various machine learning techniques. We leverage the classic Titanic dataset, which provides real-world passenger data including demographics, class, fare, and survival status, to build a predictive model.

**The Goal:** To train a **Logistic Regression** model to predict whether a passenger survived (1) or did not survive (0) the disaster based on their features.

---

### ⚙️ Technologies and Libraries

The project is built using Python and leverages the following key libraries:

* **Python 3.x**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib** & **Seaborn:** For data visualization.
* **Scikit-learn (sklearn):** For machine learning model implementation (`LogisticRegression`), model selection, and evaluation.

---

### 💾 Dataset

The dataset used is the well-known **`train.csv`** from the Kaggle Titanic competition.

| Column | Description |
| :--- | :--- |
| **PassengerId** | Unique ID of the passenger. |
| **Survived** | Survival (0 = No, 1 = Yes). This is the **Target Variable**. |
| **Pclass** | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd). |
| **Name** | Passenger's name. |
| **Sex** | Passenger's gender (male/female). |
| **Age** | Passenger's age. |
| **SibSp** | Number of siblings/spouses aboard. |
| **Parch** | Number of parents/children aboard. |
| **Ticket** | Ticket number. |
| **Fare** | Passenger fare. |
| **Cabin** | Cabin number (many missing values). |
| **Embarked** | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton). |

---

### 🚀 Methodology and Steps

The project followed a standard Machine Learning pipeline:

#### 1. Data Collection & Preprocessing
* Loaded the `train.csv` dataset into a Pandas DataFrame.
* **Handling Missing Values:**
    * The **`Cabin`** column was dropped due to a high number of missing values.
    * Missing values in the **`Age`** column were imputed using the **mean** age.
    * Missing values in the **`Embarked`** column were imputed using the **mode** (most frequent value).
* Confirmed that no missing values remained.

#### 2. Exploratory Data Analysis (EDA) & Visualization
* **Statistical Analysis:** Generated descriptive statistics (`.describe()`).
* **Survival Counts:** Visualized the number of survivors vs. non-survivors.
* **Feature Analysis:** Explored the relationship between key features and survival:
    * **Gender:** Plotted the count of male/female survivors, clearly showing that **females had a significantly higher survival rate**. 
    * **Passenger Class (Pclass):** Plotted the count of survivors across 1st, 2nd, and 3rd classes, showing that **1st class passengers had a higher survival probability**.

#### 3. Feature Engineering & Encoding
* **Categorical Encoding:** Converted categorical text columns into numerical values for model training:
    * **`Sex`**: Mapped `male` to **0** and `female` to **1**.
    * **`Embarked`**: Mapped `S` (Southampton) to **0**, `C` (Cherbourg) to **1**, and `Q` (Queenstown) to **2**.

#### 4. Model Training and Evaluation
* **Feature Selection:** Dropped irrelevant columns: `PassengerId`, `Name`, and `Ticket`.
* **Data Split:** The dataset was split into training (80%) and testing (20%) sets using `train_test_split` with `random_state=2`.
* **Model:** A **Logistic Regression** model was initialized and trained on the training data.
* **Evaluation:** The model's performance was measured using **Accuracy Score**:
    * Accuracy on Training Data: **~80.7%**
    * Accuracy on Test Data: **~78.2%** (A robust score indicating decent generalization capability on unseen data.)

---

### 🛠️ How to Run the Project

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/raj-dhawaniya/Titanic-Survival-Prediction
    cd Titanic-Survival-Prediction
    ```
2.  **Ensure you have the dataset:** Place the `train.csv` file in the appropriate directory as referenced in the script.
3.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
4.  **Execute the Script:**
    ```bash
    python Titanic\ Survival\ Prediction.py
    ```
    *Note: If you are running it in a Jupyter/Colab environment, simply run the cells sequentially.*

---
