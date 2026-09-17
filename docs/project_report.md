# Library Book Demand Forecasting System

## Abstract

Libraries face a recurring inventory management problem: they need to maintain sufficient copies of books to satisfy future borrowing demand while avoiding unnecessary purchases and excess stock. Traditional approaches often depend on manual observation and reactive restocking after shortages occur. This project develops a machine-learning-based Library Book Demand Forecasting System to support more data-driven inventory planning.

The system uses a synthetic library dataset containing **3,600 records representing 120 books across 30 months** because real library transaction data was unavailable. The machine learning pipeline includes data preprocessing, exploratory data analysis, leakage-safe feature engineering, chronological model evaluation, and comparison of Linear Regression, Random Forest Regressor, and XGBoost Regressor models. The target variable is the **next month's borrowed count for each book**.

The final system is integrated with an interactive Streamlit dashboard for demand forecasting, book analysis, inventory recommendations, and analytics.

Among the evaluated models, **Random Forest Regressor** was selected based on the lowest test RMSE. On the held-out test period, it achieved an **MAE of 6.1634, RMSE of 9.4191, and R² of 0.7079**.

---

# 1. Introduction

Libraries manage a large collection of books whose borrowing demand can vary over time. Some books may experience consistently high demand, while others may have seasonal changes caused by academic schedules, examinations, holidays, or changes in student requirements.

If inventory decisions are made only after shortages occur, students may not be able to access required books when demand increases.

This project develops a supervised machine learning system that predicts the next month's demand for individual library books using historical borrowing information and related features.

The system compares multiple regression algorithms using a chronological train/test strategy and provides the resulting predictions through an interactive web-based dashboard.

The system also converts predictions into practical inventory suggestions by comparing predicted demand with currently available copies. These recommendations are intended as decision-support information rather than automatic purchasing decisions.

---

# 2. Problem Statement

Libraries need to maintain adequate book inventory while avoiding unnecessary purchases and excess stock. Traditional inventory management can rely heavily on manual observation and reactive decisions, making it difficult to anticipate future demand accurately.

The problem addressed by this project is to develop a machine learning system capable of predicting the **next month's borrowing demand for each book** using historical library data.

The system should:

- Process and clean library borrowing data.
- Identify historical demand patterns.
- Generate suitable forecasting features.
- Train multiple machine learning models.
- Evaluate the models using appropriate regression metrics.
- Predict future book demand.
- Classify demand into Low, Medium, and High levels.
- Compare predicted demand with available inventory.
- Provide inventory recommendations through an interactive dashboard.

---

# 3. Objectives

The main objectives of the project are:

1. To create a structured dataset representing historical library book borrowing activity.
2. To preprocess and clean the library dataset.
3. To perform exploratory data analysis and identify demand patterns.
4. To construct leakage-safe time-series features for forecasting.
5. To predict next-month book demand using machine learning regression models.
6. To compare Linear Regression, Random Forest Regressor, and XGBoost Regressor.
7. To evaluate the models using MAE, RMSE, and R².
8. To select the model based on performance on future unseen data.
9. To classify predicted demand into Low, Medium, and High levels.
10. To generate inventory recommendations based on predicted demand and available stock.
11. To develop an interactive Streamlit dashboard for library demand analysis.
12. To provide a practical decision-support system for future library inventory planning.

---

# 4. Existing System

Traditional library inventory management often relies on:

- Manual observation.
- Librarian experience.
- Historical knowledge.
- Student requests.
- Reactive restocking.
- Simple counting of borrowed books.

Such approaches may not systematically use historical borrowing patterns to anticipate future demand.

The existing approach can therefore make it difficult to identify books that are likely to experience increased demand in upcoming months.

Another limitation is that manual analysis becomes increasingly difficult when the number of books and historical records increases.

---

# 5. Proposed System

The proposed system introduces a complete machine learning pipeline for library demand forecasting.

The system consists of the following major stages:

```text
Raw Library Data
       ↓
Data Preprocessing
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Time-Aware Train/Test Split
       ↓
Model Training
       ↓
Model Comparison
       ↓
Selected Model
       ↓
Demand Prediction
       ↓
Demand Classification
       ↓
Inventory Recommendation
       ↓
Streamlit Dashboard

# 6. Literature Survey

Demand forecasting is an important application of machine learning in inventory and supply-chain management. Research has investigated regression and ensemble learning methods for predicting future demand using historical observations and additional explanatory variables.

Machine learning approaches such as Random Forest and XGBoost can be applied to structured demand forecasting problems.

Tree-based ensemble methods are useful for structured or tabular demand data because they can model nonlinear relationships and interactions between variables.

Random Forest uses an ensemble of decision trees, while XGBoost uses gradient boosting to construct an ensemble of decision trees.

Forecasting research also emphasizes the importance of lag-based feature engineering and chronological evaluation to reduce future-data leakage.

In time-series forecasting, using future observations during training can produce unrealistic evaluation results.

These concepts motivate the methodology used in this project. Historical demand is transformed into lag and rolling features, and models are evaluated on a future hold-out period rather than through a random split.


# 7. Methodology

The proposed Library Book Demand Forecasting System follows a structured machine learning pipeline.

Raw Dataset
    ↓
Data Preprocessing
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Time-Aware Train/Test Split
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Model Comparison
    ↓
Random Forest Selection
    ↓
Demand Prediction
    ↓
Demand Classification
    ↓
Inventory Recommendation
    ↓
Streamlit Dashboard


## 7.1 Data Collection

A synthetic dataset was generated because real library transaction data was not available.

The dataset contains:

- 3,600 records
- 120 unique books
- 30 months
- 10 categories
- 16 original columns

The dataset was generated using a fixed random seed of 42 to ensure reproducibility.

The dataset is artificially generated for academic demonstration purposes and is not real library transaction data.

### Original Dataset Columns

record_id
date
book_id
book_title
author
category
department
publication_year
available_copies
borrowed_count
returned_count
renewal_count
reservation_count
semester
exam_period
holiday_indicator


## 7.2 Data Preprocessing

The preprocessing stage prepares the raw dataset for analysis and machine learning.

The main preprocessing operations include:

1. Checking for missing values.
2. Checking duplicate records.
3. Standardizing data types.
4. Converting the date column into an appropriate date format.
5. Sorting records chronologically.
6. Preparing the cleaned dataset for feature engineering.
7. Saving the processed dataset for further stages.

The processed datasets generated by the pipeline are:

data/processed/library_demand_clean.csv
data/processed/library_demand_features.csv


## 7.3 Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand borrowing patterns, category-wise demand, examination-period effects, demand distribution, and changes in book demand over time.

The EDA was performed on:

- 3,600 records
- 120 books
- 10 categories

### Overall Demand Statistics

| Metric | Value |
|---|---:|
| Total Records | 3,600 |
| Unique Books | 120 |
| Categories | 10 |
| Average Monthly Borrowed Count | 20.39 |
| Median Monthly Borrowed Count | 16.00 |

### Category-wise Demand Analysis

The category-wise analysis produced the following results:

| Category | Average Demand |
|---|---:|
| Civil | 24.61 |
| Physics | 16.29 |

The Civil category recorded the highest average demand at 24.61 books per month.

The Physics category recorded the lowest average demand at 16.29 books per month.

This shows that borrowing demand varies across academic categories.

### Examination Period Analysis

The EDA also examined borrowing behavior during examination periods.

| Period | Average Demand |
|---|---:|
| Exam Period | 28.65 |
| Non-Exam Period | 16.26 |

The average borrowing demand during examination periods was 76.2% higher than during non-examination periods.

This indicates that examination periods are an important factor associated with changes in library borrowing demand.

### Major EDA Findings

The main findings from the exploratory analysis are:

1. The average monthly borrowing demand across the dataset is 20.39 books.
2. The median monthly demand is 16 books.
3. Civil has the highest average category demand at 24.61 books/month.
4. Physics has the lowest average category demand at 16.29 books/month.
5. Average demand during examination periods is 28.65 books.
6. Average demand during non-examination periods is 16.26 books.
7. Demand during examination periods is 76.2% higher than non-examination demand.
8. Demand varies between different books and categories.
9. The observed patterns support the use of temporal and academic features in the forecasting model.

### EDA Visualizations

The following charts were generated by the project and stored in:

notebooks/eda_charts/

### Figure 1: Monthly Demand Trend

![Monthly Demand Trend](notebooks/eda_charts/01_monthly_demand_trend.png)

### Figure 2: Category-wise Average Demand

![Category Average Demand](notebooks/eda_charts/02_category_avg_demand.png)

### Figure 3: Top 10 Books by Demand

![Top 10 Books](notebooks/eda_charts/03_top10_books.png)

### Figure 4: Exam vs Non-Exam Demand

![Exam vs Non-Exam Demand](notebooks/eda_charts/04_exam_vs_nonexam.png)

### Figure 5: Demand Distribution

![Demand Distribution](notebooks/eda_charts/05_demand_distribution.png)

### Figure 6: Rising vs Falling Demand Example

![Rising vs Falling Demand](notebooks/eda_charts/06_rising_vs_falling_example.png)


## 7.4 Feature Engineering

The prediction target is:

Next month's borrowed count for each book.

The final model uses 21 features.

### Demand History Features

lag_1_borrowed
lag_2_borrowed
rolling_3_avg_borrowed

These features represent previous borrowing behavior and recent average demand.

### Time and Academic Features

month_index
exam_period
holiday_indicator
semester_encoded

These features capture temporal and academic effects.

### Inventory and Activity Features

available_copies
reservation_count
renewal_count

These features provide information about current stock and demand-related activity.

### Book Metadata

publication_year

### Category Encoding

The categorical book categories are represented using indicator features:

cat_Business Management
cat_Chemistry
cat_Civil
cat_Computer Science
cat_Data Science
cat_Electronics
cat_English Literature
cat_Mathematics
cat_Mechanical
cat_Physics

### Complete Feature List

| No. | Feature |
|---:|---|
| 1 | lag_1_borrowed |
| 2 | lag_2_borrowed |
| 3 | rolling_3_avg_borrowed |
| 4 | month_index |
| 5 | exam_period |
| 6 | holiday_indicator |
| 7 | semester_encoded |
| 8 | available_copies |
| 9 | reservation_count |
| 10 | renewal_count |
| 11 | publication_year |
| 12 | cat_Business Management |
| 13 | cat_Chemistry |
| 14 | cat_Civil |
| 15 | cat_Computer Science |
| 16 | cat_Data Science |
| 17 | cat_Electronics |
| 18 | cat_English Literature |
| 19 | cat_Mathematics |
| 20 | cat_Mechanical |
| 21 | cat_Physics |

### Leakage Prevention

Time-series forecasting requires special care to prevent information from the future entering the training data.

The system therefore uses historical lag and rolling features and evaluates models using a chronological train/test split.

The split date is:

2024-01-01

The model evaluation dataset contains:

- 2,640 training rows
- 600 testing rows
- 21 features

This chronological strategy provides a more realistic evaluation of forecasting performance because the model is evaluated on future observations rather than randomly mixed observations.


## 7.5 Model Development

Three regression models were trained and compared using a time-aware chronological train/test split.

### 7.5.1 Linear Regression

Linear Regression was used as the baseline model.

It provides a simple and interpretable approach and assumes approximately linear relationships between input variables and demand.

### 7.5.2 Random Forest Regressor

Random Forest is an ensemble of decision trees.

It can capture nonlinear relationships and interactions between features and is suitable for structured tabular data.

The Random Forest model achieved the lowest test RMSE among the three evaluated models.

### 7.5.3 XGBoost Regressor

XGBoost is a gradient-boosting algorithm that builds an ensemble of decision trees sequentially.

It can model complex nonlinear relationships and is commonly used for structured and tabular machine learning problems.


## 7.6 Evaluation Metrics

### Mean Absolute Error (MAE)

MAE represents the average absolute difference between actual demand and predicted demand.

A lower MAE indicates smaller average prediction errors.

### Root Mean Squared Error (RMSE)

RMSE measures prediction error while giving greater weight to larger errors.

A lower RMSE indicates better predictive performance, especially when larger prediction errors are important.

### R² Score

R² represents the proportion of variation in the target variable explained by the model.

A higher R² indicates that the model explains more of the variation in demand.


# 8. System Architecture

The overall architecture of the proposed system is:

                    +-------------------------+
                    |    Library Dataset      |
                    |     Synthetic Data      |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    |   Data Preprocessing    |
                    |  Cleaning & Validation  |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    |          EDA            |
                    |    Trends & Patterns    |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    |   Feature Engineering   |
                    | Lag & Rolling Features  |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    |   Time-Aware Split      |
                    |      Train / Test       |
                    +------------+------------+
                                 |
                                 v
              +--------------------------------------+
              |          Model Comparison             |
              |                                      |
              |   Linear Regression                   |
              |   Random Forest                       |
              |   XGBoost                             |
              +------------------+-------------------+
                                 |
                                 v
                    +-------------------------+
                    |    Selected Model       |
                    |   Random Forest         |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    |   Prediction Pipeline   |
                    |   Next-Month Demand     |
                    +------------+------------+
                                 |
                 +---------------+---------------+
                 |               |               |
                 v               v               v
          +-------------+ +-------------+ +-------------+
          |   Demand    | |  Inventory  | |  Analytics  |
          |Classification| |Recommendation| | Dashboard  |
          +------+------+ +------+------+ +------+------+
                 |               |               |
                 +---------------+---------------+
                                 |
                                 v
                    +-------------------------+
                    |    Streamlit Web App    |
                    +-------------------------+


# 9. Dataset Description

The dataset is synthetic and was generated exclusively for academic demonstration.

| Attribute | Value |
|---|---:|
| Total Records | 3,600 |
| Unique Books | 120 |
| Number of Months | 30 |
| Number of Categories | 10 |
| Original Columns | 16 |
| Feature Columns | 21 |
| Random Seed | 42 |
| Dataset Type | Synthetic |

### Dataset Columns

| Column | Description |
|---|---|
| record_id | Unique record identifier |
| date | Monthly observation date |
| book_id | Unique book identifier |
| book_title | Book title |
| author | Author name |
| category | Book category |
| department | Academic department |
| publication_year | Year of publication |
| available_copies | Currently available copies |
| borrowed_count | Number of books borrowed |
| returned_count | Number of returned books |
| renewal_count | Number of renewals |
| reservation_count | Number of reservations |
| semester | Academic semester |
| exam_period | Indicator for examination period |
| holiday_indicator | Indicator for holiday period |

### Dataset Distribution

The dataset contains:

- 3,600 records
- 120 books
- 30 months
- 10 categories

The dataset represents monthly observations for the library book collection over the 30-month period.


# 10. Algorithms Used

## 10.1 Linear Regression

Linear Regression is used as the baseline model.

It attempts to establish a relationship between the input features and the target demand using a linear relationship.

### Advantages

- Simple and easy to understand.
- Fast to train.
- Provides an interpretable baseline.
- Useful for identifying approximately linear relationships.

### Limitations

- Assumes approximately linear relationships.
- Less capable of capturing complex nonlinear interactions.
- May not represent complicated demand patterns effectively.


## 10.2 Random Forest Regressor

Random Forest is an ensemble learning algorithm based on multiple decision trees.

Each tree produces a prediction and the ensemble combines the tree predictions to produce the final regression output.

### Advantages

- Captures nonlinear relationships.
- Handles feature interactions.
- Suitable for structured/tabular datasets.
- Uses an ensemble of decision trees.
- Can provide feature importance information.
- Does not require a simple linear relationship between features and target.

### Limitations

- Less directly interpretable than Linear Regression.
- Can require more computational resources than basic regression.
- May show a training/test performance gap.
- Model size can increase with a large number of trees.


## 10.3 XGBoost Regressor

XGBoost is a gradient-boosting algorithm that builds decision trees sequentially.

Each new tree attempts to improve the errors made by the previous trees.

### Advantages

- Powerful gradient-boosting method.
- Effective for structured/tabular datasets.
- Can model complex nonlinear relationships.
- Provides a flexible regression framework.
- Supports advanced model tuning.

### Limitations

- Has more hyperparameters.
- Requires careful tuning.
- Can overfit when the model becomes too closely fitted to training data.
- Training and parameter tuning can be more complex.


# 11. Implementation

The system was implemented in Python using a modular machine learning pipeline.

## Project Structure

library-demand-forecasting/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── library_demand_data.csv
│   │   └── SYNTHETIC_DATA_NOTICE.txt
│   │
│   └── processed/
│       ├── library_demand_clean.csv
│       └── library_demand_features.csv
│
├── models/
│   ├── demand_model.pkl
│   ├── model_comparison.json
│   └── model_metadata.json
│
├── notebooks/
│   └── eda_charts/
│       ├── 01_monthly_demand_trend.png
│       ├── 02_category_avg_demand.png
│       ├── 03_top10_books.png
│       ├── 04_exam_vs_nonexam.png
│       ├── 05_demand_distribution.png
│       └── 06_rising_vs_falling_example.png
│
├── src/
│   ├── data_preprocessing.py
│   ├── db_utils.py
│   ├── eda.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── generate_synthetic_dataset.py
│   ├── prediction.py
│   └── train_model.py
│
├── requirements.txt
└── README.md

## Main Python Modules

### generate_synthetic_dataset.py

Generates the synthetic library dataset using a fixed random seed of 42.

### data_preprocessing.py

Performs data cleaning and prepares the dataset for analysis and feature engineering.

### eda.py

Performs exploratory data analysis and generates visualization charts.

The generated charts include:

- Monthly demand trend.
- Category average demand.
- Top 10 books.
- Exam vs non-exam demand.
- Demand distribution.
- Rising vs falling demand example.

### feature_engineering.py

Creates the forecasting features used by the machine learning models.

These include:

- Lag features.
- Rolling average demand.
- Time features.
- Examination indicators.
- Holiday indicators.
- Semester encoding.
- Inventory features.
- Reservation features.
- Renewal features.
- Publication year.
- Category encoding.

### train_model.py

Trains the machine learning models:

- Linear Regression.
- Random Forest Regressor.
- XGBoost Regressor.

### evaluate_model.py

Evaluates and compares the trained models using:

- MAE.
- RMSE.
- R².

### prediction.py

Provides a reusable prediction interface for generating next-month demand predictions for individual books.

The prediction pipeline also:

- Retrieves available book IDs.
- Loads the trained model.
- Loads model metadata.
- Generates the feature vector in the correct order.
- Predicts demand.
- Classifies demand.
- Compares predicted demand with current stock.
- Calculates additional-copy recommendations.

### app.py

Implements the Streamlit web application.

The dashboard contains:

1. Dashboard
2. Demand Forecasting
3. Book Analysis
4. Inventory Recommendation
5. Analytics


# 12. Results

The models were evaluated using a chronological test period beginning at the split date:

**2024-01-01**

The evaluation dataset contains:

- 2,640 training rows
- 600 testing rows
- 21 features

## Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 8.4094 | 11.3448 | 0.5762 |
| Random Forest | 6.1634 | 9.4191 | 0.7079 |
| XGBoost | 6.9963 | 9.9935 | 0.6712 |

## Selected Model

The selected model is:

**Random Forest Regressor**

The Random Forest model achieved the lowest test RMSE among the three evaluated models.

Its test performance was:

- MAE: 6.1634
- RMSE: 9.4191
- R²: 0.7079

### Random Forest Training Performance

| Metric | Training Value |
|---|---:|
| MAE | 4.3732 |
| RMSE | 6.1118 |
| R² | 0.8453 |

The training/test RMSE gap is **3.3073**, showing a difference between training performance and performance on the held-out future data.

### Training vs Testing Comparison

| Model | Train RMSE | Test RMSE | RMSE Gap |
|---|---:|---:|---:|
| Linear Regression | 10.7361 | 11.3448 | 0.6087 |
| Random Forest | 6.1118 | 9.4191 | 3.3073 |
| XGBoost | 4.3427 | 9.9935 | 5.6508 |

The XGBoost model has the largest training/test RMSE gap among the evaluated models.

---

## Prediction Pipeline Verification

The selected Random Forest model was successfully integrated into the prediction pipeline.

Example predictions generated by the system are:

| Book ID | Book | Predicted Demand | Demand Level | Current Stock | Additional Copies |
|---|---|---:|---|---:|---:|
| BK0001 | Introduction to Computer Science | 6.5 | Low | 67 | 0 |
| BK0002 | Fundamentals of Computer Science | 14.2 | Medium | 58 | 0 |
| BK0003 | Computer Science: A Practical Approach | 2.8 | Low | 69 | 0 |
| BK0004 | Advanced Computer Science | 18.7 | Medium | 28 | 0 |
| BK0005 | Principles of Computer Science | 6.2 | Low | 31 | 0 |

The prediction pipeline also successfully handled an invalid book ID by raising the expected `ValueError`.

---

## Demand Classification

The prediction system classifies predicted demand into three levels:

- Low
- Medium
- High

The demand cutoffs generated from the current dataset are:

| Demand Level | Condition |
|---|---|
| Low | Demand ≤ 11.00 |
| Medium | 11.00 < Demand ≤ 24.00 |
| High | Demand > 24.00 |

These thresholds are derived from the target demand distribution.

---

## Inventory Recommendation

The system compares predicted next-month demand with the current number of available copies.

The recommendation logic is:

Additional Copies = max(0, ceil(Predicted Demand - Available Copies))

If the available stock is sufficient for predicted demand, the system recommends zero additional copies.

The inventory recommendation is intended as a **data-driven decision-support suggestion**, not an automatic purchasing instruction.

---

## EDA Results Summary

| Analysis | Result |
|---|---|
| Average monthly demand | 20.39 |
| Median monthly demand | 16.00 |
| Highest-demand category | Civil |
| Civil average demand | 24.61 |
| Lowest-demand category | Physics |
| Physics average demand | 16.29 |
| Exam-period average demand | 28.65 |
| Non-exam average demand | 16.26 |
| Exam-period demand increase | 76.2% |

The exploratory analysis demonstrates that borrowing demand varies according to book category and academic period.


# 13. Advantages

The proposed system provides the following advantages:

### 1. Data-driven Decision Support

The system uses historical borrowing information to support inventory planning.

### 2. Future Demand Prediction

It predicts the expected borrowing demand for the following month.

### 3. Historical Demand Analysis

Lag features and rolling averages capture previous demand behavior.

### 4. Time-aware Evaluation

The system uses a chronological train/test split instead of randomly mixing observations.

### 5. Multiple Model Comparison

Three machine learning approaches are compared:

- Linear Regression.
- Random Forest.
- XGBoost.

### 6. Demand Classification

Predicted demand is classified into:

- Low.
- Medium.
- High.

### 7. Inventory Recommendation

The system compares predicted demand with available stock and calculates potential additional-copy requirements.

### 8. Interactive Dashboard

The Streamlit application provides an interactive interface for viewing predictions and analytics.

### 9. Modular Architecture

The project separates:

- Data generation.
- Preprocessing.
- EDA.
- Feature engineering.
- Training.
- Evaluation.
- Prediction.
- Web application.

### 10. Reproducibility

The synthetic dataset uses a fixed random seed of 42.


# 14. Limitations

The current system has several limitations:

1. The system was trained and evaluated using a **synthetic dataset**, not real library circulation data.

2. Real library data may contain different demand patterns and noise characteristics.

3. The current dataset does not represent every real-world factor affecting book demand.

4. Inventory recommendations do not currently model purchasing budget constraints.

5. Shelf capacity is not included in the recommendation logic.

6. Changes in university syllabus are not modeled.

7. Book discontinuation or replacement requirements are not modeled.

8. New book releases are not modeled.

9. The Random Forest model has a training/test RMSE gap of **3.3073**.

10. XGBoost has a larger training/test RMSE gap of **5.6508**.

11. Predictions should be treated as decision-support information rather than automatic purchasing instructions.

12. Model performance on real library data may differ from the current synthetic-data results.

13. The current system focuses primarily on monthly demand and does not provide daily or weekly forecasting.

14. External factors such as changes in student population, course enrollment, or sudden academic requirements are not included.


# 15. Future Scope

The system can be improved in several ways in future versions.

## 15.1 Real Library Data

The system can be retrained using real library circulation records.

This would allow the model to learn actual borrowing patterns.

## 15.2 Student Enrollment Data

Student enrollment information can be incorporated to improve demand prediction.

For example, books associated with courses having more students may experience greater demand.

## 15.3 Course-wise Demand Forecasting

Future versions can predict demand based on individual courses and subjects.

## 15.4 Syllabus Information

Course syllabus information can be incorporated to identify books that are likely to be required during particular semesters.

## 15.5 Department-level Forecasting

The system can be extended to forecast demand at department, subject, and course levels.

## 15.6 Automated Model Retraining

The model can be periodically retrained as new library data becomes available.

## 15.7 Model Monitoring

Future versions can monitor model performance after deployment and detect changes in prediction accuracy.

## 15.8 Prediction Uncertainty

Prediction intervals or uncertainty estimates can be added to communicate the expected range of future demand.

## 15.9 Library Management System Integration

The forecasting system can be integrated with an existing Integrated Library System (ILS).

This would allow the application to obtain updated borrowing, reservation, and inventory information automatically.

## 15.10 Automated Alerts

The system can generate alerts when:

- A book is predicted to have high demand.
- Current stock is below predicted demand.
- Demand is increasing rapidly.
- Inventory review may be required.

## 15.11 Budget-aware Inventory Planning

Future versions can consider:

- Purchasing budget.
- Shelf capacity.
- Book price.
- Supplier availability.
- Minimum and maximum stock levels.

## 15.12 Advanced Forecasting Models

With sufficient real-world data, additional forecasting algorithms can be evaluated, including advanced time-series and machine learning approaches.


# 16. Conclusion

The **Library Book Demand Forecasting System** demonstrates the application of machine learning to library demand forecasting and inventory decision support.

The project processes library-style data, performs exploratory data analysis, constructs leakage-safe forecasting features, compares three regression models, and evaluates them using a chronological hold-out period.

The dataset contains:

- **3,600 records**
- **120 books**
- **30 months**
- **10 categories**

The exploratory analysis identified several meaningful patterns.

The overall average monthly borrowing demand was **20.39 books**, while the median demand was **16 books**.

The **Civil** category recorded the highest average demand at **24.61 books per month**, while **Physics** recorded the lowest average demand at **16.29 books per month**.

A significant difference was also observed between examination and non-examination periods. Average demand during examination periods was **28.65 books**, compared with **16.26 books** during non-examination periods.

This represents a **76.2% increase** in average borrowing demand during examination periods.

Three machine learning models were evaluated:

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

The test results were:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 8.4094 | 11.3448 | 0.5762 |
| Random Forest | 6.1634 | 9.4191 | 0.7079 |
| XGBoost | 6.9963 | 9.9935 | 0.6712 |

The Random Forest model achieved the lowest test RMSE of **9.4191** among the evaluated models and was selected for the final prediction pipeline.

The final system successfully provides:

- Next-month demand predictions.
- Demand-level classification.
- Historical book analysis.
- Current stock comparison.
- Inventory recommendations.
- Library demand analytics.
- Interactive Streamlit dashboard functionality.

The prediction pipeline was also tested using sample books and successfully generated demand predictions and inventory recommendations.

However, the current dataset is synthetic and was created for academic demonstration. Therefore, the results should not be interpreted as representing actual library behavior.

The next major improvement would be to train and evaluate the system using real library circulation data. With real data and additional information such as student enrollment, course requirements, syllabus information, and inventory constraints, the system could become a more comprehensive library inventory decision-support platform.

Overall, the project demonstrates a complete machine learning workflow from **data generation and preprocessing to EDA, feature engineering, model training, evaluation, prediction, inventory recommendation, and web-based deployment**.

## 17. References
[List textbooks/papers/documentation used - Scikit-learn docs,
XGBoost docs, Streamlit docs, and any course materials, at minimum]