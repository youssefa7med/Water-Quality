# Water Potability Prediction

![Water Potability](https://www.bing.com/th/id/OGC.da19625ede7c23d69dae769a86777e8c?pid=1.7&rurl=https%3a%2f%2fi.pinimg.com%2foriginals%2fc4%2f1f%2f86%2fc41f861c584ddee778302aad80aab61c.gif&ehk=QAm9EIhd2rLwnYVqqDAg0y6Ba5Ev7gex0x%2bNR4dq94g%3d)

This project focuses on predicting the potability of water using machine learning models. By analyzing water quality parameters and applying multiple classification algorithms, the project aims to classify whether water is potable or not.

---

## Overview

The **Water Potability Prediction** project investigates whether water is drinkable based on features like pH levels, hardness, solids, and more. The project steps include:

- Cleaning and preprocessing the dataset.
- Applying various machine learning models to classify water potability.
- Regularizing the best-performing model to achieve optimal accuracy and robustness.

---

## Key Objectives

- Develop a reliable model for classifying water potability.
- Experiment with and compare multiple machine learning algorithms.
- Apply regularization techniques to enhance model performance.

---

## Dataset

The dataset for this project is sourced from Kaggle: [Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability?authuser=0&hl=en).

### Features
- **pH**: The pH value of water.
- **Hardness**: The hardness of water.
- **Solids**: Total dissolved solids in water.
- **Chloramines**: Chloramines levels.
- **Sulfate**: Amount of sulfate.
- **Conductivity**: Water conductivity.
- **Organic Carbon**: Organic carbon levels.
- **Trihalomethanes**: Trihalomethanes concentration.
- **Turbidity**: Water turbidity.
- **Potability**: Binary target label (1 = potable, 0 = not potable).

---

## Methodology

### Data Cleaning and Preprocessing

1. **Handling Missing Values**:
   - Imputed missing data using the mean for numerical features.

2. **Outlier Detection and Removal**:
   - Removed outliers using the Interquartile Range (IQR) method.

3. **Feature Scaling**:
   - Used **RobustScaler** to scale numerical features, handling outliers effectively.

### Model Exploration

#### 1. **Logistic Regression with Polynomial Features**
   - Enhanced feature interactions by applying polynomial feature transformations.

#### 2. **Decision Tree Classifier**
   - Captured feature interactions with interpretable tree-based logic.

#### 3. **Support Vector Classifier (SVC)**
   - Applied the RBF kernel for non-linear separability.

#### 4. **Gaussian Naive Bayes**
   - Explored probabilistic modeling for feature distributions.

#### 5. **Random Forest Classifier**
   - An ensemble method for improving generalization by averaging multiple decision trees.

#### 6. **Gradient Boosting Classifier**
   - Leveraged sequential boosting to minimize errors.

#### 7. **AdaBoost Classifier**
   - Combined weak learners iteratively for improved predictions.

### Regularization and Optimization

- Applied regularization techniques (e.g., max depth, min samples split) to Random Forest Classifier to prevent overfitting and improve generalization.

---

## Results

After comparing the models, the **Random Forest Classifier** achieved the highest accuracy and robustness, especially after applying regularization techniques. The model provided excellent precision, recall, and F1 scores, outperforming other methods.

---

## Technologies Used

- **Python**: Core programming language for analysis and modeling.
- **pandas**: Data manipulation and cleaning.
- **numpy**: Numerical computations.
- **scikit-learn**: Implementation of machine learning models and evaluation metrics.
- **matplotlib** and **seaborn**: Data visualization and exploratory analysis.

---

## Getting Started

### Prerequisites

Ensure the following Python libraries are installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/youssefa7med/WaterQuality.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd WaterQuality
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Prepare the Dataset**:
   - Place the dataset in the `data` folder.

2. **Run the Script**:
   ```bash
   python main.py
   ```

---

## Usage

- Compare the performance of different models.
- Visualize feature importance for interpretability.
- Make predictions on new data samples to classify potability.

---

## Contributing

Contributions are welcome! Fork the repository, make improvements, and submit a pull request.

---

## License

This project is licensed under the MIT License. Refer to the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Special thanks to Kaggle for providing the dataset and to the open-source community for the tools and frameworks used in this project.

