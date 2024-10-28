<!DOCTYPE html>
<html lang="en">

<body>

  <h1>Credit Card Fraud Detection</h1>

  <h2>Description</h2>
  <p>This project aims to detect credit card fraud using machine learning techniques. It utilizes a dataset containing bank transactions to train and evaluate binary classification models that can distinguish between legitimate and fraudulent transactions. Techniques tested include logistic regression and the Naive Bayes classifier.</p>
  <p>The dataset used is sourced from <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Kaggle</a> and contains PCA variables as well as the features <code>Time</code>, <code>Amount</code>, and <code>Class</code>.</p>

  <h2>Project Structure</h2>
  <ul>
    <li><code>Credit-Card-Fraud.Rmd</code>: The R Markdown file containing the complete code for data preparation, analysis, and modeling.</li>
    <li><code>Credit-Card-Fraud.pdf</code>: A PDF version of the report generated from the <code>.Rmd</code> file, presenting detailed analysis and modeling results.</li>
  </ul>

  <h2>Report Content</h2>
  <p>The report includes the following sections:</p>
  <ol>
    <li><strong>Introduction</strong>: Context and overview of credit card fraud challenges.</li>
    <li><strong>Importing Libraries and Dataset</strong>: Loading necessary R libraries and the dataset.</li>
    <li><strong>Data Exploration and Analysis</strong>:
      <ul>
        <li>Data exploration.</li>
        <li>Analysis of the <code>Time</code>, <code>Amount</code>, and <code>Class</code> variables.</li>
      </ul>
    </li>
    <li><strong>Data Manipulation</strong>:
      <ul>
        <li>Splitting the data into training and test sets.</li>
        <li>Resampling to address class imbalance.</li>
      </ul>
    </li>
    <li><strong>Data Modeling</strong>:
      <ul>
        <li>Implementation of logistic regression and Naive Bayes models.</li>
        <li>Validation and comparison of models using metrics such as accuracy, precision, recall, F1-Score, and AUC.</li>
      </ul>
    </li>
    <li><strong>Conclusion</strong>: Summary of results and improvement perspectives.</li>
  </ol>

  <h2>Project Reproduction</h2>
  <ol>
    <li>Clone the GitHub repository:
      <pre><code>git clone https://github.com/lahmabdel/Credit-Card-Fraud-with-R.git</code></pre>
    </li>
    <li>Open <code>Credit-Card-Fraud.Rmd</code> in RStudio and execute each cell to reproduce the analysis.</li>
    <li>To generate the PDF file, compile the <code>.Rmd</code> file in RStudio using the "Knit to PDF" option.</li>
  </ol>

  <h2>Libraries Used</h2>
  <ul>
    <li><code>tidyverse</code></li>
    <li><code>dplyr</code></li>
    <li><code>ggplot2</code></li>
    <li><code>ROSE</code></li>
    <li><code>naivebayes</code></li>
    <li><code>caret</code></li>
    <li><code>ROCR</code></li>
  </ul>

</body>
</html>

