<!DOCTYPE html>
<html lang="fr">

<body>

  <h1>Détection de Fraude par Carte de Crédit</h1>

  <h2>Description</h2>
  <p>Ce projet vise à détecter les fraudes par carte de crédit en utilisant des techniques de Machine Learning. Il exploite un ensemble de données contenant des transactions bancaires pour entraîner et évaluer des modèles de classification binaire permettant de distinguer les transactions légitimes des transactions frauduleuses. Les techniques testées incluent la régression logistique et le classifieur Naive Bayes.</p>
  <p>Les données utilisées sont issues de <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Kaggle</a> et contiennent des variables PCA ainsi que les caractéristiques <code>Time</code>, <code>Amount</code>, et <code>Class</code>.</p>

  <h2>Structure du Projet</h2>
  <ul>
    <li><code>Credit-Card-Fraud.Rmd</code> : le fichier R Markdown contenant le code complet pour la préparation, l’analyse, et la modélisation des données.</li>
    <li><code>Credit-Card-Fraud.pdf</code> : une version PDF du rapport généré par le fichier <code>.Rmd</code>, présentant les résultats détaillés de l'analyse et de la modélisation.</li>
  </ul>

  <h2>Contenu du Rapport</h2>
  <p>Le rapport présente les sections suivantes :</p>
  <ol>
    <li><strong>Introduction</strong> : Contexte et présentation des enjeux de la fraude par carte de crédit.</li>
    <li><strong>Importation des bibliothèques et du jeu de données</strong> : Chargement des bibliothèques R nécessaires et du jeu de données.</li>
    <li><strong>Exploration et Analyse des Données</strong> :
      <ul>
        <li>Exploration des données.</li>
        <li>Analyse des variables <code>Time</code>, <code>Amount</code> et <code>Class</code>.</li>
      </ul>
    </li>
    <li><strong>Manipulation des Données</strong> :
      <ul>
        <li>Séparation des données en ensembles d'entraînement et de test.</li>
        <li>Rééchantillonnage pour gérer le déséquilibre de classes.</li>
      </ul>
    </li>
    <li><strong>Modélisation des Données</strong> :
      <ul>
        <li>Implémentation des modèles de régression logistique et de Naive Bayes.</li>
        <li>Validation et comparaison des modèles avec les métriques d'exactitude, précision, rappel, F1-Score, et AUC.</li>
      </ul>
    </li>
    <li><strong>Conclusion</strong> : Synthèse des résultats et perspectives d’amélioration.</li>
  </ol>

  <h2>Reproduction du Projet</h2>
  <ol>
    <li>Cloner le dépôt GitHub :
      <pre><code>git clone https://github.com/username/credit-card-fraud-detection.git</code></pre>
    </li>
    <li>Ouvrir <code>Credit-Card-Fraud.Rmd</code> dans RStudio et exécuter chaque cellule pour reproduire l’analyse.</li>
    <li>Pour générer le fichier PDF, compiler le fichier <code>.Rmd</code> dans RStudio avec l’option "Knit to PDF".</li>
  </ol>

  <h2>Bibliothèques Utilisées</h2>
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
