# 🏛️ Predicting Building Age & Type in Lombardy – ResNet50 & Multitask Learning  
This project trains a convolutional neural network to **predict the construction year** (regression) and **categorize the general typology** (classification) of buildings in Lombardy, Italy, based on façade images. The approach combines data scraping, preprocessing, transfer learning with ResNet50, and gradient-based interpretability techniques.

## 📁 Files  
- `scrape.ipynb` – Web scraping pipeline using Selenium to extract ~15k buildings from lombardiabeniculturali.it  
- `preprocess.ipynb` – Data cleaning and transformation (e.g., Roman numeral century conversion, date averaging)  
- `RESNET.ipynb` – Multitask model implementation using ResNet50 backbone  
- `download.ipynb` – Utility for dataset retrieval  
- `df_dates.csv` – Preprocessed tabular dataset including images, construction year, and typology  
- `REPORT.pdf` – Full report detailing methodology, experimental results, and GradCAM visualizations  

## 📊 Summary  
### 1. Data Pipeline  
- Scraped over 15,000 building entries from Lombardy’s cultural heritage archive  
- Filtered to ~9,800 valid entries with reliable construction dates and typology  
- Converted textual date ranges and centuries into numerical values  

### 2. Multitask Learning Model (CNN)  
- Used **ResNet50** as the feature extractor  
- Built two task-specific heads:  
  - **Regression head** to predict construction year  
  - **Classification head** to assign one of six general building types  
- Implemented training and fine-tuning phases with dropout, batch normalization, and early stopping  

### 3. Gradient-Based Interpretability  
- Applied **Grad-CAM** to visualize which regions of each image activated the network’s predictions  
- Classification focus was often on windows and façade structure  
- Regression attention maps were more diffuse, showing the difficulty of the task  

### 4. Key Findings  
- Achieved **mean absolute error of ~84 years** for predicting construction date  
- **Classification F1 score improved** after fine-tuning, despite class imbalance  
  → Window structure emerged as a key visual cue  
  → Photos with clear, centered façades led to better performance  

## 🔧 Tools & Libraries  
- Python (TensorFlow/Keras, NumPy, pandas, Selenium, OpenCV, matplotlib)  
- ResNet50 pretrained on ImageNet for feature extraction  
- Custom Grad-CAM implementation for model explainability  

## 📌 Notes  
- The dataset is class-imbalanced and includes noisy images (e.g., trees, other buildings)  
- Regression on architectural style is inherently difficult due to visual ambiguity over decades  
- Multitask approach showed promise but could be improved with more granular labels or street-view data  

## 📚 References  
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition  
- https://www.lombardiabeniculturali.it/architetture/tipologie/  

---  
🏗️ This project was completed as part of a Deep Learning for Computer Vision course.
