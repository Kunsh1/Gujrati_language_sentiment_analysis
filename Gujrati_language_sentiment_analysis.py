import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import os

# --- Imports for Oversampling and Splitting ---
from imblearn.over_sampling import RandomOverSampler # <-- Using ROS, as SMOTE is impossible on text
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# --- Updated imports for metrics and plotting ---
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Configuration ---
os.environ["WANDB_DISABLED"] = "true"
RANDOM_SEED = 42

# --- Data Configuration ---
DATASET_PATH = "gujarati-movie-review-sentiments.xlsx"
COLS_LOAD = [0, 1]
new_column_names = ['text', 'label']

# --- Label Mapping ---
ID_TO_LABEL_NAME = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
target_names = [ID_TO_LABEL_NAME[i] for i in range(len(ID_TO_LABEL_NAME))]
labels_numeric = list(ID_TO_LABEL_NAME.keys()) # [0, 1, 2]

# --- Output File Names ---
RESULTS_EXCEL_FILE = "classification_results.xlsx" #
CM_PLOT_DIR = "confusion_matrices" # 
if not os.path.exists(CM_PLOT_DIR):
    os.makedirs(CM_PLOT_DIR)

# --- 2. Load and Prepare Full Dataset ---
print(f"Loading full dataset from: {DATASET_PATH}")
try:
    full_df = pd.read_excel(
        DATASET_PATH, usecols=COLS_LOAD, header=0, names=new_column_names
    ).dropna(subset=['text', 'label'])
    full_df['label'] = full_df['label'].astype(int)
    full_df['text'] = full_df['text'].astype(str)
    print(f"✅ Full dataset loaded. Shape: {full_df.shape}")
    print(f"   Original distribution:\n{full_df['label'].value_counts().sort_index()}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# --- 3. Apply Random Oversampling ---
print(f"\nApplying Random Oversampling to entire dataset...")
X = full_df[['text']] # Must be 2D for imblearn
y = full_df['label']

ros = RandomOverSampler(random_state=RANDOM_SEED)
X_resampled_np, y_resampled = ros.fit_resample(X, y)
# Convert X_resampled back to a list of strings
X_resampled = X_resampled_np['text'].tolist()

print(f"✅ Oversampling complete.")
print(f"   New data shape: {len(X_resampled)} samples")
print(f"   New distribution:\n{y_resampled.value_counts().sort_index()}")


# --- 4. Create Stratified Train/Test Split (70/30) ---
# --- We are now splitting the data ---
print(f"\nCreating 70/30 train/test split from oversampled data...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.30,
    random_state=RANDOM_SEED,
    shuffle=True,
    stratify=y_resampled # Stratify on the new *balanced* labels
)
# Convert y_train and y_test back to lists
y_train = y_train.tolist()
y_test = y_test.tolist()

print(f"✅ Training samples: {len(y_train)}, Test samples: {len(y_test)}")
print(f"   (Note: Test set now contains synthetic data)")

# ===============================================================================
# --- 5. Define Models and Classifiers to Test ---
# ===============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

MODEL_IDS = [
    "FacebookAI/xlm-roberta-large",
    "microsoft/mdeberta-v3-base",
    "google/muril-base-cased",
    "ai4bharat/indic-bert",
    "xlm-roberta-base",
    "bert-base-multilingual-cased",
]

CLASSIFIERS = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "Gaussian Naive Bayes": GaussianNB()
}

# ===============================================================================
# --- 6. Main Loop: Iterate Through Models and Classifiers ---
# ===============================================================================

# --- Initialize list to store results ---
all_results = []

for model_id in MODEL_IDS:
    print(f"\n" + "="*80)
    print(f"--- 🚀 STARTING EVALUATION FOR MODEL: {model_id} ---")

    # --- 6a. Load the SentenceTransformer model ---
    try:
        model = SentenceTransformer(model_id, device=device)
        print(f"✅ Loaded model: {model_id}")
    except Exception as e:
        print(f"❌ Error loading {model_id}. Skipping. Error: {e}")
        continue

    # --- 6b. Generate Embeddings ---
    print(f"Generating embeddings for {len(X_train_text)} train texts...")
    X_train_embeddings = model.encode(X_train_text, show_progress_bar=True)

    print(f"Generating embeddings for {len(X_test_text)} test texts...")
    X_test_embeddings = model.encode(X_test_text, show_progress_bar=True)

    print("✅ Embeddings generated.")

    # --- SMOTE step is removed from here, as oversampling was done at the start ---

    # --- 6c. Inner Loop: Test each classifier on these embeddings ---
    for clf_name, classifier in CLASSIFIERS.items():
        print(f"\n--- Testing Classifier: {clf_name} ---")

        # Train
        print(f"Training {clf_name}...")
        classifier.fit(X_train_embeddings, y_train) # Train on the 70% split

        # --- Predict & Evaluate on Test Set ---
        print(f"\n--- Evaluation on Test Set ---")
        y_pred = classifier.predict(X_test_embeddings) # Test on the 30% split

        # --- Get Classification Report as Dictionary ---
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0, output_dict=True)
        print(f"Classification Report (Text):\n{classification_report(y_test, y_pred, target_names=target_names, zero_division=0)}")

        # --- Store Results ---
        result_entry = {
            "Model": model_id.split('/')[-1], # Get short model name
            "Classifier": clf_name,
            "Accuracy": report['accuracy'],
            "Macro Precision": report['macro avg']['precision'],
            "Macro Recall": report['macro avg']['recall'],
            "Macro F1-Score": report['macro avg']['f1-score'],
            "Weighted Precision": report['weighted avg']['precision'],
            "Weighted Recall": report['weighted avg']['recall'],
            "Weighted F1-Score": report['weighted avg']['f1-score']
        }
        all_results.append(result_entry)

        # --- Plot and Save Confusion Matrix ---
        print(f"Plotting and saving Confusion Matrix...")
        cm = confusion_matrix(y_test, y_pred, labels=labels_numeric)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')

        # Create descriptive title and filename
        plot_title = f"Confusion Matrix (Test Set)\n{model_id.split('/')[-1]}\n+ {clf_name}"
        plot_filename = f"CM_{model_id.replace('/', '_')}_{clf_name.replace(' ', '_')}.png"
        plot_path = os.path.join(CM_PLOT_DIR, plot_filename)

        plt.title(plot_title)
        plt.tight_layout() # Adjust layout
        plt.savefig(plot_path) # Save the figure
        plt.close(fig) # Close the figure to free memory (important in loops)
        print(f"   Saved confusion matrix to: {plot_path}")

    print(f"\n--- ✅ FINISHED EVALUATION FOR MODEL: {model_id} ---")

# ===============================================================================
# --- 7. Save Results to Excel ---
# ===============================================================================
print("\n" + "="*80)
print(f"💾 Saving all results to Excel file: {RESULTS_EXCEL_FILE}")

# Convert results list to DataFrame
results_df = pd.DataFrame(all_results)

# Round numeric columns for readability
numeric_cols = results_df.select_dtypes(include=np.number).columns
results_df[numeric_cols] = results_df[numeric_cols].round(4)

# Save to Excel
try:
    results_df.to_excel(RESULTS_EXCEL_FILE, index=False)
    print("✅ Results saved successfully.")
except Exception as e:
    print(f"❌ Error saving results to Excel: {e}")

print("\n✨ All model and classifier combinations evaluated and results saved. ✨")
