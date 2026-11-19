import pandas as pd
import re
import json

from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("combined_hate_speech_dataset.csv")


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

# print("Original dataset shape:", df.shape)

# print("\nLanguage distribution:")
# print(df["language"].value_counts())


# ============================================================
# 3. KEEP ONLY HINGLISH DATA
# ============================================================

df = df[df["language"] == "hinglish"].copy()

# print("\nHinglish dataset shape:", df.shape)


# ============================================================
# 4. REMOVE DUPLICATE TEXTS
# ============================================================

# print("\nDuplicate texts in Hinglish dataset:")
# print(df.duplicated(subset=["text"]).sum())

df = df.drop_duplicates(subset=["text"])
df = df.reset_index(drop=True)


# ============================================================
# 5. CHECK HATE LABEL DISTRIBUTION
# ============================================================

# print("\nHate label distribution:")
# print(df["hate_label"].value_counts())


# ============================================================
# 6. CHECK PERCENTAGE OF EACH CLASS
# ============================================================

# print("\nHate label percentage:")
# print(
#     df["hate_label"]
#     .value_counts(normalize=True)
#     .mul(100)
#     .round(2)
# )


# ============================================================
# 7. SHOW SOME HINGLISH EXAMPLES
# ============================================================

# print("\nSample Hinglish texts:")
# print(
#     df[["text", "hate_label"]]
#     .sample(10, random_state=42)
#     .to_string(index=False)
# )


# ============================================================
# 8. FINAL DATASET SIZE
# ============================================================

# print("\nFinal dataset shape:")
# print(df.shape)


# ============================================================
# 9. SHOW EXAMPLES FROM EACH CLASS
# ============================================================

# print("\n========== LABEL 0 ==========\n")

# print(
#     df[df["hate_label"] == 0][["text", "hate_label"]]
#     .sample(20, random_state=42)
#     .to_string(index=False)
# )

# print("\n========== LABEL 1 ==========\n")

# print(
#     df[df["hate_label"] == 1][["text", "hate_label"]]
#     .sample(20, random_state=42)
#     .to_string(index=False)
# )


# ============================================================
# 10. TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    """
    Clean Hinglish text before machine learning.
    """

    # Convert text to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)


# ============================================================
# 11. COMPARE ORIGINAL AND CLEANED TEXT
# ============================================================

# print("\n========== ORIGINAL VS CLEANED TEXT ==========\n")

# for i in range(5):
#     print("Original :", df["text"].iloc[i])
#     print("Cleaned  :", df["clean_text"].iloc[i])
#     print("-" * 80)


# ============================================================
# 12. SEPARATE FEATURES AND LABELS
# ============================================================

X = df["clean_text"]
y = df["hate_label"]


# ============================================================
# 13. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# 80% training, 20% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Split remaining 20% into:
# 10% validation
# 10% testing

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# ============================================================
# 14. DISPLAY DATASET SPLIT
# ============================================================

# print("\n========== DATASET SPLIT ==========")

# print("Training   :", len(X_train))
# print("Validation :", len(X_val))
# print("Testing    :", len(X_test))

# print("\nTotal      :", len(X_train) + len(X_val) + len(X_test))

# ============================================================
# 15. STAGE 3 - TF-IDF + LOGISTIC REGRESSION
# ============================================================



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 16. CREATE MODEL PIPELINE
# ============================================================

model = Pipeline([
    
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=50000
        )
    ),
    
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# ============================================================
# 17. TRAIN MODEL
# ============================================================

# print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

# print("Training completed!")


# ============================================================
# 18. VALIDATION PREDICTIONS
# ============================================================

y_val_pred = model.predict(X_val)


# ============================================================
# 19. EVALUATION
# ============================================================

accuracy = accuracy_score(y_val, y_val_pred)

precision = precision_score(
    y_val,
    y_val_pred,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_val_pred,
    zero_division=0
)

f1 = f1_score(
    y_val,
    y_val_pred,
    zero_division=0
)


# print("\n========== VALIDATION RESULTS ==========")

# print(f"Accuracy  : {accuracy:.4f}")
# print(f"Precision : {precision:.4f}")
# print(f"Recall    : {recall:.4f}")
# print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

# print("\n========== CLASSIFICATION REPORT ==========")

# print(
#     classification_report(
#         y_val,
#         y_val_pred,
#         target_names=["Non-Hate", "Hate"],
#         digits=4
#     )
# )


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_val, y_val_pred)

# print("\n========== CONFUSION MATRIX ==========")

# print(cm)

from sklearn.svm import LinearSVC



# ============================================================
# 22. STAGE 4 - TF-IDF + LINEAR SVM
# ============================================================

svm_model = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=50000
        )
    ),

    (
        "classifier",
        LinearSVC(
            class_weight="balanced"
        )
    )
])

# ============================================================
# 23. TRAIN SVM
# ============================================================

# print("\nTraining Linear SVM model...")

svm_model.fit(X_train, y_train)

# print("SVM training completed!")

# ============================================================
# 24. SVM VALIDATION PREDICTIONS
# ============================================================

y_val_pred_svm = svm_model.predict(X_val)

# ============================================================
# 25. SVM EVALUATION
# ============================================================

accuracy_svm = accuracy_score(
    y_val,
    y_val_pred_svm
)

precision_svm = precision_score(
    y_val,
    y_val_pred_svm,
    zero_division=0
)

recall_svm = recall_score(
    y_val,
    y_val_pred_svm,
    zero_division=0
)

f1_svm = f1_score(
    y_val,
    y_val_pred_svm,
    zero_division=0
)


# print("\n========== SVM VALIDATION RESULTS ==========")

# print(f"Accuracy  : {accuracy_svm:.4f}")
# print(f"Precision : {precision_svm:.4f}")
# print(f"Recall    : {recall_svm:.4f}")
# print(f"F1 Score  : {f1_svm:.4f}")

# ============================================================
# 27. SVM CONFUSION MATRIX
# ============================================================

cm_svm = confusion_matrix(
    y_val,
    y_val_pred_svm
)

# print("\n========== SVM CONFUSION MATRIX ==========")

# print(cm_svm)


# ============================================================
# 28. CHARACTER TF-IDF + LINEAR SVM
# ============================================================

char_svm_model = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=50000
        )
    ),

    (
        "classifier",
        LinearSVC(
            class_weight="balanced"
        )
    )
])

# ============================================================
# 29. TRAIN CHARACTER SVM
# ============================================================

# print("\nTraining Character TF-IDF + Linear SVM model...")

char_svm_model.fit(X_train, y_train)

# print("Character SVM training completed!")

# ============================================================
# 30. CHARACTER SVM VALIDATION PREDICTIONS
# ============================================================

y_val_pred_char_svm = char_svm_model.predict(X_val)

# ============================================================
# 31. CHARACTER SVM EVALUATION
# ============================================================

accuracy_char_svm = accuracy_score(
    y_val,
    y_val_pred_char_svm
)

precision_char_svm = precision_score(
    y_val,
    y_val_pred_char_svm,
    zero_division=0
)

recall_char_svm = recall_score(
    y_val,
    y_val_pred_char_svm,
    zero_division=0
)

f1_char_svm = f1_score(
    y_val,
    y_val_pred_char_svm,
    zero_division=0
)


# print("\n========== CHARACTER SVM VALIDATION RESULTS ==========")

# print(f"Accuracy  : {accuracy_char_svm:.4f}")
# print(f"Precision : {precision_char_svm:.4f}")
# print(f"Recall    : {recall_char_svm:.4f}")
# print(f"F1 Score  : {f1_char_svm:.4f}")

# ============================================================
# 32. CHARACTER SVM CLASSIFICATION REPORT
# ============================================================

# print("\n========== CHARACTER SVM CLASSIFICATION REPORT ==========")

# print(
#     classification_report(
#         y_val,
#         y_val_pred_char_svm,
#         target_names=["Non-Hate", "Hate"],
#         digits=4
#     )
# )

# ============================================================
# 33. CHARACTER SVM CONFUSION MATRIX
# ============================================================

cm_char_svm = confusion_matrix(
    y_val,
    y_val_pred_char_svm
)

# print("\n========== CHARACTER SVM CONFUSION MATRIX ==========")

# print(cm_char_svm)

# ============================================================
# 34. WORD + CHARACTER TF-IDF
# ============================================================

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=50000
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=50000
)


# Fit only on training data
X_train_word = word_vectorizer.fit_transform(X_train)
X_train_char = char_vectorizer.fit_transform(X_train)


# Transform validation data
X_val_word = word_vectorizer.transform(X_val)
X_val_char = char_vectorizer.transform(X_val)


# Combine word + character features
X_train_combined = hstack([
    X_train_word,
    X_train_char
])

X_val_combined = hstack([
    X_val_word,
    X_val_char
])


# print("\n========== WORD + CHARACTER TF-IDF ==========")

# print("Word features      :", X_train_word.shape)
# print("Character features :", X_train_char.shape)
# print("Combined features  :", X_train_combined.shape)

# ============================================================
# 35. TRAIN WORD + CHARACTER SVM
# ============================================================

combined_svm = LinearSVC(
    class_weight="balanced"
)

# print("\nTraining Word + Character TF-IDF + SVM...")

combined_svm.fit(
    X_train_combined,
    y_train
)

# print("Combined SVM training completed!")

# ============================================================
# 36. COMBINED MODEL PREDICTIONS
# ============================================================

y_val_pred_combined = combined_svm.predict(
    X_val_combined
)

# ============================================================
# 37. COMBINED MODEL EVALUATION
# ============================================================

accuracy_combined = accuracy_score(
    y_val,
    y_val_pred_combined
)

precision_combined = precision_score(
    y_val,
    y_val_pred_combined,
    zero_division=0
)

recall_combined = recall_score(
    y_val,
    y_val_pred_combined,
    zero_division=0
)

f1_combined = f1_score(
    y_val,
    y_val_pred_combined,
    zero_division=0
)


# print("\n========== WORD + CHARACTER SVM RESULTS ==========")

# print(f"Accuracy  : {accuracy_combined:.4f}")
# print(f"Precision : {precision_combined:.4f}")
# print(f"Recall    : {recall_combined:.4f}")
# print(f"F1 Score  : {f1_combined:.4f}")

# ============================================================
# 38. CLASSIFICATION REPORT
# ============================================================

# print("\n========== WORD + CHARACTER CLASSIFICATION REPORT ==========")

# print(
#     classification_report(
#         y_val,
#         y_val_pred_combined,
#         target_names=["Non-Hate", "Hate"],
#         digits=4
#     )
# )

# ============================================================
# 39. CONFUSION MATRIX
# ============================================================

cm_combined = confusion_matrix(
    y_val,
    y_val_pred_combined
)

# print("\n========== WORD + CHARACTER CONFUSION MATRIX ==========")

# print(cm_combined)


# ============================================================
# 40. HYPERPARAMETER TUNING
#     WORD + CHARACTER TF-IDF + LINEAR SVM
# ============================================================

from itertools import product
from scipy.sparse import hstack


# ------------------------------------------------------------
# PARAMETERS TO TEST
# ------------------------------------------------------------

word_ngram_ranges = [
    (1, 1),
    (1, 2),
    (1, 3)
]

char_ngram_ranges = [
    (2, 5),
    (3, 5),
    (3, 6),
    (4, 6)
]

svm_C_values = [
    0.1,
    0.5,
    1,
    2,
    5
]


# ------------------------------------------------------------
# STORE RESULTS
# ------------------------------------------------------------

tuning_results = []


# ------------------------------------------------------------
# TOTAL EXPERIMENTS
# ------------------------------------------------------------

total_experiments = (
    len(word_ngram_ranges)
    * len(char_ngram_ranges)
    * len(svm_C_values)
)

# print("\n============================================================")
# print("STARTING HYPERPARAMETER TUNING")
# print("============================================================")

# print("Total configurations:", total_experiments)


experiment_number = 0


# ============================================================
# TUNING LOOP
# ============================================================

for word_ngram, char_ngram, C in product(
    word_ngram_ranges,
    char_ngram_ranges,
    svm_C_values
):

    experiment_number += 1

    # print(
    #     f"\nExperiment {experiment_number}/{total_experiments}"
    # )

    # print("Word n-gram :", word_ngram)
    # print("Char n-gram :", char_ngram)
    # print("SVM C       :", C)


    # --------------------------------------------------------
    # WORD TF-IDF
    # --------------------------------------------------------

    word_vectorizer_tune = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram,
        min_df=2,
        sublinear_tf=True,
        max_features=50000
    )


    X_train_word_tune = word_vectorizer_tune.fit_transform(
        X_train
    )

    X_val_word_tune = word_vectorizer_tune.transform(
        X_val
    )


    # --------------------------------------------------------
    # CHARACTER TF-IDF
    # --------------------------------------------------------

    char_vectorizer_tune = TfidfVectorizer(
        analyzer="char",
        ngram_range=char_ngram,
        min_df=2,
        sublinear_tf=True,
        max_features=50000
    )


    X_train_char_tune = char_vectorizer_tune.fit_transform(
        X_train
    )

    X_val_char_tune = char_vectorizer_tune.transform(
        X_val
    )


    # --------------------------------------------------------
    # COMBINE WORD + CHARACTER FEATURES
    # --------------------------------------------------------

    X_train_combined_tune = hstack([
        X_train_word_tune,
        X_train_char_tune
    ])

    X_val_combined_tune = hstack([
        X_val_word_tune,
        X_val_char_tune
    ])


    # --------------------------------------------------------
    # TRAIN SVM
    # --------------------------------------------------------

    svm_tune = LinearSVC(
        C=C,
        class_weight="balanced"
    )

    svm_tune.fit(
        X_train_combined_tune,
        y_train
    )


    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
    # --------------------------------------------------------

    y_val_pred_tune = svm_tune.predict(
        X_val_combined_tune
    )


    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    accuracy_tune = accuracy_score(
        y_val,
        y_val_pred_tune
    )

    precision_tune = precision_score(
        y_val,
        y_val_pred_tune,
        zero_division=0
    )

    recall_tune = recall_score(
        y_val,
        y_val_pred_tune,
        zero_division=0
    )

    f1_tune = f1_score(
        y_val,
        y_val_pred_tune,
        zero_division=0
    )


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    tuning_results.append({
        "word_ngram": word_ngram,
        "char_ngram": char_ngram,
        "C": C,
        "accuracy": accuracy_tune,
        "precision": precision_tune,
        "recall": recall_tune,
        "f1": f1_tune
    })


    # print(
    #     f"Accuracy: {accuracy_tune:.4f} | "
    #     f"Hate F1: {f1_tune:.4f}"
    # )


# ============================================================
# CONVERT RESULTS TO DATAFRAME
# ============================================================

results_df = pd.DataFrame(tuning_results)


# ============================================================
# SORT BY HATE F1
# ============================================================

results_df = results_df.sort_values(
    by="f1",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SHOW TOP RESULTS
# ============================================================

# print("\n============================================================")
# print("TOP HYPERPARAMETER CONFIGURATIONS")
# print("============================================================")

# print(
#     results_df.head(10).to_string(index=False)
# )


# ============================================================
# BEST CONFIGURATION
# ============================================================

best_result = results_df.iloc[0]


# ============================================================
# SAVE BEST VALIDATION RESULTS
# ============================================================

import json

validation_results = {
    "word_ngram": list(best_result["word_ngram"]),
    "char_ngram": list(best_result["char_ngram"]),
    "C": float(best_result["C"]),

    "accuracy": float(best_result["accuracy"]),
    "precision": float(best_result["precision"]),
    "recall": float(best_result["recall"]),
    "f1": float(best_result["f1"])
}

with open("validation_results.json", "w") as file:
    json.dump(validation_results, file, indent=4)

print("\nBest validation results saved to validation_results.json")


# print("\n============================================================")
# print("BEST CONFIGURATION")
# print("============================================================")

# print("Word n-gram :", best_result["word_ngram"])
# print("Char n-gram :", best_result["char_ngram"])
# print("SVM C       :", best_result["C"])

# print(f"Accuracy    : {best_result['accuracy']:.4f}")
# print(f"Precision   : {best_result['precision']:.4f}")
# print(f"Recall      : {best_result['recall']:.4f}")
# print(f"Hate F1     : {best_result['f1']:.4f}")

# ==========================================
# FINAL MODEL TRAINING + TEST EVALUATION
# ==========================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from scipy.sparse import hstack

# ------------------------------------------
# 1. Combine training + validation data
# ------------------------------------------

X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

print("Final training samples:", len(X_train_final))
print("Test samples:", len(X_test))


# ------------------------------------------
# 2. Create the selected Word TF-IDF
# ------------------------------------------

word_vectorizer_final = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=50000
)

X_train_word_final = word_vectorizer_final.fit_transform(X_train_final)
X_test_word_final = word_vectorizer_final.transform(X_test)


# ------------------------------------------
# 3. Create the selected Character TF-IDF
# ------------------------------------------

char_vectorizer_final = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),
    min_df=2,
    sublinear_tf=True,
    max_features=50000
)

X_train_char_final = char_vectorizer_final.fit_transform(X_train_final)
X_test_char_final = char_vectorizer_final.transform(X_test)


# ------------------------------------------
# 4. Combine Word + Character features
# ------------------------------------------

X_train_final_combined = hstack([
    X_train_word_final,
    X_train_char_final
])

X_test_final_combined = hstack([
    X_test_word_final,
    X_test_char_final
])

print("Final training feature shape:",
      X_train_final_combined.shape)

print("Test feature shape:",
      X_test_final_combined.shape)


# ------------------------------------------
# 5. Train the FINAL SVM
# ------------------------------------------

final_svm = LinearSVC(
    C=0.5,
    class_weight="balanced"
)

final_svm.fit(
    X_train_final_combined,
    y_train_final
)


# ------------------------------------------
# 6. Predict on untouched TEST data
# ------------------------------------------

y_test_pred = final_svm.predict(
    X_test_final_combined
)


# ------------------------------------------
# 7. Final evaluation
# ------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print("\n===== FINAL TEST RESULTS =====")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["Non-Hate", "Hate"]
    )
)


# ------------------------------------------
# 8. Confusion Matrix
# ------------------------------------------

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# SAVE FINAL TEST RESULTS
# ==========================================

test_results = {
    "accuracy": float(accuracy_score(y_test, y_test_pred)),
    "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
    "confusion_matrix": cm.tolist()
}

with open("test_results.json", "w") as file:
    json.dump(test_results, file, indent=4)

print("\nFinal test results saved to test_results.json")
