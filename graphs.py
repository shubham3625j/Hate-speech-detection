
# import json
# import matplotlib.pyplot as plt


# # ==========================================
# # LOAD VALIDATION RESULTS
# # ==========================================

# with open("validation_results.json", "r") as file:
#     validation_results = json.load(file)


# # ==========================================
# # LOAD TEST RESULTS
# # ==========================================

# with open("test_results.json", "r") as file:
#     test_results = json.load(file)


# # ==========================================
# # EXTRACT METRICS
# # ==========================================

# validation_accuracy = validation_results["accuracy"] * 100
# validation_precision = validation_results["precision"] * 100
# validation_recall = validation_results["recall"] * 100
# validation_f1 = validation_results["f1"] * 100

# test_accuracy = test_results["accuracy"] * 100
# test_precision = test_results["precision"] * 100
# test_recall = test_results["recall"] * 100
# test_f1 = test_results["f1"] * 100


# # ==========================================
# # GRAPH 1
# # VALIDATION VS TEST PERFORMANCE
# # ==========================================

# metrics = [
#     "Accuracy",
#     "Hate Precision",
#     "Hate Recall",
#     "Hate F1"
# ]

# validation_values = [
#     validation_accuracy,
#     validation_precision,
#     validation_recall,
#     validation_f1
# ]

# test_values = [
#     test_accuracy,
#     test_precision,
#     test_recall,
#     test_f1
# ]


# x = range(len(metrics))
# width = 0.35

# plt.figure(figsize=(9, 5))

# bars1 = plt.bar(
#     [i - width / 2 for i in x],
#     validation_values,
#     width,
#     label="Validation"
# )

# bars2 = plt.bar(
#     [i + width / 2 for i in x],
#     test_values,
#     width,
#     label="Test"
# )

# plt.title("Validation vs Test Performance")
# plt.ylabel("Score (%)")
# plt.xticks(list(x), metrics)
# plt.ylim(0, 100)
# plt.legend()


# # Add values above bars
# for bar in bars1:
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         bar.get_height() + 1,
#         f"{bar.get_height():.2f}%",
#         ha="center"
#     )

# for bar in bars2:
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         bar.get_height() + 1,
#         f"{bar.get_height():.2f}%",
#         ha="center"
#     )


# plt.tight_layout()

# plt.savefig(
#     "validation_vs_test_performance.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# # plt.show()


# # ==========================================
# # GRAPH 2
# # FINAL TEST CONFUSION MATRIX
# # ==========================================

# cm = test_results["confusion_matrix"]

# plt.figure(figsize=(6, 5))

# plt.imshow(cm)

# plt.title("Final Test Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("Actual Label")

# plt.xticks(
#     [0, 1],
#     ["Non-Hate", "Hate"]
# )

# plt.yticks(
#     [0, 1],
#     ["Non-Hate", "Hate"]
# )


# # Add numbers inside cells
# for i in range(2):
#     for j in range(2):
#         plt.text(
#             j,
#             i,
#             cm[i][j],
#             ha="center",
#             va="center"
#         )


# plt.colorbar()

# plt.tight_layout()

# plt.savefig(
#     "test_confusion_matrix.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# # plt.show()


# # ==========================================
# # GRAPH 3
# # VALIDATION METRICS ONLY
# # ==========================================

# plt.figure(figsize=(8, 5))

# bars = plt.bar(
#     metrics,
#     validation_values
# )

# plt.title("Best Model Validation Performance")
# plt.ylabel("Score (%)")
# plt.ylim(0, 100)

# for bar in bars:
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         bar.get_height() + 1,
#         f"{bar.get_height():.2f}%",
#         ha="center"
#     )

# plt.tight_layout()

# plt.savefig(
#     "validation_performance.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# # plt.show()


# # ==========================================
# # PRINT RESULTS
# # ==========================================

# print("\n==========================================")
# print("BEST VALIDATION CONFIGURATION")
# print("==========================================")

# print(
#     "Word N-gram:",
#     tuple(validation_results["word_ngram"])
# )

# print(
#     "Character N-gram:",
#     tuple(validation_results["char_ngram"])
# )

# print(
#     "SVM C:",
#     validation_results["C"]
# )


# print("\n==========================================")
# print("VALIDATION RESULTS")
# print("==========================================")

# print(f"Accuracy:  {validation_accuracy:.2f}%")
# print(f"Precision: {validation_precision:.2f}%")
# print(f"Recall:    {validation_recall:.2f}%")
# print(f"F1 Score:  {validation_f1:.2f}%")


# print("\n==========================================")
# print("FINAL TEST RESULTS")
# print("==========================================")

# print(f"Accuracy:  {test_accuracy:.2f}%")
# print(f"Precision: {test_precision:.2f}%")
# print(f"Recall:    {test_recall:.2f}%")
# print(f"F1 Score:  {test_f1:.2f}%")


# print("\n==========================================")
# print("GRAPHS CREATED")
# print("==========================================")

# print("1. validation_performance.png")
# print("2. validation_vs_test_performance.png")
# print("3. test_confusion_matrix.png")


import json
import matplotlib.pyplot as plt


# ==========================================
# LOAD RESULTS
# ==========================================

with open("validation_results.json", "r") as file:
    validation_results = json.load(file)

with open("test_results.json", "r") as file:
    test_results = json.load(file)


# ==========================================
# EXTRACT METRICS
# ==========================================

validation_accuracy = validation_results["accuracy"] * 100
validation_precision = validation_results["precision"] * 100
validation_recall = validation_results["recall"] * 100
validation_f1 = validation_results["f1"] * 100

test_accuracy = test_results["accuracy"] * 100
test_precision = test_results["precision"] * 100
test_recall = test_results["recall"] * 100
test_f1 = test_results["f1"] * 100


metrics = [
    "Accuracy",
    "Hate Precision",
    "Hate Recall",
    "Hate F1"
]

validation_values = [
    validation_accuracy,
    validation_precision,
    validation_recall,
    validation_f1
]

test_values = [
    test_accuracy,
    test_precision,
    test_recall,
    test_f1
]


# ==========================================
# GRAPH 1
# VALIDATION PERFORMANCE
# ==========================================

plt.figure(figsize=(8, 5))

bars = plt.bar(metrics, validation_values)

plt.title("Best Model Validation Performance")
plt.ylabel("Score (%)")
plt.ylim(0, 100)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{bar.get_height():.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "validation_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# GRAPH 2
# VALIDATION VS TEST PERFORMANCE
# ==========================================

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(9, 5))

bars1 = plt.bar(
    [i - width / 2 for i in x],
    validation_values,
    width,
    label="Validation"
)

bars2 = plt.bar(
    [i + width / 2 for i in x],
    test_values,
    width,
    label="Test"
)

plt.title("Validation vs Test Performance")
plt.ylabel("Score (%)")
plt.xticks(list(x), metrics)
plt.ylim(0, 100)
plt.legend()

for bar in bars1:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{bar.get_height():.2f}%",
        ha="center"
    )

for bar in bars2:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{bar.get_height():.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "validation_vs_test_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# GRAPH 3
# FINAL TEST CONFUSION MATRIX
# ==========================================

cm = test_results["confusion_matrix"]

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Final Test Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Non-Hate", "Hate"]
)

plt.yticks(
    [0, 1],
    ["Non-Hate", "Hate"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i][j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "test_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# PRINT INFORMATION
# ==========================================

print("\n==========================================")
print("BEST VALIDATION CONFIGURATION")
print("==========================================")

print(
    "Word N-gram:",
    tuple(validation_results["word_ngram"])
)

print(
    "Character N-gram:",
    tuple(validation_results["char_ngram"])
)

print(
    "SVM C:",
    validation_results["C"]
)


print("\n==========================================")
print("VALIDATION RESULTS")
print("==========================================")

print(f"Accuracy:  {validation_accuracy:.2f}%")
print(f"Precision: {validation_precision:.2f}%")
print(f"Recall:    {validation_recall:.2f}%")
print(f"F1 Score:  {validation_f1:.2f}%")


print("\n==========================================")
print("FINAL TEST RESULTS")
print("==========================================")

print(f"Accuracy:  {test_accuracy:.2f}%")
print(f"Precision: {test_precision:.2f}%")
print(f"Recall:    {test_recall:.2f}%")
print(f"F1 Score:  {test_f1:.2f}%")


print("\n==========================================")
print("GRAPHS CREATED SUCCESSFULLY")
print("==========================================")

print("1. validation_performance.png")
print("2. validation_vs_test_performance.png")
print("3. test_confusion_matrix.png")

