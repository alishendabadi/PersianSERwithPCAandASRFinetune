import numpy as np

def print_CV_results(label_to_category, w_accuracy, u_accuracy, loss, precision, recall, f1, accuracy_per_class, cm):
    print("*************************************************************************************************************************************")
    print("CrossValidation Finished and Results are like this:")
    print(f"Weighted Accuracy for all folds means to {np.array(w_accuracy).mean()} with Variance {np.array(w_accuracy).var()} and STD {np.array(w_accuracy).std()}")
    print(f"UnWeighted Accuracy for all folds means to {np.array(u_accuracy).mean()} with Variance {np.array(u_accuracy).var()} and STD {np.array(u_accuracy).std()}")
    print(f"Loss for all folds means to {np.array(loss).mean()} with Variance {np.array(loss).var()} and STD {np.array(loss).std()}")
    print(f"Precision for all folds means to {np.array(precision).mean()} with Variance {np.array(precision).var()} and STD {np.array(precision).std()}")
    print(f"Recall for all folds means to {np.array(recall).mean()} with Variance {np.array(recall).var()} and STD {np.array(recall).std()}")
    print(f"F1 for all folds means to {np.array(f1).mean()} with Variance {np.array(f1).var()} and STD {np.array(f1).std()}")
    for category in range(len(accuracy_per_class[0])): # each class
        accs_for_this_class = []
        for fold in accuracy_per_class:
            accs_for_this_class.append(fold[category])
        print(f"\t\tCategory {label_to_category[category]} accuracy for all folds means to {np.array(accs_for_this_class).mean()} with Variance {np.array(accs_for_this_class).var()} and STD {np.array(accs_for_this_class).std()}")
    print("*************************************************************************************************************************************")