import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


def generate_test_acc_plot(
    exp_data_dir, non_private_exp_id,
    private_exp_ids, legend_labels,
    filename):
    """
    Plot test accuracies per epoch for non-private and private models.
    """
    fig, ax = plt.subplots()

    for i, label in enumerate(legend_labels):
        if label == "Non-private":
            exp_id = non_private_exp_id
            perf_data_filepath = f"{exp_data_dir}/models/{exp_id}_non_private_perf.npz"
        else:
            exp_id = private_exp_ids[i - 1]
            perf_data_filepath = f"{exp_data_dir}/models/{exp_id}_private_perf.npz"

        data = np.load(perf_data_filepath)
        test_accs = data["test_accs"]
        ax.plot(test_accs, label=label)

    ax.set_title(f"Test Accuracy of Model")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Test Accuracy")

    ax.legend()
    plt.savefig(filename, dpi=500)

def generate_mia_roc_plot(
    exp_data_dir, non_private_exp_id, 
    private_exp_ids, legend_labels, 
    filename, focus_region=None):
    """
    Plot membership inference attack ROC curves for non-private and private models.
    """
    fig, ax = plt.subplots()

    for i, label in enumerate(legend_labels):
        if label == "Non-private":
            exp_id = non_private_exp_id
            mia_roc_data_filepath = f"{exp_data_dir}/plots_data/{exp_id}_non_private_mia_roc_data.npz"
        else:
            exp_id = private_exp_ids[i - 1]
            mia_roc_data_filepath = f"{exp_data_dir}/plots_data/{exp_id}_private_mia_roc_data.npz"

        data = np.load(mia_roc_data_filepath)
        fpr = data["fpr"]
        tpr = data["tpr"]

        ax.plot(fpr, tpr, label=label)

        if focus_region is not None:
            if focus_region == "Low FPR":
                ax.set_yscale("log")
                ax.set_xscale("log")
                ax.set_ylim([1e-4, 1])
                ax.set_xlim([1e-4, 1])
            elif focus_region == "High TPR":
                ax.set_ylim([0.7, 1])
                ax.set_xlim([0.5, 1])

    title = "Performance of Membership Inference Attack"
    if focus_region is not None:
        title = f"{title} ({focus_region} Region)"
    ax.set_title(title)

    if focus_region == "Low FPR":
        ax.set_xlabel("False Positive Rate (log-scale)")
        ax.set_ylabel("True Positive Rate (log-scale)")
    else:
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

    ax.legend()

    if focus_region is not None:
        filename = filename + "_" + str(focus_region).lower().replace(" ", "_")

    plt.savefig(filename, dpi=500)


if __name__ == "__main__":
    exp_data_dir = "exp_data"

    ############################### MODEL PERFORMANCE ###############################
    legend_labels = [
        "Non-private", 
        "Private, eps = 1.0", 
        "Private, eps = 0.5", 
        "Private, eps = 0.125"
    ]

    non_private_simple_nn_exp_id = "simple_nn_epochs5_eps0_125_delta1e_5_clipnorm1.0"
    private_simple_nn_exp_ids = [
        "simple_nn_epochs5_eps1_0_delta1e_5_clipnorm1.0",
        "simple_nn_epochs5_eps0_5_delta1e_5_clipnorm1.0",
        "simple_nn_epochs5_eps0_125_delta1e_5_clipnorm1.0",
    ]
    simple_nn_filename = f"{exp_data_dir}/plots_data/all_simple_nn_test_acc"

    generate_test_acc_plot(
        exp_data_dir=exp_data_dir, 
        non_private_exp_id=non_private_simple_nn_exp_id,
        private_exp_ids=private_simple_nn_exp_ids,
        legend_labels=legend_labels,
        filename=simple_nn_filename
    )

    non_private_cnn_exp_id = "cnn_epochs10_eps0_125_delta1e_5_clipnorm1.0"
    private_cnn_exp_ids = [
            "cnn_epochs10_eps1_0_delta1e_5_clipnorm1.0",
            "cnn_epochs10_eps0_5_delta1e_5_clipnorm1.0",
            "cnn_epochs10_eps0_125_delta1e_5_clipnorm1.0",
            ]
    cnn_filename = f"{exp_data_dir}/plots_data/all_cnn_test_acc"

    generate_test_acc_plot(
        exp_data_dir=exp_data_dir, 
        non_private_exp_id=non_private_cnn_exp_id,
        private_exp_ids=private_cnn_exp_ids,
        legend_labels=legend_labels,
        filename=cnn_filename
    )

    ################################ MIA PERFORMANCE ################################
    legend_labels = [
        "Non-private",
        "Private, eps = 1.0",
        "Private, eps = 0.5",
        "Private, eps = 0.125"
    ]

    non_private_simple_nn_exp_id = "simple_nn_epochs5_eps0_125_delta1e_5_clipnorm1.0"
    private_simple_nn_exp_ids = [
        "simple_nn_epochs5_eps1_0_delta1e_5_clipnorm1.0",
        "simple_nn_epochs5_eps0_5_delta1e_5_clipnorm1.0",
        "simple_nn_epochs5_eps0_125_delta1e_5_clipnorm1.0",
    ]
    simple_nn_filename = f"{exp_data_dir}/plots_data/all_simple_nn_mia"

    generate_mia_roc_plot(
        exp_data_dir=exp_data_dir, 
        non_private_exp_id=non_private_simple_nn_exp_id,
        private_exp_ids=private_simple_nn_exp_ids,
        legend_labels=legend_labels,
        filename=simple_nn_filename
    )

    generate_mia_roc_plot(
        exp_data_dir=exp_data_dir,
        non_private_exp_id=non_private_simple_nn_exp_id,
        private_exp_ids=private_simple_nn_exp_ids,
        legend_labels=legend_labels,
        filename=simple_nn_filename,
        focus_region="Low FPR"
    )

    generate_mia_roc_plot(
        exp_data_dir=exp_data_dir,
        non_private_exp_id=non_private_simple_nn_exp_id,
        private_exp_ids=private_simple_nn_exp_ids,
        legend_labels=legend_labels,
        filename=simple_nn_filename,
        focus_region="High TPR"
    )

    non_private_cnn_exp_id = "cnn_epochs10_eps0_125_delta1e_5_clipnorm1.0"
    private_cnn_exp_ids = [
            "cnn_epochs10_eps1_0_delta1e_5_clipnorm1.0",
            "cnn_epochs10_eps0_5_delta1e_5_clipnorm1.0",
            "cnn_epochs10_eps0_125_delta1e_5_clipnorm1.0",
            ]
    cnn_filename = f"{exp_data_dir}/plots_data/all_cnn_mia"

    generate_mia_roc_plot(
        exp_data_dir=exp_data_dir, 
        non_private_exp_id=non_private_cnn_exp_id,
        private_exp_ids=private_cnn_exp_ids,
        legend_labels=legend_labels,
        filename=cnn_filename
    )

    generate_mia_roc_plot(
        exp_data_dir=exp_data_dir,
        non_private_exp_id=non_private_cnn_exp_id,
        private_exp_ids=private_cnn_exp_ids,
        legend_labels=legend_labels,
        filename=cnn_filename,
        focus_region="Low FPR"
    )

    generate_mia_roc_plot(
        exp_data_dir=exp_data_dir,
        non_private_exp_id=non_private_cnn_exp_id,
        private_exp_ids=private_cnn_exp_ids,
        legend_labels=legend_labels,
        filename=cnn_filename,
        focus_region="High TPR"
    )
