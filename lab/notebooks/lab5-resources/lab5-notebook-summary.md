## Notebook Implementation Walkthrough

This lab notebook focuses on **Pruning** techniques to reduce the size and latency of a neural network, specifically a VGG model trained on the CIFAR-10 dataset.

### 1. Setup

*   **Initial Setup**: Installs the `torchprofile` library and imports all necessary Python libraries (`torch`, `numpy`, `matplotlib`, `torchvision`, etc.).
*   **Random Seeds**: Sets random seeds for reproducibility across `random`, `numpy`, and `torch`.
*   **Helper Functions**: Defines several utility functions:
    *   `download_url`: For fetching pretrained model checkpoints.
    *   `VGG`: The architecture of the VGG neural network.
    *   `train` and `evaluate`: Standard training and evaluation loops.
    *   `get_model_macs`, `get_sparsity`, `get_model_sparsity`, `get_num_parameters`, `get_model_size`: Functions for measuring model characteristics like computational operations (MACs), sparsity, and parameter count.
    *   `test_fine_grained_prune`: A visual test for the fine-grained pruning implementation.
*   **Load Model and Data**: Downloads and loads a pretrained VGG model and prepares the CIFAR-10 dataset for training and testing.

### 2. Dense Model Evaluation

*   **Initial Performance**: Evaluates the *dense* (unpruned) VGG model. The notebook reports an accuracy of `92.95%` and a model size of `35.20 MiB`.

### 3. Weight Distribution

*   **Visualization**: Plots histograms of weight distributions for each layer. This visually demonstrates that many weights are clustered around zero, suggesting the potential for pruning.
*   **Question 1**: Discusses the common characteristics of weight distribution (weights around zero) and how this helps in pruning (by removing small-magnitude weights).

### 4. Fine-grained Pruning

*   **`fine_grained_prune` function (Question 2)**: Implements magnitude-based fine-grained pruning. It calculates the number of zeros based on target sparsity, determines a pruning threshold, creates a binary mask, and applies it to the weight tensor. This is tested and verified.
*   **Question 3**: Adjusts target sparsity on a dummy tensor to achieve a specific number of non-zero elements, confirming understanding of the pruning mechanism.
*   **`FineGrainedPruner` class**: A class to apply and manage pruning masks across the entire model, ensuring sparsity is maintained during training.
*   **Sensitivity Scan**: This process prunes individual layers at various sparsity levels and measures the impact on accuracy. The `plot_sensitivity_scan` function visualizes these results.
    *   **Question 4.1**: Confirms that increasing sparsity generally decreases accuracy.
    *   **Question 4.2**: Notes that different layers exhibit varying sensitivities to pruning.
    *   **Question 4.3**: Identifies `backbone.conv0.weight` as the most sensitive layer.
*   **#Parameters Distribution**: A bar chart shows the distribution of parameters across different layers, highlighting which layers contribute most to the model's size.
*   **Sparsity Selection (Question 5)**: Based on sensitivity curves and parameter distribution, a `sparsity_dict` is defined to set per-layer pruning rates. The goal is to achieve a `25%` model size while maintaining high accuracy. The chosen `sparsity_dict` results in a sparse model size of `8.20 MiB` (`23.30%` of dense model) but an accuracy drop to `87.66%`.
*   **Finetuning**: The fine-grained pruned model is finetuned for 5 epochs. The pruning mask is reapplied after each training iteration to maintain sparsity. Accuracy is recovered to `92.84%` after finetuning.

### 5. Channel Pruning

*   **Restore Model**: The model is reverted to its original dense state for this section.
*   **`get_num_channels_to_keep` and `channel_prune` functions (Question 6)**: Implements the core logic for channel pruning. This involves calculating the number of channels to preserve and adjusting the `weight` tensor of convolutional layers accordingly. A sanity check confirms correct MACs reduction.
*   **Initial Channel Pruning Accuracy**: A naive uniform channel pruning (30% `prune_ratio`) leads to a significant accuracy drop to `28.14%`.
*   **Ranking Channels by Importance (Question 7)**: Implements `get_input_channel_importance` (using Frobenius norm) and `apply_channel_sorting` to sort channels by their importance. This allows for more intelligent pruning decisions. Sorting is verified to not change accuracy.
*   **Pruning with Sorting**: Channel pruning (30% ratio) after sorting channels by importance improves accuracy to `36.81%`, demonstrating the benefit of importance-based pruning.
*   **Finetuning Channel Pruned Model**: The channel-pruned model (with sorting) is finetuned for 5 epochs, recovering accuracy to `92.26%`.
*   **Measure Acceleration (Question 8)**: Compares the latency, MACs, and number of parameters between the original and channel-pruned models.
    *   **Question 8.1**: Explains why 30% channel removal leads to ~50% computation reduction (due to quadratic scaling of operations with channel count).
    *   **Question 8.2**: Discusses why latency reduction is slightly less than computation reduction (due to other factors like data movement and overheads).

### 6. Comparison of Pruning Methods

*   **Question 9**: Compares fine-grained and channel pruning.
    *   **Question 9.1**: Discusses advantages and disadvantages of each (e.g., fine-grained: better compression/accuracy but specialized hardware; channel: direct speedup on generic hardware but potentially harder accuracy recovery).
    *   **Question 9.2**: Concludes that channel pruning is generally preferred for smartphone deployment due to its direct speedup on off-the-shelf mobile hardware.
