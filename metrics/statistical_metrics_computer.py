import torch


class StatisticalMetricsComputer:
    """
    Class for computing statistical metrics on a sufficient number of samples.
    Given neural network features with 2D shape (batch_size, feature_dim),
    for example when computing Stable rank using singular value decomposition,
    if batch_size < feature_dim, the rank is constrained by batch_size,
    so samples are accumulated until batch_size >= feature_dim.

    Computing Dormant Ratio also requires a sufficient number of samples, so the same stack is used.
    """

    def __init__(self) -> None:
        self.features = None

    def __call__(self, features: torch.Tensor) -> dict:
        assert features.dim() == 2, "Feature must be a 2D tensor"

        # Append
        if self.features is None:
            self.features = features
        else:
            self.features = torch.cat([self.features, features], dim=0)

        result_dict = {}

        # Check if enough samples have been accumulated
        if self.features.shape[0] < self.features.shape[1]:
            return result_dict

        # SVD
        singular_vals = torch.linalg.svdvals(self.features)
        normalized_singular_vals = singular_vals / (torch.sum(singular_vals) + 1e-8)

        # stable_rank := number of singular values needed for cumulative sum to exceed 99%
        threshold = 0.99
        sorted_vals = torch.sort(normalized_singular_vals, descending=True).values
        cumulative_sum = torch.cumsum(sorted_vals, dim=0)
        srank = torch.sum(cumulative_sum < threshold).item() + 1
        result_dict["stable_rank"] = srank

        # Computing entropy of normalized singular values as potentially useful metric
        entropy = -torch.sum(normalized_singular_vals * torch.log(normalized_singular_vals + 1e-8))
        result_dict["SV_entropy"] = entropy.item()

        # Compute Dormant Ratio
        dormant_ratio = self._compute_dormant_ratio(self.features)
        result_dict["dormant_ratio"] = dormant_ratio

        # Reset
        self.features = None

        return result_dict

    def _compute_dormant_ratio(self, features: torch.Tensor) -> float:
        """
        Compute Dormant Ratio

        Paper definition:
        - Compute mean of absolute values of neuron activations
        - Activation score for each neuron = mean activation of that neuron / sum of mean activations of all neurons
        - Neurons with activation score below threshold are considered dormant neurons
        - Dormant Ratio = proportion of dormant neurons

        Args:
            features: Feature tensor of shape (batch_size, feature_dim)

        Returns:
            float: Dormant Ratio (range 0-1)
        """
        # Compute mean of absolute values of activations for each neuron
        neuron_activations = torch.mean(torch.abs(features), dim=0)  # (feature_dim,)

        # Compute mean
        mean_activation = torch.mean(neuron_activations)

        # Compute activation score for each neuron (normalized)
        neuron_scores = neuron_activations / (mean_activation + 1e-8)

        # Count neurons below threshold as dormant neurons
        dormant_threshold = 0.025
        dormant_neurons = torch.sum(neuron_scores <= dormant_threshold)

        # Compute Dormant Ratio
        dormant_ratio = dormant_neurons.float() / features.shape[1]

        return dormant_ratio.item()
