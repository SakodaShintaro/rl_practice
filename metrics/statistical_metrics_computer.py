import torch


class StatisticalMetricsComputer:
    """
    ある程度のサンプル数に対しての統計的な指標を計算するためのクラス
    (batch_size, feature_dim)という2次元shapeのニューラルネットワークの特徴量が与えられるとして
    たとえばStable rankを求める際には特異値分解をするが、そのときbatch_size < feature_dimであると
    batch_sizeによってランクが制約されてしまうため、batch_size >= feature_dimとなるように貯める

    その他Dormant Ratioを求める場合にもある程度のサンプル数が必要になるため、同じstackを利用する
    """

    def __init__(self) -> None:
        self.features = None

    def __call__(self, features: torch.Tensor) -> dict:
        assert features.dim() == 2, "Feature must be a 2D tensor"

        # 追加
        if self.features is None:
            self.features = features
        else:
            self.features = torch.cat([self.features, features], dim=0)

        result_dict = {}

        # 十分貯まっていることのチェック
        if self.features.shape[0] < self.features.shape[1]:
            return result_dict

        # SVD
        singular_vals = torch.linalg.svdvals(self.features)
        normalized_singular_vals = singular_vals / (torch.sum(singular_vals) + 1e-8)

        # stable_rank := 累積和が99%を超えるための特異値の数
        threshold = 0.99
        sorted_vals = torch.sort(normalized_singular_vals, descending=True).values
        cumulative_sum = torch.cumsum(sorted_vals, dim=0)
        srank = torch.sum(cumulative_sum < threshold).item() + 1
        result_dict["stable_rank"] = srank

        # 正規化した特異値のエントロピーも良い指標になるのではないかと勝手に思ったので計算する
        entropy = -torch.sum(normalized_singular_vals * torch.log(normalized_singular_vals + 1e-8))
        result_dict["SV_entropy"] = entropy.item()

        # Dormant Ratio計算
        dormant_ratio = self._compute_dormant_ratio(self.features)
        result_dict["dormant_ratio"] = dormant_ratio

        # リセット
        self.features = None

        return result_dict

    def _compute_dormant_ratio(self, features: torch.Tensor) -> float:
        """
        Dormant Ratioを計算する

        論文の定義:
        - ニューロンの活性化の絶対値の平均を計算
        - 各ニューロンの活性化スコア = そのニューロンの平均活性化 / 全ニューロンの平均活性化の合計
        - 活性化スコアが閾値以下のニューロンを休眠ニューロンとする
        - Dormant Ratio = 休眠ニューロンの割合

        Args:
            features: (batch_size, feature_dim) の特徴量テンソル

        Returns:
            float: Dormant Ratio (0-1の範囲)
        """
        # 各ニューロンの活性化の絶対値の平均を計算
        neuron_activations = torch.mean(torch.abs(features), dim=0)  # (feature_dim,)

        # 全ニューロンの活性化の合計
        total_activation = torch.sum(neuron_activations)

        # 各ニューロンの活性化スコアを計算 (正規化)
        neuron_scores = neuron_activations / (total_activation + 1e-8)

        # 閾値以下のニューロンを休眠ニューロンとして数える
        dormant_threshold = 0.025
        dormant_neurons = torch.sum(neuron_scores <= dormant_threshold)

        # Dormant Ratioを計算
        dormant_ratio = dormant_neurons.float() / features.shape[1]

        return dormant_ratio.item()
