import torch

class KNearestNeighbor:
    """
    PyTorch-based KNN using squared Euclidean distance and topk.
    """

    def __init__(self, k):
        self.k = k

    def __call__(self, ref, query):
        """
        Args:
            ref: Tensor of shape (B, D, N) — reference points
            query: Tensor of shape (B, D, M) — query points
        Returns:
            inds: LongTensor of shape (B, M, k) — indices of k-NN in ref for each query
        """
        B, D, N = ref.shape
        _, _, M = query.shape

        # Transpose to (B, N, D) and (B, M, D)
        ref = ref.transpose(1, 2).contiguous()   # (B, N, D)
        query = query.transpose(1, 2).contiguous()  # (B, M, D)

        # Compute squared L2 distance
        ref_expand = ref.unsqueeze(1)    # (B, 1, N, D)
        query_expand = query.unsqueeze(2)  # (B, M, 1, D)
        dist = torch.sum((query_expand - ref_expand) ** 2, dim=3)  # (B, M, N)

        _, inds = torch.topk(dist, self.k, dim=2, largest=False, sorted=True)  # (B, M, k)
        return inds

if __name__ == "__main__":
    knn = KNearestNeighbor(k=2)

    B, D, N, M = 2, 128, 100, 1000
    ref = torch.rand(B, D, N)
    query = torch.rand(B, D, M)

    inds = knn(ref, query)
    print("KNN indices shape:", inds.shape)  # Should be (B, M, k)
