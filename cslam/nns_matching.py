#!/usr/bin/env python
import torch

class NearestNeighborsMatching(object):
    """Nearest Neighbor matching of description vectors
    """

    def __init__(self, dim=None):
        """Initialization

        Args:
            dim (int, optional): Global descriptor size. Defaults to None.
        """
        self.n = 0
        self.device = torch.device('cuda')
        self.dim = dim
        self.items = dict()
        self.data = []
        if dim is not None:
            self.data = torch.zeros((1000, dim), dtype=torch.float32, device=self.device)

    def add_item(self, vector, item):
        """Add item to the matching list

        Args:
            vector (np.array): descriptor
            item: identification info (e.g., int)
        """
        if self.n >= len(self.data):
            if self.dim is None:
                self.dim = len(vector)
                self.data = torch.zeros((1000, self.dim), dtype=torch.float32, device=torch.device('cuda'))
            else:
                torch.reshape(self.data, (2 * self.data.shape[0], self.dim))

        self.items[self.n] = item
        self.data[self.n] = vector.reshape(1,self.dim).to(self.device)
        self.n += 1

    def search(self, query: torch.Tensor, k):  # searching from 100000 items consume 30ms
        """Search for nearest neighbors

        Args:
            query (np.array): descriptor to match
            k (int): number of best matches to return

        Returns:
            list(int, np.array): best matches
        """
        if len(self.data) == 0:
            return [], []

        view = self.data[:self.n,:]
        qTensor = query.reshape(1, self.dim).to(self.device)
        similarity = torch.nn.functional.cosine_similarity(view, qTensor)
        ns = torch.argsort(similarity, descending=True)[:k]
        return [self.items[n] for n in ns.tolist()], torch.gather(similarity,0,ns).cpu().numpy()

    def search_best(self, query):
        """Search for the nearest neighbor

        Args:
            query (np.array): descriptor to match

        Returns:
            int, np.array: best match
        """
        if len(self.data) == 0:
            return None, None

        items, similarities = self.search(query, 1)
        return items[0], similarities[0]
