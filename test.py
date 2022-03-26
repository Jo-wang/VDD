from scipy import stats
import torch

def _get_traversal_range(mean=0, std=1):
        """Return the corresponding traversal range in absolute terms."""
        max_traversal = 0.475

        if max_traversal < 0.5:
            max_traversal = (1 - 2 * max_traversal) / 2  # from 0.45 to 0.05
            max_traversal = stats.norm.ppf(max_traversal, loc=mean, scale=std)  # from 0.05 to -1.645

        # symmetrical traversals
        return (-1 * max_traversal, max_traversal)




n_samples=5
latent_dim=10
samples = torch.zeros(n_samples, latent_dim)
traversals = torch.linspace(*_get_traversal_range(mean=0, std=1),steps=n_samples)

print(type(traversals))
for i in range(n_samples):
    samples[i, 3] = traversals[i]