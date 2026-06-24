import numpy as np


class MertonJumpDiffusion:
    def __init__(
        self,
        S_0: float,
        dt: int,
        mu: float,
        vol: float,
        mu_jump: float,
        sigma_jump: float,
        poiss_lambda: float,
    ) -> None:
        # General params
        self.S_0 = S_0
        self.dt = dt  # timestep

        # Brownian motion component params
        self.mu = mu  # interest rate (determines drift)
        self.vol = vol  # volatility (constant)

        # Poisson jumps params
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.k = (
            np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1
        )  # Expectation E[Y_j - 1], drift correction
        self.poiss_lambda = (
            poiss_lambda  # lambda determining jump arrival Poisson process
        )

    def _simulate_path(self, n_steps):
        # Initialize numpy array to store the simulated path
        path = np.zeros(n_steps)
        path[0] = np.log(
            self.S_0
        )  # Class expect nominal S_0, but this function assumes log price

        # Brownian Motion component
        W = np.random.normal(0, self.vol * np.sqrt(self.dt), size=n_steps - 1)

        # Draw number of jumps for each element in the path. This follows Poisson(poiss_lambda) distribution
        n_jumps = np.random.poisson(
            self.poiss_lambda * self.dt, n_steps - 1
        )  # n_steps - 1 because we have jumps only between each 2 timestamps
        jump_sizes = np.array(
            [
                np.random.normal(n * self.mu_jump, np.sqrt(n) * self.sigma_jump)
                for n in n_jumps
            ]
        )  # Sizes of log(jump) - this follows normal distribution

        # Total component
        total = (
            (self.mu - self.poiss_lambda * self.k - 0.5 * self.vol**2) * self.dt
            + W
            + jump_sizes
        )

        # Loop and generate path
        for i in range(1, n_steps):
            # Recall that we are still working on logarithmic prices
            path[i] = path[i - 1] + total[i - 1]

        # Convert from log prices to nominal prices
        path = np.exp(path)
        return path

    def simulate(self, n_paths: int) -> np.ndarray:
        # Initialize numpy array to store generated paths
        n_steps = int(
            1 / self.dt
        )  # We assume that dt is a fraction of a unit time and path is simulated up to this unit of time
        paths = np.zeros((n_paths, n_steps))
        for i in range(n_paths):
            paths[i, :] = self._simulate_path(n_steps)
        return paths
