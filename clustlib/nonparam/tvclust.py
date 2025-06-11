import numpy as np
import math as m
from scipy.special import digamma as phi, betaln, gammaln
from sklearn.preprocessing import normalize

from .. import BaseEstimator

import logging
logger = logging.getLogger(__name__)

class TVClust(BaseEstimator):
    """
    TVClust: A constrained variational Bayesian clustering algorithm based on a 
    truncated Dirichlet Process mixture model (TVClust).

    Attributes
    ----------
    cov_inverse : np.ndarray
        The scale matrix of the Wishart distribution associated with each cluster.

        Shape: (n_clusters, p, p), where:
            - n_clusters is the number of mixture components (clusters).
            - p is the dimensionality of the data.

        Interpretation:
            For each cluster k, `W[k]` defines the scale matrix of the Wishart distribution 
            used as a prior (or variational posterior) over the precision matrix (inverse covariance) 
            of the Gaussian component.

    responsabilities : np.ndarray
        The responsibilities of each cluster for each instance.

        Shape: (n_instances, n_clusters), where:
            - n_instances is the number of data points.
            - n_clusters is the number of mixture components (clusters).

        Interpretation:
            Each entry `responsabilities[i, k]` represents the probability that instance `i` belongs to cluster `k`.
    
    mu : np.ndarray
        The mean vector of the Gaussian component for each cluster.

        Shape: (n_clusters, p) where:
            - n_clusters is the number of mixture components (clusters).
            - p is the dimensionality of the data.

        Interpretation:
            For cluster k, `mu[k]` represents the expected location of the data in p-dimensional space.
            Updated using a precision-weighted average between prior and data responsibilities.

    nu : np.ndarray
        The degrees of freedom of the Wishart distribution for each cluster.

        Shape: (n_clusters,) where:
            - n_clusters is the number of mixture components (clusters).

        Interpretation:
            Controls the expected variability of the precision matrix.
            Larger `nu[k]` implies more confidence (tighter distribution) around `cov_inverse[k]`.

    beta : np.ndarray
        The scaling parameter of the Gaussian mean distribution for each cluster.

        Shape: (n_clusters,) where:
            - n_clusters is the number of mixture components (clusters).

        Interpretation:
            Acts as a pseudo-count indicating the strength of belief in the mean `mu[k]`.
            Affects the variance of the mean estimate; larger `beta[k]` implies lower variance.

    gamma : np.ndarray
        The variational parameters of the stick-breaking Beta distributions over cluster weights.

        Shape: (n_clusters - 1, 2) where:
            - n_clusters is the number of mixture components (clusters).
            - Each row corresponds to a Beta distribution parameterized by two values (alpha, beta).

        Interpretation:
            Each row `gamma[k] = [a, b]` encodes the Beta distribution for the stick-breaking variable `v_k`,
            which defines the prior weight of cluster k. These parameters are used to construct the
            variational approximation of the Dirichlet Process.

    """
    # Concentration of the clusters
    __alpha_0: float = 1.2

    # Responsabilities of each cluster for each instance
    # Higher values mean that the instance is more likely to belong to the cluster
    __responsabilities: np.ndarray
    __cov_inverse: np.ndarray

    __mu: np.ndarray
    __mu0: np.ndarray

    # Constraints modeling parameters
    __prior_cl_success: float = 10.0
    __prior_cl_error: float = 1.0
    __prior_ml_success: float = 1.0
    __prior_ml_error: float = 10.0

    __ml_error_prior: float = 10.0
    __cl_error_prior: float = 1.0
    __ml_success_prior: float = 1.0
    __cl_success_prior: float = 10.0

    __beta: np.ndarray
    __beta0: float

    __nu: np.ndarray
    __nu0: float

    # Variance of the instances to the clusters
    __variance: np.ndarray

    __INF = 1e20
    __ZERO = 1e-20

    mean_position: np.ndarray
    centroids: np.ndarray = None

    def __init__(
        self,
        constraints,
        n_clusters: int = 2,
        tol: float = 1e-4,
        max_iter: int = 00,
        ml_prior: tuple = (1.0, 10.0),
        cl_prior: tuple = (10.0, 1.0),
        alpha_0: float = 1.2,
        beta0: float = 1.0,
        init = "random",
    ):

        self.constraints = constraints
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.__alpha_0 = alpha_0
        self.__beta0 = beta0
        self.__prior_ml_success, self.__prior_ml_error = ml_prior
        self.__prior_cl_success, self.__prior_cl_error = cl_prior
        self.__gamma = np.ones((n_clusters, 2))
        self.__beta = np.repeat(self.__beta0, n_clusters)

    
    def concentration(self):
        """
        This measure is used to determine how "concentrated" a Gaussian component (cluster) is around its mean.
        It is calculated as the sum of the squared Mahalanobis distances between each data point and the mean of the 
        cluster, weighted by the probability of each data point belonging to that cluster.
        """
        p = self.X.shape[1]

        aux = (self.__nu[:, np.newaxis] + 1 - np.tile(np.arange(1, p + 1) * 0.5, (self.n_clusters, 1)))
        sum = np.sum(phi(aux))

        logger.debug(f"Calculating the determinant of the covariance")
        (sign, logdet) = np.linalg.slogdet(self.__cov_inverse)

        return sum + p * m.log(2) + (sign * logdet)
    
    def expected_distance(self):
        """
        Calculate the Mahalanobis distance between a point and a cluster.
        """
        p = self.X.shape[1]
        n = self.X.shape[0]
        mahalanobis_distance = np.zeros((self.n_clusters, n))
    
        for cluster in range(self.n_clusters):
            diff = self.X - self.__mu[cluster, :]
            mahalanobis_distance[cluster, :] = np.diag(
                np.linalg.multi_dot([diff, self.__cov_inverse[cluster], diff.T])
            )

        weighted_distances = (self.__nu[np.newaxis, :] * mahalanobis_distance.T).T
        beta_terms = (p / self.__beta)
        
        return beta_terms[:, np.newaxis] + weighted_distances
    
    def sbp(self):
        """
        Apply the sticky breaking process to the cluster
        """
        trust = (self.concentration()[:, np.newaxis] - self.expected_distance()) * 0.5
        trust = trust.T

        phi_alpha_beta = phi(np.sum(self.__gamma, axis = 1))[:-1]
        phi_alpha = phi(self.__gamma[:, 0])[:-1]
        phi_beta = phi(self.__gamma[:, 1])[:-1]
  
        alpha = np.concatenate(((phi_alpha - phi_alpha_beta), [0]))
        beta = np.concatenate(([0], np.cumsum(phi_beta - phi_alpha_beta)))
        
        trust += alpha + beta

        return trust
    
    def update_beta(self):
        """
        Update the beta matrix
        """
        self.__beta = np.sum(self.__responsabilities, axis = 0) + self.__beta0
    
    def update_mu(self):
        """
        Compute the posterior mean muQ[k] for cluster k.

        Note: 
        The N_k is the number of points in the cluster k, however, since it appears in the denominator and the numerator
        it cancels out, so we can ignore it.
        """
        totals = np.sum(self.__responsabilities, axis = 0)
        self.__mu = (self.__mu0 * self.__beta0 + self.mean_position * totals[:, np.newaxis]) / self.__beta[:, np.newaxis]
    
    def update_nu(self):
        """
        Update the degrees of freedom for each cluster
        """
        self.__nu = self.__nu0 + np.sum(self.__responsabilities, axis = 0)

    def update_W(self):
        totals = np.sum(self.__responsabilities, axis = 0)
        p = self.X.shape[1]

        for i in range(self.n_clusters):
            diff = self.X - self.mean_position[i]
            empirical_cov = np.dot(
                diff.T,
                np.multiply(diff, self.__responsabilities[:, i][:, np.newaxis])
            ) / totals[i]
        
            centroid_mu = np.dot((self.mean_position[i] - self.__mu0).T, (self.mean_position[i] - self.__mu0))

            aux = (
                centroid_mu / self.__beta[i] + 
                totals[i] * empirical_cov + 
                np.linalg.inv(np.identity(p))
            )

            if np.linalg.det(aux) == 0:
                logger.warning(f"Matrix is not invertible, aux shape: {aux.shape}")
                raise ValueError("Matrix is not invertible, check your data or parameters.")
                
            self.__cov_inverse[i] = np.linalg.inv(aux)

    def ml_correction(self):
       phi_alpha_beta_p = phi(self.__ml_success_prior + self.__ml_error_prior)
       phi_alpha = phi(self.__ml_success_prior)
       phi_beta = phi(self.__cl_error_prior)
       phi_alpha_beta_q = phi(self.__cl_success_prior + self.__cl_error_prior)

       return phi_alpha - phi_alpha_beta_p - phi_beta + phi_alpha_beta_q
    
    def cl_correction(self):
        phi_alpha_beta_p = phi(self.__ml_success_prior + self.__ml_error_prior)
        phi_alpha = phi(self.__cl_success_prior)
        phi_beta = phi(self.__ml_error_prior)
        phi_alpha_beta_q = phi(self.__cl_success_prior + self.__cl_error_prior)

        return phi_alpha - phi_alpha_beta_q - phi_beta + phi_alpha_beta_p

    def constraints_correction(self):
        """
        This code need to be fixes as currently it is not doing what it is supposed to do

        the main error is negative values should not have effect in the corrections
        """

        constraints = np.copy(self.constraints) # (n_clusters,)
        constraints[np.where(constraints <= 0)] = 0. # convert to binary constraints
        
        corrections = constraints * self.ml_correction() - (1 - constraints) * self.cl_correction()
        return corrections.dot(self.__responsabilities)
    
    def update_responsabilities(self):
        self.__responsabilities = np.clip(
            np.exp(self.sbp() + self.constraints_correction()), 
            a_min=self.__ZERO, 
            a_max = self.__INF
        )


        self.__responsabilities = self.__responsabilities / np.sum(
            self.__responsabilities, axis=1
        )[:, np.newaxis]

    def update_gamma(self):
        """
        Update the gamma matrix
        """
        self.__gamma = np.zeros((self.n_clusters, 2))

        responsability_sum = np.sum(self.__responsabilities, axis=0)
        cumulative_sum = np.cumsum(responsability_sum)

        self.__gamma[:-1, 0] = responsability_sum[:-1] + 1
        self.__gamma[:-1, 1] = ((cumulative_sum[-1:] - cumulative_sum) + self.__alpha_0)[:-1]

    def update_prior(self):
        """
        Update the posterior parameters of the Beta distributions used to model 
        the reliability of must-link and cannot-link constraints.

        This method recalculates the posterior shape parameters (alpha and beta) 
        for each of the four Beta distributions based on:
        - the current soft cluster assignment probabilities (responsibilities),
        - the pairwise constraint matrix,
        - and the original prior values.

        The Beta distributions being updated correspond to:
        - Must-link success     (constraint = 1, same cluster)
        - Must-link error       (constraint ≠ 1, same cluster)
        - Cannot-link success   (constraint ≠ 1, different clusters)
        - Cannot-link error     (constraint = 1, different clusters)

        It computes the "distance" between instances using the inner product of responsibilities
        (i.e., probability of co-clustering) and adjusts the shape parameters accordingly.

        Updates:
        --------
        - self.__ml_success_prior
        - self.__ml_error_prior
        - self.__cl_success_prior
        - self.__cl_error_prior

        Notes:
        ------
        - This step is part of the variational inference procedure in TVClust, where Beta-distributed
        latent variables represent the probability of observing a correct or incorrect constraint.
        - Posterior updates incorporate both soft evidence from clustering and prior beliefs.
        """
        distance = np.dot(self.__responsabilities, self.__responsabilities.T)

        inversed_distance = 1 - distance

        are_positive = np.where(self.constraints > 0)
        are_zero = np.where(self.constraints != 1) and np.where(self.constraints != -1)

        self.__ml_success_prior = (
            np.sum(distance[are_positive] * self.constraints[are_positive]) + self.__prior_ml_success
        )
        self.__ml_error_prior = (
            np.sum(distance[are_zero] * (1 - self.constraints[are_zero])) + self.__prior_ml_error
        )
        self.__cl_success_prior = (
            np.sum(inversed_distance[are_zero] * (1 - self.constraints[are_zero])) + self.__prior_cl_success
        )
        self.__cl_error_prior = (
            np.sum(inversed_distance[are_positive] * self.constraints[are_positive]) + self.__prior_cl_error
        )


    def negative_entropy(self):
        """
        Calculate the negative entropy of the model
        """
        self.__responsabilities = normalize(self.__responsabilities, axis=1, norm='l1')
        aux = self.__responsabilities + np.finfo(float).eps # avoid log(0)

        return np.sum(aux * np.log(aux))
    
    def verosimilitude(self, cluster):
        distance = np.sum(self.__responsabilities, axis = 1)

        if distance[cluster] < 1e-20:
            return 0

        conc = self.concentration()
        mean_k = np.sum(np.multiply(self.X, self.__responsabilities[:, cluster]), 0) / distance[cluster]
        diff = self.X - mean_k

        self.__variance[cluster] = (
            np.dot(
                diff.T, np.multiply(diff, self.__responsabilities[:, cluster])
            ) / distance[cluster]
        )

        return 0.5 * distance[cluster] * (conc - self.expected_distance())
   
    def check_improvement(self):
        """
        Check if the iteration has improved over the previous one
        """
        totals = np.sum(self.__responsabilities, axis = 0)
        verosimilitude = 0
        concentration = self.concentration()
        expected_log_joint = 0
        expected_log_stick_weight = 0
        expected_log_alpha_term = 0

        negative_entropy = np.sum(self.__responsabilities * np.log(self.__responsabilities + 1e-20))
        for cluster in range(self.n_clusters):
            if totals[cluster] < self.tol:
                verosimilitude += self.verosimilitude(cluster)
                if cluster < self.n_clusters - 1:
                    a, b = self.__gamma[cluster, 0], self.__gamma[cluster, 1]
                    expected_log_stick_weight += (
                        totals[cluster] * (phi(a) - phi(a + b)) +
                        np.sum(totals[cluster + 1:] * (phi(b) - phi(a + b)))
                    )
                    expected_log_alpha_term += (self.__alpha0 - 1) * (phi(b) - phi(a + b))


            expected_log_joint += self.compute_expected_log_prior(cluster, concentration[cluster])
        
        entropy_sbp = self.entropy_sbp()
        entropy_wishart = self.entropy_wishart()
        penalty_constraints = self.penalty_constraints()
        return (
            verosimilitude + 
            expected_log_joint +
            expected_log_stick_weight +
            expected_log_alpha_term - 
            negative_entropy - 
            entropy_sbp - 
            entropy_wishart +
            penalty_constraints
        )
            
    def compute_expected_log_prior(self, cluster, concentration):
        p = self.X.shape[1]
        diff_mu = self.__mu[cluster].T - self.__mu0[np.newaxis, :]
        mahal_term = np.dot(diff_mu, np.dot(self.__cov_inverse[cluster], diff_mu.T))[0,0]
        penalty_over_beta = (p * self.__beta0) / self.__beta[cluster]
        penalty_over_mean = self.__beta0 * self.__nu[cluster] * mahal_term
        expected_log_prior_mean = (penalty_over_beta - penalty_over_mean) * 0.5

        trace = 0.5 * np.trace(
            np.dot(
                np.linalg.inv(np.identity(p)), 
                self.__cov_inverse[cluster]
            )
        )
        prior_precision_term = self.__nu[cluster] * trace

        return (concentration * (self.__nu0 - p) * 0.5) + expected_log_prior_mean - prior_precision_term           

    def entropy_sbp(self):
        total = 0.
        for k in range(self.n_clusters - 1):
            a,b = self.__gamma[k, 0], self.__gamma[k, 1]
            total += (
                (a - 1) * (phi(a) - phi(a + b)) +
                (b - 1) * (phi(b) - phi(a + b)) -
                betaln(a, b)
            )

        return total
    
    def entropy_wishart(self):
        """
        Computes the entropy contribution of the Wishart distributions
        over the precision matrices of all clusters.

        Returns:
            total (float): Sum of entropies for all clusters.
        """
        total = 0.0
        p = self.X.shape[1]
        p_values = np.arange(1, p + 1)

        for k in range(self.n_clusters):
            nu = self.__nu[k]
            beta = self.__beta[k]

            log_det_W = np.linalg.slogdet(self.__cov_inverse[k])[1]
            psi_term = np.sum(phi((nu + 1 - p_values) * 0.5))

            # Log normalizing constant of the Wishart
            log_normal = (
                -0.5 * nu * log_det_W
                - 0.5 * nu * p * np.log(2)
                - 0.25 * p * (p - 1) * np.log(np.pi)
                - np.sum(gammaln((nu + 1 - p_values) * 0.5))
            )

            # Entropy of the Wishart
            entropy_wishart = (
                -log_normal
                - 0.5 * (nu - p - 1) * (psi_term + p * np.log(2) + log_det_W)
                + 0.5 * nu * p
            )

            total += (
                0.5 * psi_term
                + 0.5 * log_det_W
                + 0.5 * p * np.log(beta)
                - entropy_wishart
            )

        return total

    def penalty_constraints(self):
        # Compute the expected values from the beta distributions
        expected_ml = phi(self.__ml_success_prior) - phi(self.__ml_success_prior + self.__ml_error_prior)
        expected_inverse_ml = phi(self.__ml_error_prior) - phi(self.__ml_success_prior + self.__ml_error_prior)
        expected_cl = phi(self.__cl_success_prior) - phi(self.__cl_success_prior + self.__cl_error_prior)
        expected_inverse_cl = phi(self.__cl_error_prior) - phi(self.__cl_success_prior + self.__cl_error_prior)

        ml_believe = (self.__ml_success_prior - self.__prior_ml_success)
        ml_sceptic = (self.__ml_error_prior - self.__prior_ml_error)

        cl_believe = (self.__cl_success_prior - self.__prior_cl_success)
        cl_sceptic = (self.__cl_error_prior - self.__prior_cl_error)
        likehood_constraints = (
            ml_believe * expected_ml + 
            ml_sceptic * expected_inverse_ml + 
            cl_believe * expected_cl + 
            cl_sceptic * expected_inverse_cl
        )
        divergence = (
            (self.__prior_ml_success - self.__ml_success_prior) * expected_ml + 
            (self.__prior_ml_error - self.__ml_error_prior) * expected_inverse_ml + 
            (self.__prior_cl_success - self.__cl_success_prior) * expected_cl + 
            (self.__prior_cl_error - self.__cl_error_prior) * expected_inverse_cl + 
            betaln(self.__ml_success_prior, self.__ml_error_prior) + 
            betaln(self.__cl_success_prior, self.__cl_error_prior)
        )

        return likehood_constraints + divergence

    def initialize_parameters(self):
        n, p = self.X.shape

        logger.debug("Initializing parameters for TVClust, n_clusters=%d, p=%d", self.n_clusters, p)
        self.__responsabilities = np.ones((n, self.n_clusters), dtype=np.float64)
        self.__responsabilities = self.__responsabilities / float(self.n_clusters)

        self.__cov_inverse = np.tile(np.identity(p), (self.n_clusters, 1, 1))
        logger.debug(f"Covariance inverse: {self.__cov_inverse.shape}")
        
        self.__mu = np.random.randn(self.n_clusters, p)
        self.__nu = np.repeat(p, self.n_clusters)
        self.mean_position = np.copy(self.__mu)
        self.__nu0 = p
        self.__mu0 = np.zeros(p)
        self._delta = None

    def _convergence(self):
        if self._delta is None:
            return False

        return np.abs(self._delta) < self.tol
    
    def calculte_delta(self, _):
        if self._delta is None:
            self._delta = self.check_improvement()
        else:
            self._delta = (self.check_improvement() - self._delta) / np.absolute(self._delta)
        logger.debug(f"Delta: {self._delta}")

    def update(self):
        self._update()
        self.calculte_delta(None)

    def _update(self):
        logger.debug("Updating responsibilities")
        self.update_responsabilities()

        logger.debug("Updating gamma")
        self.update_gamma()

        logger.debug("Updating beta")
        self.update_beta()

        logger.debug("Updating mu")
        self.update_mu()

        logger.debug("Updating W")
        self.update_W()

        logger.debug("Updating nu")
        self.update_nu()

        logger.debug("Updating prior")
        self.update_prior()

    def _condicional_prob(self):   
        self._labels = np.argmax(self.__responsabilities, axis = 1)

    def _mean_position(self):
        """
        Calculate the mean_position of the cluster based on the current responsibilities.

        Update the `self.mean_position` attribute with the mean of the data points
        """
        totals = np.clip(
            np.sum(self.__responsabilities, axis = 0), 
            a_min=self.__ZERO, 
            a_max=self.__INF
        )

        self.mean_position = np.clip(
            self.__responsabilities.T.dot(self.X) / totals[:, np.newaxis],
            a_min=self.__ZERO,
            a_max=self.__INF
        )

    def get_centroids(self):
        """
        Get the centroids of the clusters.

        Returns
        -------
        numpy.ndarray
            The centroids of the clusters.
        """
        for i in range(self.n_clusters):
            weighted_sum = (
                self.X * self.__responsabilities[:, i][:, np.newaxis]
            )[self._labels == i]
            self.centroids[i] = np.sum(weighted_sum, axis=0) / np.sum(self.__responsabilities[:, i][self._labels == i])

    def _fit(self):
        self.initialize_parameters()

        for iteration in range(self.max_iter):
            logger.debug(f"Iteration {iteration + 1}/{self.max_iter}")
            self.update()
            self._condicional_prob()
            self._mean_position()
            self.get_centroids()

            if self.stop_criteria(iteration):
                break
