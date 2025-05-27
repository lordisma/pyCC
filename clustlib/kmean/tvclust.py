import numpy as np
import math as m
from scipy.special import digamma as phi, betaln, gammaln
import scipy.stats as st
from sklearn.preprocessing import normalize

from .. import BaseEstimator
from ..utils.distance import match_distance

import itertools as it


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
    __scale_matrices: np.ndarray

    __INF = 1e20
    __ZERO = 1e-20

    __centroids: np.ndarray

    def __init__(
        self,
        constraints,
        n_clusters: int = 2,
        tol: float = 1e-4,
        max_iter: int = 00,
        ml_prior: tuple = (1.0, 10.0),
        cl_prior: tuple = (10.0, 1.0),
        alpha_0: float = 1.2,
        beta0: float = 1.0
        ):

        self.constraints = constraints
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
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
        sum = np.sum(phi(((self.__nu + 1) - np.repeat(np.arange(1, p + 1), self.n_clusters)) * 0.5))

        return sum + p * m.log(2) + np.log(np.linalg.det(self.__cov_inverse))
    
    def expected_distance(self):
        """
        Calculate the Mahalanobis distance between a point and a cluster.
        """
        p = self.X.shape[1]
        totals = np.sum(self.__responsabilities, axis = 1)
        mahalanobis_distance = np.zeros(self.n_clusters)

        for cluster in range(self.n_clusters):
            self.__centroids[cluster] = (
                np.mean(
                    np.multiply(self.X, self.__responsabilities[:, cluster][:, np.newaxis]), 
                    axis = 0
                ) / totals[cluster]
            )
    
            diff = self.X - self.__centroids[cluster]
            mahalanobis_distance[cluster] = np.dot(np.dot(diff, self.__cov_inverse[cluster]), diff.T)
        
        return p/self.__beta + self.__nu * mahalanobis_distance
    
    def sbp(self, i):
        """
        Apply the sticky breaking process to the cluster
        """
        trust = (self.concentration() - self.expected_distance()) * 0.5

        phi_alpha_beta = phi(np.sum(self.__gamma, axis = 0))
        phi_alpha = phi(self.__gamma[:, 0])
        phi_beta = phi(self.__gamma[:, 1])

        trust += np.sum(phi_alpha - phi_alpha_beta) + np.sum(phi_beta - phi_alpha_beta)

        return trust
    
    def update_beta(self):
        """
        Update the beta matrix
        """
        self.__beta = np.zeros(self.n_clusters)
        for cluster in range(self.n_clusters):
            self.__beta[cluster] = np.sum(self.__responsabilities[:, cluster]) + self.__beta0
    
    def update_posterior_mean(self):
        """
        Compute the posterior mean muQ[k] for cluster k.

        Note: 
        The N_k is the number of points in the cluster k, however, since it appears in the denominator and the numerator
        it cancels out, so we can ignore it.
        """
        for k in range(self.n_clusters):
            weighted_responsabilities = np.sum(self.X * self.__responsabilities[:, k][:, np.newaxis], axis=0)
            self.__mu[k] = (self.__beta0 * self.__mu0 + weighted_responsabilities) / self.__beta[k]
        pass

    def wishart_scale_update(self, cluster):
        total = np.sum(self.__responsabilities[:, cluster])
        mean_x = np.mean(np.multiply(self.X, self.__responsabilities[:, cluster][:, np.newaxis]), axis = 0) / total

        empirical_cov = np.dot(
            (self.X - mean_x).T,
            np.multiply((self.X - mean_x), self.__responsabilities[:, cluster][:, np.newaxis])
        ) / total
        
        diff = mean_x - self.__mu0
        penalty = self.__beta0 * total / np.outer(diff, diff) * self.__beta[cluster]
        
        self.__scale_matrices[cluster] = np.linalg.inv(penalty + total * empirical_cov + np.linalg.inv(self.__W0))

    def degrees_of_freedom_update(self):
        """
        Update the degrees of freedom for each cluster
        """
        for k in range(self.n_clusters):
            self.__nu[k] = self.__nu0 + np.sum(self.__responsabilities[:, k])

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

    def constraints_correction(self, instance, cluster):
        """
        This code need to be fixes as currently it is not doing what it is supposed to do

        the main error is negative values should not have effect in the corrections
        """

        constraints = self.constraints[instance]
        to_consider = np.argwhere(constraints >= 0)

        constraints = constraints[to_consider]
        if len(to_consider) == 0:
            return 0

        correction = constraints * self.ml_correction() - (1 - constraints) * self.cl_correction()
        return np.sum(correction * self.__responsabilities[to_consider, cluster])
    
    def update_responsabilities(self):
        
        for i, instance in enumerate(self.X.shape[0]):
            for cluster in range(self.n_clusters):
                self.__responsabilities[i, cluster] = max(
                    min(
                        np.exp(self.sbp(instance, cluster) + self.constraints_correction(i, cluster)),
                        self.__INF
                    ), 
                    self.__ZERO
                )

        self.__responsabilities = self.__responsabilities / np.sum(
            self.__responsabilities, axis=1, keepdims=True
        )

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

        conc = self.concentration(cluster)
        mean_k = np.sum(np.multiply(self.X, self.__responsabilities[:, cluster]), 0) / distance[cluster]
        diff = self.X - mean_k

        self.__variance[cluster] = (
            np.dot(
                diff.T, np.multiply(diff, self.__responsabilities[:, cluster])
            ) / distance[cluster]
        )

        return 0.5 * distance[cluster] * (conc - self.expected_distance(cluster))
   
    def check_improvement(self):
        """
        Check if the iteration has improved over the previous one
        """
        if self.__previous is None:
            return np.inf
        
        totals = np.sum(self.__responsabilities, axis = 1)
        verosimilitude = 0

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


            expected_log_joint += self.compute_expected_log_prior(cluster, self.concentration(cluster))
        
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

        trace = 0.5 * np.trace(np.dot(np.linalg.inv(self.__W0), self.__cov_inverse[cluster]))
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

            log_det_W = np.linalg.slogdet(self.__scale_matrices[k])[1]
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
        self.__responsabilities = np.ones((self.X.shape[0], self.n_clusters)) / self.n_clusters
        self.__cov_inverse = np.stack([np.identity(self.X.shape[1])] * self.n_clusters)
        self.__mu = np.random.randn((self.n_clusters, self.X.shape[1]))
        self.__nu = np.repeat(self.X.shape[1], self.n_clusters)
        self.__centroids = np.zeros((self.n_clusters, self.X.shape[1]))

    def fit(self, dataset: np.ndarray, labels: np.array = None):
        self.X = dataset

        self.initialize_parameters()

        if labels:
            self._labels = labels
        else: 
            self._labels = np.random.randint(0, self.n_clusters, self.X.shape[0])

        self.__E_ = np.ones((self.X.shape[0], self.n_clusters)) / self.n_clusters

        for iteration in range(self.max_iter):
            condicional_prob = np.zeros((self.X.shape[0], self.num_clusters + 1))

            for i in range(self.X.shape[0]):
                condicional_prob[i] = self.calculate_condicional_prob(i)

            self.labels = np.argmax(condicional_prob, 1)
            self.update()

            if self.stop_criteria(iteration):
                break
