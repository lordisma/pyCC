import numpy as np
from numpy import matlib
import skfuzzy as fuzz
from set_centers_ecm import set_centers_ecm
from set_distances import set_distances
from utils import solqp


class CEKM:
    def __init__(self, X, K, constraints, max_iter=300, rho=100, xi=0.5, stop_thr=1e-3, init='rand', alpha=1):
        self.X = X
        self.n_clusters = K
        self.constraints = constraints
        self.max_iter = max_iter
        self.rho = rho
        self.xi = xi
        self.stop_thr = stop_thr
        self.init = init
        self.alpha = alpha

        self.AEQ = np.kron(np.identity(X.shape[0]), np.ones((1, 2 ** K)))
        self.BEQ = np.ones((X.shape[0], 1))
        self.beta = 2
        self.alpha = 1

        self.__believe_partitions = np.zeros((2**K, X.shape[1]))
        self.__focal_matrix = self.__construct_focal_set_matrix()
        self._distances = np.zeros((self.X.shape[0], self.__believe_partitions.shape[0] - 1))
    
    def __construct_focal_set_matrix(self):
        result = np.zeros((2 ** self.n_clusters, self.n_clusters))

        for i in range(2 ** self.n_clusters):
            result[i, :] =  np.array(list(map(int, bin(i)[2:][::-1].ljust(self.n_clusters, '0'))))
        
        return result
    
    def __compute_masses(self):
        numb_obj = self.X.shape[0]
        believe_size = self.__believe_partitions.shape[0]

        self.__set_distances()

        beta_inv = 1 / (self.beta - 1)
        c = np.asarray((np.sum(self.__focal_matrix, axis=1)))
        c = c ** (self.alpha * beta_inv)
        masses = np.zeros((numb_obj, believe_size - 1))

        distances_c = ((c * self._distances) / self.rho) ** beta_inv 

        for i, distances in enumerate(self._distances):
            for j, distance in enumerate(distances):
                vect1 = (distance / distances) ** beta_inv
                vect3 = vect1 * (c[j] / c)
                div = (np.sum(vect3) + distances_c[i, j])

                if (div == 0):
                    div = 1

                masses[i, j] = 1 / div

        empty_mass = np.abs(np.ones((numb_obj, 1)) - np.sum(masses, 1))
        masses = np.concatenate(empty_mass, axis = 1)

        return masses
    
    def __set_centers_ecm(self):
        pass

    def __set_distances(self):
        focal_dot_centroids = np.dot(self.__focal_matrix, self._centroids)[1:, :]
        believe_partition = focal_dot_centroids / np.sum(self.__focal_matrix, axis=1)[1:, :]
        self.__believe_partitions = believe_partition.reshape(self.__believe_partitions.shape)

        for j in range(self.__believe_partitions.shape[0] - 1):
            self._distances[:, j] = np.linalg.norm(
                self.X - np.tile(self.__believe_partitions[j, :], (self.X.shape[1], 1)), axis=1
            )

    def _update_masses(self, f, H):
        masses, _, _ = solqp(H, self.AEQ, self.BEQ, f, self.__masses)

        self.__masses = masses

    def fit(self):
        # Initialization TODO

        self.__compute_masses()

        # Build F matrix
        f = np.zeros((self.__believe_partitions.shape[0] - 1, self.X.shape[0]))
        f[0, :] = np.sum(self.constraints[np.where(self.constraints > 0)], axis = 1)
        # Build H matrix
        H = self.__build_H_matrix()

        while True:
            if self.__should_stop():
                break

            self._update_masses(f, H)

        pass


def get_bit(numb, K, ind):
    # Primero el menos relevante
    return int((''.join(str(1 & int(numb) >> i) for i in range(K)))[ind])


def CEKM(X, K, constraints, max_iter=300, rho=100, xi=0.5, stop_thr=1e-3, init='rand', alpha=1):

    if alpha < 1:
        alpha = 1

    if rho < 0:
        rho = 100

    if xi < 0 or xi > 1:
        xi = 0.5

    rows, cols = np.shape(X)
    ident_matrix = np.identity(rows)
    beta = 2

    # constraint matrix reformulations
    mat_contraintes = np.sign(constraints + constraints.conj().T - ident_matrix)
    aux = constraints * np.sign(constraints)
    aux = np.maximum(aux, aux.conj().T)
    mat_contraintes = mat_contraintes * aux

    # construction of the focal set matrix (Done)
    F = np.zeros((2 ** K, K))

    for i in range(2 ** K):
        F[i, :] =  np.array(list(map(int, bin(i)[2:][::-1].ljust(K, '0'))))



    # set Aeq and beq matrix (Done)
    aeq = np.kron(ident_matrix, np.ones((1, nb_foc)))
    beq = np.ones((rows, 1))

    # centroids inicialization (Done)
    if (init == 'rand'):
        g = np.random.rand(K, cols) * np.matlib.repmat(np.max(X) - np.min(X), K, 1) + np.matlib.repmat(np.min(X), K, 1)
    else:
        g = fuzz.cluster.cmeans(X.T, K, 2, 1e-5, 100)[0]

    # centers calculus for all the subsets (Done)
    gplus = np.zeros( ((2**K) - 1, cols) )

    for i in range(1, 2**K):
        fi = np.array([F[i, :]]) # 1,0,0,0,0
        truc = np.matlib.repmat(fi.conj().T, 1, cols)
        gplus[i-1,:] = np.sum(g * truc, axis = 0) / np.sum(truc, axis = 0)

    gplus = np.random.rand(2 ** K - 1, cols) * (np.max(X) - np.min(X)) + np.min(X)

    # compute euclidean distance (Done)
    D = np.zeros((rows, nb_foc - 1))
    for j in range(nb_foc - 1):
        aux = (X - np.dot(np.ones((rows, 1)), np.matrix(gplus[j, :])))
        B = np.diag(np.dot(aux, aux.conj().T))
        D[:, j] = B

    # compute masses (Done)
    c = np.asarray((np.sum(F[1:, :], axis=1)))
    masses = np.zeros((rows, nb_foc - 1))

    for i in range(rows):
        for j in range(nb_foc - 1):
            vect1 = D[i, :]
            vect1 = np.dot(D[i, j], np.ones((1, nb_foc - 1)) / vect1) ** (1 / (beta - 1)) # 1/D
            vect2 = ((c[j] ** (alpha / (beta - 1))) * np.ones((1, nb_foc - 1))) / (c ** (alpha / (beta - 1)))
            vect3 = vect1 * vect2
            div = (np.sum(vect3) + ((c[j] ** (alpha / (beta - 1))) * D[i, j] / rho) ** (1 / (beta - 1)))

            if (div == 0):
                div = 1

            masses[i, j] = 1 / div

    masses = np.concatenate((np.abs(np.ones((rows, 1)) - np.matrix(np.sum(masses, 1)).T), np.abs(masses)), 1) # Aniadimos la masa de la clase vacia
    
    x0 = masses.conj().T.reshape(rows * nb_foc, 1)
    D, S, Smeans = set_distances(X, F, g)

    # Setting f matrix
    aux = mat_contraintes - np.identity(rows)
    contraintes_ml = np.maximum(aux, np.zeros((rows, rows)))
    nb_cont_par_object = np.sum(contraintes_ml[np.where(constraints > 0)], axis = 1)
    aux_zeros = np.zeros((nb_foc, 1))
    aux_zeros[0, 0] = 1
    fvide = np.kron(nb_cont_par_object, aux_zeros)
    f = fvide 
    # Build a matrix of (2^K -1) x rows, where first row is the number of ML constraints for each object

    # Setting constraints matrix
    ind = np.tril_indices(rows, -1)
    nb_ml = len(np.where(mat_contraintes[ind] == 1)[0])
    nb_cl = len(np.where(mat_contraintes[ind] == -1)[0])

    if (nb_ml == 0):
        nb_ml = 1

    if (nb_cl == 0):
        nb_cl = 1

    ml_mat = np.power(((np.sign(np.power(np.dot(F, np.ones((K, 1))) - 1, 2))) - 1), 2)
    ml_mat = np.dot(ml_mat, ml_mat.conj().T) * np.dot(F, F.conj().T)
    cl_mat = np.sign(np.dot(F, F.conj().T))
    ml_mat = ml_mat * -np.sign(xi) / (2 * nb_ml)
    cl_mat = cl_mat * np.sign(xi) / (2 * nb_cl)

    # contraints matrix with respect to the constraints give in parameters
    aux = np.tril(mat_contraintes, -1)
    contraintes_ml = np.maximum(aux, np.zeros((rows, rows)))
    contraintes_cl = np.absolute(np.minimum(aux, np.zeros((rows, rows))))

    ml_aux = np.kron(contraintes_ml, np.ones((nb_foc, nb_foc)))
    cl_aux = np.kron(contraintes_cl, np.ones((nb_foc, nb_foc)))

    contraintes_mat = np.matlib.repmat(ml_mat, rows, rows) * ml_aux + np.matlib.repmat(cl_mat, rows, rows) * cl_aux
    contraintes_mat = contraintes_mat + contraintes_mat.conj().T

    # Setting H matrix
    aux = np.dot(D, np.concatenate((np.zeros((nb_foc - 1, 1)), np.identity(nb_foc - 1)), 1))
    aux = aux + np.concatenate((np.ones((rows, 1)) * rho, np.zeros((rows, nb_foc - 1))), 1)

    vect_dist = aux.flatten()

    card = np.sum(F.conj().T, 0)
    card[0] = 1
    card = np.matlib.repmat(card ** alpha, 1, rows)

    if (xi > 0):
        H = (1 - xi) * np.diag(vect_dist * card / (rows * nb_foc)) + xi * contraintes_mat
    else:
        H = np.diag(vect_dist.T * card / (rows * nb_foc)) + contraintes_mat

    not_finished = True
    gold = g
    it_count = 0
    while not_finished and it_count < max_iter:
        # Alter the masses to solve the system of lineal equations
        mass, l, fval = solqp(H, aeq, beq, f, x0)

        x0 = mass

        # reshape m
        m = mass.reshape(nb_foc, rows)
        m = np.asmatrix(m[1:nb_foc, :]).conj().T

        # calculation of centers
        g = set_centers_ecm(X, m, F, Smeans, alpha, beta)
        D, S, Smeans = set_distances(X, F, g)

        # Setting H matrix
        aux = np.dot(D, np.concatenate((np.zeros((nb_foc - 1, 1)), np.identity(nb_foc - 1)), 1))
        aux = aux + np.concatenate((np.ones((rows, 1)) * rho, np.zeros((rows, nb_foc - 1))), 1)

        vect_dist = aux.flatten()

        card = np.sum(F.conj().T, 0)
        card[0] = 1
        card = np.matlib.repmat(card ** alpha, 1, rows)

        H = (1 - xi) * np.diag(vect_dist * card / (rows * nb_foc)) + xi * contraintes_mat

        J = np.dot(np.dot(mass.conj().T, H), mass) + xi

        diff = np.abs(g - gold)
        grater_than_threshold = diff > stop_thr
        not_finished = sum(diff[grater_than_threshold]) > 0
        gold = g
        it_count += 1

    m = np.concatenate( (np.abs(np.ones((rows, 1)) - np.sum(m, 1)), np.abs(m)), 1)

    # Bet calcul
    bet_p = np.zeros((rows, K))

    cardinals = np.array([0, 1])

    for i in range(1, K):
        cardinals = np.append(cardinals, cardinals + 1)

    cardinals[0] = 1

    aux = m / np.matlib.repmat(cardinals, rows, 1)

    for i in range(1, K+1):

        ind = np.array(range(2 ** (i-1) + 1, 2 ** i +1))

        if (i < K):
            for j in range(1,K - i + 1):
                ind = np.append(ind, ind + 2 ** (i + j - 1))

        ind = ind - 1
        sum_ind = np.sum(aux[:, ind], 1).T
        bet_p[:, i-1] = sum_ind

    predicted = np.array([np.argmax(bet_p[i, :]) for i in range(np.shape(bet_p)[0])], dtype=np.uint8)

    return predicted, bet_p, m, g, J