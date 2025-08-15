# Architecture

``` mermaid
classDiagram
direction TB
	namespace GeneticAlgorithms {
        class SHADE {
        }

        class BKGA {
        }

        class Genetic {
	        + mutation()
	        + cross()
	        + selection()
	        + generation()
        }

	}
	namespace KMeans {
        class COPKMeans {
        }

        class SoftCOPKMeans {
        }

	}
	namespace LocalSearch {
        class DILS {
        }

	}
	namespace Fuzzylogic {
        class LCVQE {
        }

	}
    class BaseEstimator {
	    - _n_clusters_
	    - _X_
	    - _constraints_
	    + convergence()
	    + update()
    }

    class ClusterMixin {
	    - _labels_
	    + fit_predict()
    }

    class Estimator {
	    + fit()
	    + predict()
    }

	<<Interface>> Genetic
	<<Interface>> BaseEstimator
	<<Interface>> ClusterMixin
	<<Interface>> Estimator

    SoftCOPKMeans --|> BaseEstimator
    COPKMeans --|> BaseEstimator
    LCVQE --|> BaseEstimator
    DILS --|> BaseEstimator
    Genetic --|> BaseEstimator
    BaseEstimator --|> ClusterMixin
    ClusterMixin --|> Estimator
    BKGA --|> Genetic
    SHADE --|> Genetic
```