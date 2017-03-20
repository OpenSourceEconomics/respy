MODULE shared_types

    !/* external modules    */

    USE shared_constants

    !/*	setup	                */

    IMPLICIT NONE

!******************************************************************************
!******************************************************************************

    ! We create a type that resembles the dictionary with the optimizer options in PYTHON.
    TYPE OPTIMIZER_BFGS
        INTEGER(our_int)        :: maxiter

        REAL(our_dble)          :: stpmx
        REAL(our_dble)          :: gtol
        REAL(our_dble)          :: eps
    END TYPE

    TYPE OPTIMIZER_NEWUOA
        INTEGER(our_int)        :: maxfun
        INTEGER(our_int)        :: npt

        REAL(our_dble)          :: rhobeg
        REAL(our_dble)          :: rhoend
    END TYPE

    TYPE OPTIMIZER_BOBYQA
        INTEGER(our_int)        :: maxfun
        INTEGER(our_int)        :: npt

        REAL(our_dble)          :: rhobeg
        REAL(our_dble)          :: rhoend
    END TYPE

    TYPE OPTIMIZER_SLSQP
        INTEGER(our_int)        :: maxiter

        REAL(our_dble)          :: ftol
        REAL(our_dble)          :: eps
    END TYPE

    TYPE OPTIMIZER_COLLECTION
        TYPE(OPTIMIZER_BOBYQA)  :: bobyqa
        TYPE(OPTIMIZER_NEWUOA)  :: newuoa
        TYPE(OPTIMIZER_BFGS)    :: bfgs
        TYPE(OPTIMIZER_SLSQP)   :: slsqp
    END TYPE

    ! This container holds the specification for the preconditioning step.
    TYPE PRECOND_DICT
        REAL(our_dble)          :: minimum
        REAL(our_dble)          :: eps
        CHARACTER(10)           :: type
    END TYPE

    ! This container holds the specification for the ambiguity setup.
    TYPE AMBI_DICT
        LOGICAL                 :: mean

        CHARACTER(10)           :: measure
    END TYPE

    ! This container holds all the parameters that are potentially updated during the estimation step.
    TYPE OPTIMPARAS_DICT
        REAL(our_dble)          :: shocks_cholesky(4, 4)
        REAL(our_dble)          :: paras_bounds(2, 28)
        REAL(our_dble)          :: coeffs_edu(3)
        REAL(our_dble)          :: coeffs_home(1)
        REAL(our_dble)          :: coeffs_a(6)
        REAL(our_dble)          :: coeffs_b(6)
        REAL(our_dble)          :: level(1)
        REAL(our_dble)          :: delta(1)

        LOGICAL                 :: paras_fixed(28)
    END TYPE

!******************************************************************************
!******************************************************************************
END MODULE
