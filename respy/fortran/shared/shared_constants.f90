MODULE shared_constants

    !/*	setup	                */

    IMPLICIT NONE

!******************************************************************************
!******************************************************************************

    INTEGER, PARAMETER          :: our_int      = selected_int_kind(9)
    INTEGER, PARAMETER          :: our_dble     = selected_real_kind(15, 307)

    INTEGER(our_int), PARAMETER :: zero_int     = 0_our_int
    INTEGER(our_int), PARAMETER :: one_int      = 1_our_int
    INTEGER(our_int), PARAMETER :: two_int      = 2_our_int
    INTEGER(our_int), PARAMETER :: three_int    = 3_our_int

    REAL(our_dble), PARAMETER   :: zero_dble    = 0.00_our_dble
    REAL(our_dble), PARAMETER   :: half_dble    = 0.50_our_dble
    REAL(our_dble), PARAMETER   :: one_dble     = 1.00_our_dble
    REAL(our_dble), PARAMETER   :: two_dble     = 2.00_our_dble
    REAL(our_dble), PARAMETER   :: three_dble   = 3.00_our_dble

    REAL(our_dble), PARAMETER   :: pi           = 3.141592653589793238462643383279502884197_our_dble
    REAL(our_dble), PARAMETER   :: eps          = epsilon(one_dble)

    ! Variables that are part of the FORTRAN initialization file and remain unchanged during the run. These are often used to construct explicit-shape arrays. The same reasoning applied to MAX_STATES_PERIOD which is determined durint the state space creation.
    INTEGER(our_int)            :: max_states_period
    INTEGER(our_int)            :: num_points_interp
    INTEGER(our_int)            :: num_agents_est
    INTEGER(our_int)            :: num_agents_sim
    INTEGER(our_int)            :: num_draws_emax
    INTEGER(our_int)            :: num_draws_prob
    INTEGER(our_int)            :: num_periods
    INTEGER(our_int)            :: num_free
    INTEGER(our_int)            :: min_idx

    ! This variable needs to be accessible during optimization. It is defined here as it is required by the optimizers as well.
    INTEGER(our_int)            :: num_eval = zero_int

    ! Variables that need to be aligned across FORTRAN and PYTHON implementations.
    INTEGER(our_int), PARAMETER :: MISSING_INT                  = -99_our_int

    REAL(our_dble), PARAMETER   :: INADMISSIBILITY_PENALTY      = -40000.00_our_dble
    REAL(our_dble), PARAMETER   :: MISSING_FLOAT                = -99.0_our_dble
    REAL(our_dble), PARAMETER   :: SMALL_FLOAT                  = 1.0e-5_our_dble
    REAL(our_dble), PARAMETER   :: TINY_FLOAT                   = 1.0e-8_our_dble
    REAL(our_dble), PARAMETER   :: HUGE_FLOAT                   = 1.0e20_our_dble
    REAL(our_dble), PARAMETER   :: LARGE_FLOAT                  = 1.0e8_our_dble



    type optimizer_collection
       INTEGER(our_int)   :: fort_slsqp_maxiter
       REAL(our_dble)   :: fort_slsqp_ftol
    end type optimizer_collection

    type (optimizer_collection) optimizer_options

!******************************************************************************
!******************************************************************************
END MODULE
