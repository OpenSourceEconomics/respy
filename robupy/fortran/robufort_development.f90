MODULE robufort_development

	!/*	external modules	*/

    USE robufort_program_constants
    USE robufort_auxiliary
    USE robufort_slsqp

	!/*	setup	*/

    IMPLICIT NONE

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE slsqp_debug(x_internal, x_start, is_upgraded, maxiter, ftol, &
                num_dim)

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: x_internal(num_dim)
    DOUBLE PRECISION, INTENT(IN)    :: x_start(num_dim)
    DOUBLE PRECISION, INTENT(IN)    :: ftol

    INTEGER, INTENT(IN)             :: num_dim
    INTEGER, INTENT(IN)             :: maxiter

    LOGICAL, INTENT(IN)             :: is_upgraded

    !/* internal objects    */

    INTEGER                         :: m
    INTEGER                         :: meq
    INTEGER                         :: la
    INTEGER                         :: n
    INTEGER                         :: len_w
    INTEGER                         :: len_jw
    INTEGER                         :: mode
    INTEGER                         :: iter
    INTEGER                         :: n1
    INTEGER                         :: mieq
    INTEGER                         :: mineq
    INTEGER                         :: l_jw
    INTEGER                         :: l_w

    INTEGER, ALLOCATABLE            :: jw(:)

    DOUBLE PRECISION, ALLOCATABLE   :: xl(:)
    DOUBLE PRECISION, ALLOCATABLE   :: xu(:)
    DOUBLE PRECISION, ALLOCATABLE   :: c(:)
    DOUBLE PRECISION, ALLOCATABLE   :: g(:)
    DOUBLE PRECISION, ALLOCATABLE   :: a(:,:)
    DOUBLE PRECISION, ALLOCATABLE   :: w(:)

    DOUBLE PRECISION                :: f
    DOUBLE PRECISION                :: acc

    LOGICAL                         :: is_finished

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! This is hard-coded for the ROBUPY package requirements
    !---------------------------------------------------------------------------
    meq = 0         ! Number of equality constraints
    mieq = 1        ! Number of inequality constraints
    !---------------------------------------------------------------------------
    !---------------------------------------------------------------------------

    ! Initialize starting values
    x_internal = x_start

    ! Derived attributes
    m = meq + mieq
    la = MAX(1, m)
    n = SIZE(x_internal)
    n1 = n + 1
    mineq = m - meq + n1 + n1

    len_w =  (3 * n1 + m) * (n1 + 1) + (n1 - meq + 1) * (mineq + 2) + &
                2 * mineq + (n1 + mineq) * (n1 - meq) + 2 * meq + n1 + &
                ((n + 1) * n) / two_dble + 2 * m + 3 * n + 3 * n1 + 1

    len_jw = mineq

    ! Allocate and initialize containers
    ALLOCATE(w(len_w)); w = zero_dble
    ALLOCATE(jw(len_jw)); jw = zero_int
    ALLOCATE(a(la, n + 1)); a = zero_dble

    ALLOCATE(g(n + 1)); g = zero_dble
    ALLOCATE(c(la)); c = zero_dble

    ! Decompose upper and lower bounds
    ALLOCATE(xl(n)); ALLOCATE(xu(n))
    xl = - huge_dble; xu = huge_dble

    ! Initialize the iteration counter and mode value
    acc = ftol
    iter = maxiter

    ! Transformations to match interface, deleted later
    l_jw = len_jw
    l_w = len_w

    ! Initialization of SLSQP
    mode = zero_int

    is_finished = .False.

    CALL debug_criterion_function(f, x_internal, n)
    CALL debug_criterion_derivative(g, x_internal, n)

    CALL debug_constraint_function(c, x_internal, n, la)
    CALL debug_constraint_derivative(a, x_internal, n, la)

    ! Iterate until completion
    DO WHILE (.NOT. is_finished)

        ! Evaluate criterion function and constraints
        IF (mode == one_int) THEN
            CALL debug_criterion_function(f, x_internal, n)
            CALL debug_constraint_function(c, x_internal, n, la)
        ! Evaluate gradient of criterion function and constraints
        ELSEIF (mode == - one_int) THEN
            CALL debug_criterion_derivative(g, x_internal, n)
            CALL debug_constraint_derivative(a, x_internal, n, la)
        END IF

        !SLSQP Interface
        IF (is_upgraded) THEN
            CALL slsqp(m, meq, n, x_internal, xl, xu, f, c, g, a, acc, iter, &
                    mode, w, l_w)
        ELSE
            CALL slsqp_original(m, meq, la, n, x_internal, xl, xu, f, c, g, &
                    a, acc, iter, mode, w, l_w, jw, l_jw)
        END IF

        ! Check if SLSQP has completed
        IF (.NOT. ABS(mode) == one_int) THEN
            is_finished = .True.
        END IF

    END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
SUBROUTINE debug_criterion_function(rslt, x, n)

    !/* external libraries    */

    USE robufort_program_constants

    !/* external objects    */

    INTEGER, INTENT(IN)             :: n

    DOUBLE PRECISION, INTENT(OUT)   :: rslt
    DOUBLE PRECISION, INTENT(IN)    :: x(n)

    !/* internal objects    */

    INTEGER                         :: i

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! Initialize containers
    rslt = zero_dble

    DO i = 2, n
        rslt = rslt + one_hundred_dble * (x(i) - x(i - 1)**2)**2
        rslt = rslt + (one_dble - x(i - 1))**2
    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE debug_criterion_derivative(rslt, x, n)

    !/* external libraries    */

    USE robufort_program_constants

    !/* external objects    */

    INTEGER, INTENT(IN)             :: n

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(n + 1)
    DOUBLE PRECISION, INTENT(IN)    :: x(n)

    !/* internals objects    */

    DOUBLE PRECISION                :: xm(n - 2)
    DOUBLE PRECISION                :: xm_m1(n - 2)
    DOUBLE PRECISION                :: xm_p1(n - 2)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Extract sets of evaluation points
    xm = x(2:(n - 1))

    xm_m1 = x(:(n - 2))

    xm_p1 = x(3:)

    ! Construct derivative information
    rslt(1) = -four_hundred_dble * x(1) * (x(2) - x(1) ** 2) - 2 * (1 - x(1))

    rslt(2:(n - 1)) =  (two_hundred_dble * (xm - xm_m1 ** 2) - &
            four_hundred_dble * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm))

    rslt(n) = two_hundred_dble * (x(n) - x(n - 1) ** 2)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE debug_constraint_function(rslt, x, n, la)

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(la)

    DOUBLE PRECISION, INTENT(IN)    :: x(n)

    INTEGER, INTENT(IN)             :: n
    INTEGER, INTENT(IN)             :: la

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt(:) = SUM(x) - 10.0

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE debug_constraint_derivative(rslt, x, n, la)

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(n + 1)

    DOUBLE PRECISION, INTENT(IN)    :: x(n)

    INTEGER, INTENT(IN)             :: n
    INTEGER, INTENT(IN)             :: la

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt = 1

    rslt(n + 1) = 0.0

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE