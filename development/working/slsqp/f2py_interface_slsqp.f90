!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_slsqp_f90(x_internal, x_start, maxiter, ftol, num_dim)

    !/* external libraries    */

    USE slsqp_updated
    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: x_internal(num_dim)
    DOUBLE PRECISION, INTENT(IN)    :: x_start(num_dim)    
    DOUBLE PRECISION, INTENT(IN)    :: ftol    

    INTEGER, INTENT(IN)             :: num_dim
    INTEGER, INTENT(IN)             :: maxiter

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
    INTEGER                         :: majiter_prev
    INTEGER                         :: majiter
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

!------------------------------------------------------------------------------ 
! Algorithm
!------------------------------------------------------------------------------ 
    
    ! Initialize starting values    
    x_internal = x_start

    ! Set
    meq = 0         ! Number of equality constraints
    mieq = 0        ! Number of inequality constraints
    
    ! Define workspace for SLSQP
    m = meq + mieq  ! Total number of constraints
    la = MAX(1, m)  ! The number of constraints or one if there are not constraints

    ! Housekeeping
    n = SIZE(x_internal)
    n1 = n + 1
    mineq = m - meq + n1 + n1

    len_w =  (3 * n1 + m) * (n1 + 1) + (n1 - meq + 1) * (mineq + 2) + &
                2 * mineq + (n1 + mineq) * (n1 - meq) + 2 * meq + n1 + &
                ((n + 1) * n) / two_dble + 2 * m + 3 * n + 3 * n1 + 1

    len_jw = mineq

    ALLOCATE(w(len_w)); w = zero_dble
    ALLOCATE(jw(len_jw)); jw = zero_int
    ALLOCATE(a(1, n + 1)); a = zero_dble

    ALLOCATE(g(n))
    ALLOCATE(c(0))

    ! Decompose upper and lower bounds
    ALLOCATE(xl(n)); ALLOCATE(xu(n))
    xl = -huge_dble; xu = huge_dble

    ! Initialize the iteration counter and mode value
    acc  = ftol
    iter = maxiter

    majiter = iter
    majiter_prev = 0

    ! Transformations to match interface, deleted later
    l_jw = len_jw
    iter = majiter
    l_w = len_w

    ! Initialization of SLSQP
    mode = zero_int 

    is_finished = .False.

    CALL rosenbrock(f, x_internal)

    CALL rosenbrock_derivative(g, x_internal)

    ! Iterate until completion
    DO WHILE (.NOT. is_finished)
        
        ! Update information 
        IF (mode == one_int) THEN
            CALL rosenbrock(f, x_internal)
        ELSEIF (mode == -one_int) THEN      
            CALL rosenbrock_derivative(g, x_internal)
        END IF

        !SLSQP Interface
        CALL slsqp_f90(m, meq, la, n, x_internal, xl, xu, f, c, g, a, acc, iter, & 
                mode, w, l_w, jw, l_jw)!

        ! Check if SLSQP has completed
        IF (.NOT. ABS(mode) == one_int) THEN
            is_finished = .True.
        END IF

    END DO


END SUBROUTINE 
!******************************************************************************
!******************************************************************************
SUBROUTINE wrapper_slsqp(x_internal, x_start, maxiter, ftol, num_dim)

    !/* external libraries    */

    USE robufort_auxiliary

    !/* setup    */

    IMPLICIT NONE

    !/* external objects    */

    DOUBLE PRECISION, INTENT(OUT)   :: x_internal(num_dim)
    DOUBLE PRECISION, INTENT(IN)    :: x_start(num_dim)    
    DOUBLE PRECISION, INTENT(IN)    :: ftol    

    INTEGER, INTENT(IN)             :: num_dim
    INTEGER, INTENT(IN)             :: maxiter

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
    INTEGER                         :: majiter_prev
    INTEGER                         :: majiter
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

!------------------------------------------------------------------------------ 
! Algorithm
!------------------------------------------------------------------------------ 
    
    ! Initialize starting values    
    x_internal = x_start

    ! Set
    meq = 0         ! Number of equality constraints
    mieq = 0        ! Number of inequality constraints
    
    ! Define workspace for SLSQP
    m = meq + mieq  ! Total number of constraints
    la = MAX(1, m)  ! The number of constraints or one if there are not constraints

    ! Housekeeping
    n = SIZE(x_internal)
    n1 = n + 1
    mineq = m - meq + n1 + n1

    len_w =  (3 * n1 + m) * (n1 + 1) + (n1 - meq + 1) * (mineq + 2) + &
                2 * mineq + (n1 + mineq) * (n1 - meq) + 2 * meq + n1 + &
                ((n + 1) * n) / two_dble + 2 * m + 3 * n + 3 * n1 + 1

    len_jw = mineq

    ALLOCATE(w(len_w)); w = zero_dble
    ALLOCATE(jw(len_jw)); jw = zero_int
    ALLOCATE(a(1, n + 1)); a = zero_dble

    ALLOCATE(g(n))
    ALLOCATE(c(0))

    ! Decompose upper and lower bounds
    ALLOCATE(xl(n)); ALLOCATE(xu(n))
    xl = -huge_dble; xu = huge_dble

    ! Initialize the iteration counter and mode value
    acc  = ftol
    iter = maxiter

    majiter = iter
    majiter_prev = 0

    ! Transformations to match interface, deleted later
    l_jw = len_jw
    iter = majiter
    l_w = len_w

    ! Initialization of SLSQP
    mode = zero_int 

    is_finished = .False.

    CALL rosenbrock(f, x_internal)

    CALL rosenbrock_derivative(g, x_internal)

    ! Iterate until completion
    DO WHILE (.NOT. is_finished)
        
        ! Update information 
        IF (mode == one_int) THEN
            CALL rosenbrock(f, x_internal)
        ELSEIF (mode == -one_int) THEN      
            CALL rosenbrock_derivative(g, x_internal)
        END IF

        !SLSQP Interface
        CALL slsqp(m, meq, la, n, x_internal, xl, xu, f, c, g, a, acc, iter, & 
                mode, w, l_w, jw, l_jw)!

        ! Check if SLSQP has completed
        IF (.NOT. ABS(mode) == one_int) THEN
            is_finished = .True.
        END IF

    END DO


END SUBROUTINE 