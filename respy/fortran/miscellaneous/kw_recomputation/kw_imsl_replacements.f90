!*******************************************************************************
!*******************************************************************************
MODULE IMSL_REPLACEMENTS

    !/* setup   */

    IMPLICIT NONE
        
    ! External procedures defined in LAPACK
    EXTERNAL DGETRF
    EXTERNAL DGETRI
    EXTERNAL DPOTRF

    PUBLIC

CONTAINS
!*******************************************************************************
!*******************************************************************************
SUBROUTINE LFCDS(N, A, LDA, FACT, LDFACT, RCOND)

    !/* external objects    */

    REAL, INTENT(OUT)       :: FACT(N, N)

    INTEGER, INTENT(IN)     :: LDFACT
    INTEGER, INTENT(IN)     :: LDA 
    INTEGER, INTENT(IN)     :: N
    
    REAL, INTENT(IN)        :: A(N, N)  
    REAL, INTENT(IN)        :: RCOND

    !/* internal objects    */

    INTEGER                 :: INFO

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

    ! Initialize matrix for replacement.
    FACT = A

    ! Compute the Cholesky factorization of a real symmetric positive 
    ! definite matrix A (single precision).
    CALL SPOTRF('L', N, FACT, N, INFO)

END SUBROUTINE 
!******************************************************************************* 
!******************************************************************************* 
SUBROUTINE RNNOR(dim, draw)

    !
    ! This subroutine generates deviates from a standard normal distribution 
    ! using the Box-Muller algorithm.
    !

    !/* external objects    */

    INTEGER, INTENT(IN)     :: dim

    REAL, INTENT(OUT)       :: draw(dim)
        
    !/* internal objects    */

    INTEGER                 :: g 

    REAL, PARAMETER         :: pi = 3.141592653589793238462643383279502884197
    REAL, ALLOCATABLE       :: u(:), r(:)

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

    ! Allocate containers
    ALLOCATE(u(2*dim)); ALLOCATE(r(2*dim))

    ! Call uniform deviates
    CALL random_number(u)

    ! Apply Box-Muller transform
    DO g = 1, 2*dim, 2
    
            r(g)   = SQRT(-2*LOG(u(g)))*COS(2*pi*u(g+1)) 
            r(g+1) = SQRT(-2*LOG(u(g)))*SIN(2*pi*u(g+1)) 
    
    END DO

    ! Extract relevant floats
    DO g = 1, dim 

            draw(g) = r(g)     
    
    END DO

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE LINDS(N, A, LDA, AINV, LDAINV)

    !/* external objects    */

    REAL, INTENT(OUT)       :: AINV(N, N)

    INTEGER, INTENT(IN)     :: LDAINV
    INTEGER, INTENT(IN)     :: LDA
    INTEGER, INTENT(IN)     :: N
    
    REAL, INTENT(IN)        :: A(N, N)

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 
    
    AINV = inverse(A, N)

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
FUNCTION inverse(A, n)

    !/* external objects        */

    INTEGER, INTENT(IN)     :: n

    REAL, INTENT(IN)        :: A(:, :)

    !/* internal objects        */

    INTEGER                 :: ipiv(n)
    INTEGER                 :: info

    REAL                    :: inverse(n, n)
    REAL                    :: work(n)
        
!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
        
    ! Initialize matrix for replacement
    inverse = A
    
    ! SGETRF computes an LU factorization of a general M-by-N matrix A
    ! using partial pivoting with row interchanges.
    CALL SGETRF(n, n, inverse, n, ipiv, info)

    ! SGETRI computes the inverse of a matrix using the LU factorization
    ! computed by DGETRF.
    CALL SGETRI(n, inverse, n, ipiv, work, n, info)

END FUNCTION
!*******************************************************************************
!*******************************************************************************
SUBROUTINE RNOPT(seed)

    !/* external objects    */

    INTEGER, INTENT(IN)     :: seed

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE RNGET(seed)

    !/* external objects    */

    INTEGER, INTENT(IN)     :: seed

    !/* internal objects    */

    INTEGER                 :: size

    INTEGER                 :: auxiliary(55)

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

    CALL RANDOM_SEED(size=size)

    auxiliary = seed

    CALL RANDOM_SEED(get=auxiliary)

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE RNSET(seed)

    !/* external objects    */

    INTEGER, INTENT(IN)     :: seed

    !/* internal objects    */

    INTEGER                 :: auxiliary(55)
    INTEGER                 :: size

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

    CALL RANDOM_SEED(size=size)

    auxiliary = seed

    CALL RANDOM_SEED(put=auxiliary)

END SUBROUTINE 
!*******************************************************************************
!*******************************************************************************
SUBROUTINE RNSRI(NSAMP, NPOP, INDEX)

    !/* external objects    */

    INTEGER, INTENT(OUT)        :: INDEX(:)

    INTEGER, INTENT(IN)         :: NSAMP
    INTEGER, INTENT(IN)         :: NPOP

    !/* internal objects    */

    INTEGER                     :: shuffled(NPOP)
    INTEGER                     :: initial(NPOP)
    INTEGER                     :: i

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

    DO i = 1, NPOP
        initial(i) = i
    END DO

    shuffled = initial
    CALL shuffle(shuffled)

    INDEX(:NSAMP) = shuffled(:NSAMP)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE shuffle(a)

    !/* external objects    */

    INTEGER, INTENT(INOUT)  :: a(:)
    
    !/* internal objects    */
    
    INTEGER                 :: randpos
    INTEGER                 :: temp
    INTEGER                 :: i

    REAL                    :: r

!------------------------------------------------------------------------------- 
! Algorithm
!------------------------------------------------------------------------------- 

    DO i = SIZE(a), 2, -1
        
        CALL RANDOM_NUMBER(r)

        randpos = int(r * i) + 1
        
        temp = a(randpos)
        
        a(randpos) = a(i)
        
        a(i) = temp
    
    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
END MODULE