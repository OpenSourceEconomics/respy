!******************************************************************************
!******************************************************************************
MODULE shared_lapack_interfaces

    INTERFACE

        SUBROUTINE DGESDD(JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, IWORK, INFO)
            
            !/* external modules    */

            USE shared_constants

            !/* external objects    */

            INTEGER(our_int), INTENT(OUT)   :: INFO

            REAL(our_dble), INTENT(INOUT)   :: A(LDA, *)
            REAL(our_dble), INTENT(OUT)     :: S(*)
            REAL(our_dble), INTENT(OUT)     :: U(LDU, *)
            REAL(our_dble), INTENT(OUT)     :: VT(LDVT,*)
            REAL(our_dble), INTENT(OUT)     :: WORK(*)

            CHARACTER(1), INTENT(IN)        :: JOBZ

            INTEGER(our_int), INTENT(IN)    :: M
            INTEGER(our_int), INTENT(IN)    :: N
            INTEGER(our_int), INTENT(IN)    :: LDA
            INTEGER(our_int), INTENT(IN)    :: LDU
            INTEGER(our_int), INTENT(IN)    :: LDVT
            INTEGER(our_int), INTENT(IN)    :: LWORK

            INTEGER(our_int), INTENT(IN)    :: IWORK(*)

        END SUBROUTINE


        SUBROUTINE DPOTRF(UPLO, N, A, LDA, INFO)

            !/* external modules    */

            USE shared_constants

            !/* external objects    */

            REAL(our_dble), INTENT(INOUT)   :: A(N, N)
 
            INTEGER(our_int), INTENT(IN)    :: INFO
            INTEGER(our_int), INTENT(IN)    :: LDA
            INTEGER(our_int), INTENT(IN)    :: N

            CHARACTER(1), INTENT(IN)        :: UPLO

        END SUBROUTINE

   
        SUBROUTINE DGETRF(M, N, A, LDA, IPIV, INFO)

            !/* external modules    */

            USE shared_constants

            !/* external objects    */

            INTEGER(our_int), INTENT(OUT)   :: IPIV(N)
            INTEGER(our_int), INTENT(OUT)   :: INFO

            REAL(our_dble), INTENT(INOUT)   :: A(N, N)

            INTEGER(our_int), INTENT(IN)    :: LDA
            INTEGER(our_int), INTENT(IN)    :: M
            INTEGER(our_int), INTENT(IN)    :: N

         END SUBROUTINE

   
        SUBROUTINE DGETRI(N, A, LDA, IPIV, WORK, LWORK, INFO)
            
            !/* external modules    */

            USE shared_constants

            !/* external objects    */

            REAL(our_dble), INTENT(INOUT)   :: WORK(LWORK)
            REAL(our_dble), INTENT(INOUT)   :: A(N, N)

            INTEGER(our_int), INTENT(IN)    :: IPIV(N)
            INTEGER(our_int), INTENT(IN)    :: LWORK
            INTEGER(our_int), INTENT(IN)    :: INFO
            INTEGER(our_int), INTENT(IN)    :: LDA
            INTEGER(our_int), INTENT(IN)    :: N

         END SUBROUTINE

    END INTERFACE

!******************************************************************************
!******************************************************************************
END MODULE