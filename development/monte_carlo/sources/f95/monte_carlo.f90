PROGRAM monte_carlo

      INTEGER KSTATE(40,14000,4)
      INTEGER FSTATE(40,40,40,11,2)
      INTEGER KMAX(40)
      DIMENSION EMAX(40,14000),EMAX1(40,14000)
      DIMENSION BETA(2,6),A(4,4),RHO(4,4)
      DIMENSION SIGMA(4)
      DIMENSION EU1(5000,40),EU2(5000,40),C(5000,40),B(5000,40)
      DIMENSION RNN(4)
      DIMENSION PROB(40,4)
      DIMENSION PROB1(40,4)
      INTEGER X1,X2,E,T

  open(9,file='in1.txt')
 1500 FORMAT(1x,i3,1x,i5,1x,f7.0,1x,f6.0,1x,f6.2)

      READ(9,1500) NPER,NPOP,DRAW,DRAW1,TAU
!      WRITE(10,1500) NPER,NPOP,DRAW,DRAWW1,TAU
!C     GAMA= 0.577
!      WNA=-9.99
      DO  J=1,2
      READ(9,1501) (BETA(J,k),k=1,6)
!      WRITE(10,1501) (BETA(J,k),k=1,6)
 1501 FORMAT(6(1x,f10.6))
      END DO
!    1 continue
      READ(9,1502) CBAR1,CBAR2,CS,VHOME,DELTA
!      WRITE(10,1502) CBAR1,CBAR2,CS,VHOME,DELTA
 1502 FORMAT(5(1x,f10.5))
      DO   J=1,4
      READ(9,1503) (RHO(J,K),K=1,J)
!      WRITE(10,1503) (RHO(J,K),K=1,J)
 1503 FORMAT(4(1x,f10.5))
      END DO

      READ(9,1503) (SIGMA(J),J=1,4)
!      WRITE(10,1503) (SIGMA(J),J=1,4)

  

!*********************
!*  TRANSFORMATIONS  *
!*********************
      CBAR1 = CBAR1*1000.00
      CBAR2 = CBAR2*1000.00
      CS    = CS   *1000.00
      VHOME = VHOME*1000.00
      DO J=3,4             
        SIGMA(J) = SIGMA(J)*1000.0
      END DO

!PRINT *, SIGMA
!*********************************************************
!*  TAKE THE CHOLESKY DECOMPOSITION OF RHO AND PUT IN A  *
! !*********************************************************
!       DO J=2,4
!       DO K=1,J-1
!         RHO(K,J) = RHO(J,K)
!       END DO
!       END DO

!       !CALL LFCDS(4,RHO,4,A,4,COND)
      
!       DO J=2,4
!       DO K=1,J-1
!         A(J,K) = A(K,J)
!       END DO
!       END DO
      
!       !DO 7 J=1,4
!       !  WRITE(10,1503) (RHO(J,K),K=1,4)
!       !7 CONTINUE
       DO  J=1,4
         DO K=1,4
! !         A(J,K)=A(J,K)*SIGMA(J)
          A(J,K)=SIGMA(J)       ! This is amended as I do not have LFCDS, but should work for data one

         END DO
!       !  WRITE(10,1503) (A(J,K),K=1,4)
       END DO


!****************************
!*  CREATE THE STATE INDEX  *
!****************************
DO T = 1, NPER
  K=0
        DO E=10,20
          IF(E.GT.10+T-1) THEN
            CYCLE
        END IF



        DO X1=0,T-1
        DO X2=0,T-1
          IF(X1+X2+E-10.LT.T) THEN
             DO LS=0,1 
                
                IF((LS.eq.0).and.((E-T).eq.9)) THEN
                  CYCLE
                END IF
                  
                IF((LS.eq.1).and.(E.eq.10).and.(T.gt.1)) THEN
                  CYCLE
                END IF
 
             K=K+1
             KSTATE(T,K,1)=X1
             KSTATE(T,K,2)=X2
             KSTATE(T,K,3)=E
             KSTATE(T,K,4)=LS
             FSTATE(T,X1+1,X2+1,E-9,LS+1)=K
             END DO
          ENDIF
  END DO
  END DO
  END DO


        KMAX(T)=K
END DO



PRINT *, KMAX

END PROGRAM 