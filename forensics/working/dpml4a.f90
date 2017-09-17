!*******************************************************************************
!*******************************************************************************
PROGRAM dpml4a

  ! PEI: Interface to IMSL replacements
  USE IMSL_REPLACEMENTS

!**************************************************
!*  PROGRAM TO ESTIMATE DP MODEL BY SIMULATED ML  *
!**************************************************
!********************
!*  WAGES INCLUDED  *
!********************
!****************
!*  TRANSFORMS  *
!****************
!***************************************************
!*  ITERATE ON ELEMENTS OF CHOLESKY DECOMPOSITION  *
!***************************************************
!*******************************
!*  SET UP TO DO REPLICATIONS  *
!*******************************
!**********************************************
!*  ONLY SIMULATE HOME AND SCHOOL PROBS ONCE  *
!**********************************************
!********************************
!*  INTERPOLATE EXPECTED MAX'S  *
!********************************
!**************************************************************
!*  INCLUDE TUITION COST AND LAGGED SCHOOL AS STATE VARIABLE  *
!**************************************************************
      INTEGER KSTATE(13150,4)
      INTEGER FSTATE(40,40,11,2),FSTATE1(40,40,11,2)
      INTEGER KMAX,KKMAX
      DIMENSION EMAX(13150),EMAX1(13150)
      DIMENSION RMAXV(13150)
      DIMENSION ICOUNT(13150,4)
      DIMENSION ICOUNT0(13150)
      DIMENSION PROBI3(13150),PROBI4(13150)
      DIMENSION BETA(2,6),A(4,4),RHO(4,4),SIGMA(4)
!C***DIMENSIONS ALLOW FOR 27 VARIABLES IN MODEL
      DIMENSION CPARM(27),SPARM(27),CPARMB(27)
      DIMENSION CPARMT(27),CPARMS(27)
      DIMENSION IPARM(27)
      CHARACTER*8 NAME(31)
      DIMENSION DELTAP(27),FOC(27),RLOGLF(28)
      DIMENSION STDERR(27),TSTAT(27),C1(27)
      DIMENSION SPDM(27,27),SPDMI(27,27)
      DIMENSION SPDMR(27,27),SPDMRI(27,27)
!C***DIMENSIONS ALLOW FOR 1000 DRAWS AND UP TO 40 PERIODS
      DIMENSION EU1(1000,40),EU2(1000,40),C(1000,40),B(1000,40)
      DIMENSION RV(4)
      DIMENSION RNNL(1000,40,4),RNN(1000,40,4)
!C***DIMENSIONS ALLOW FOR 1000 PEOPLE AND UP TO 40 PERIODS
      DIMENSION RLLFI(1000,28)
      INTEGER TIME(1000)
      INTEGER STATE(1000,40),EDUC(1000,40),LSCHL(1000,40)
      INTEGER EXPER(1000,40,2)
      DIMENSION WAGE(1000,40)
!C***DIMENSIONS ALLOW FOR 10 VARIABLES in INTERPOLATING REGRESSION
      DIMENSION RGAMA(10),SEGAMA(10),VGAMA(10)
      DIMENSION XY(10),XR(10)
      DIMENSION XX(10,10),XXI(10,10)
      DIMENSION XXIV(100)
!C***ALLOWS FOR FORTY PERIODS AND 500 POINTS IN INTERPOLATING
      DIMENSION EMAXS(500)
      INTEGER TK(500,40)
      INTEGER TK1(500)
      DIMENSION Y(500)
      INTEGER X1,X2,E,T
      INTEGER TM20,T20
      CHARACTER*4 TRANS

      ! PEI: Set up file connectors
      open(9,file='in1.txt'); open(10,file='ftest.txt')
      open(11,file='output1.txt'); open(12,file='in1new.txt')
      open(13,file='seed.txt'); open(14,file='PARMtest.txt')
      open(15,file='STDERRtest.txt'); open(17,file='emax1test.txt')

      READ(9,1500) NPER,NPOP,DRAW,DRAW1,TAU,NITER,MAXIT,MAXSTP,TRANS,INTP,PMIN
      WRITE(11,1500) NPER,NPOP,DRAW,DRAW1,TAU,NITER,MAXIT,MAXSTP,TRANS,INTP,PMIN
      WRITE(12,1500) NPER,NPOP,DRAW,DRAW1,TAU,NITER,MAXIT,MAXSTP,TRANS,INTP,PMIN
 1500 FORMAT(1x,i3,1x,i5,1x,f7.0,1x,f6.0,1x,f6.0,1x,i2,1x,i2,1X,I2,1x,A3,1x,I5,1x,F4.0)
!C     DRAW1=DRAW
      NPARM=27
      PMIN = 1.0/(10.0**PMIN)
!C***SET NUMBER OF PARAMS IN INTERPOLATING REGRESSION
      NPTA=8
      NPT=8
!C***SET NUMBER OF INTERPOLATING POINTS
!C     INTP=50
!C     INTP=100
!C     INTP=250
!C     INTP=500
!C     INTP=1000
!C     INTP=2000
!C     INTP=13150
!C**CHOOSE PERIOD AT WHICH INTERPOLATION BEGINS
      IF(INTP.EQ.50) INTPER=5
      IF(INTP.EQ.100) INTPER=7
      IF(INTP.EQ.150) INTPER=8
      IF(INTP.EQ.200) INTPER=8
      IF(INTP.EQ.250) INTPER=9
      IF(INTP.EQ.500) INTPER=11
      IF(INTP.EQ.1000) INTPER=15
      IF(INTP.EQ.2000) INTPER=19
      IF(INTP.EQ.3000) INTPER=22
      IF(INTP.EQ.13150) INTPER=41
!C     INTPER = 3
      ALPHA=0.1
!C     GAMA=0.57
      WNA=-9.99
      PI=3.141592654
!*****************************************************************
!* READ IN REPLICATION NUMBER AND THE SEEDS FOR THIS REPLICATION *
!*****************************************************************
      READ(13,1313) IRUN,ISEED,ISEED1,ISEED2
      WRITE(11,1313) IRUN,ISEED,ISEED1,ISEED2
 1313 FORMAT(I4,1x,I10,1x,I10,1x,I10)
!*****************************
!*  READ IN STARTING VALUES  *
!*****************************
      WRITE(11,1504)
 1504 FORMAT(' STARTING PARAMETER VECTOR:')
      DO 1 J=1,2
      READ(9,1501) (BETA(J,K),K=1,6)
      WRITE(11,1501) (BETA(J,K),K=1,6)
 1501 FORMAT(6(1x,f10.6))
    1 CONTINUE
      READ(9,1502) CBAR1,CBAR2,CS,VHOME,DELTA
      WRITE(11,1502) CBAR1,CBAR2,CS,VHOME,DELTA
 1502 FORMAT(5(1x,f10.5))
      DO 2 J=1,4
      READ(9,1503) (RHO(J,K),K=1,J)
      WRITE(11,1503) (RHO(J,K),K=1,J)
 1503 FORMAT(4(1x,f10.5))
    2 CONTINUE
      READ(9,1503) (SIGMA(J),J=1,4)
      WRITE(11,1503) (SIGMA(J),J=1,4)
!*******************
!*  READ IN NAMES  *
!*******************
      WRITE(11,1607)
 1607 FORMAT(' NAMES OF PARAMETERS:')
      DO 201 J=1,2
       READ(9,1601) (NAME(K),K=(J-1)*6+1,J*6)
       WRITE(11,1601) (NAME(K),K=(J-1)*6+1,J*6)
 1601  FORMAT(6(1X,A8))
  201 CONTINUE
      READ(9,1602) (NAME(K),K=2*6+1,2*6+5)
      WRITE(11,1602) (NAME(K),K=2*6+1,2*6+5)
 1602 FORMAT(5(1X,A8))
      DO 202 J=1,4
       READ(9,1603) (NAME(K),K=2*6+5+J*(J-1)/2+1,2*6+5+J*(J+1)/2)
       WRITE(11,1603) (NAME(K),K=2*6+5+J*(J-1)/2+1,2*6+5+J*(J+1)/2)
 1603  FORMAT(4(1X,A8))
  202 CONTINUE
      READ(9,1603) (NAME(K),K=2*6+5+4*(4+1)/2+1,2*6+5+4*(4+1)/2+4)
      WRITE(11,1603) (NAME(K),K=2*6+5+4*(4+1)/2+1,2*6+5+4*(4+1)/2+4)
!**************************
!*  READ IN IPARM VECTOR  *
!**************************
      WRITE(11,1608)
 1608 FORMAT(' PARAMETERS ITERATED ON:')
      DO 203 J=1,2
       READ(9,1604) (IPARM(K),K=(J-1)*6+1,J*6)
       WRITE(11,1604) (IPARM(K),K=(J-1)*6+1,J*6)
 1604  FORMAT(6I1)
  203 CONTINUE
      READ(9,1605) (IPARM(K),K=2*6+1,2*6+5)
      WRITE(11,1605) (IPARM(K),K=2*6+1,2*6+5)
 1605 FORMAT(5I1)
      DO 204 J=1,4
      READ(9,1606) (IPARM(K),K=2*6+5+J*(J-1)/2+1,2*6+5+J*(J+1)/2)
      WRITE(11,1606)  (IPARM(K),K=2*6+5+J*(J-1)/2+1,2*6+5+J*(J+1)/2)
 1606  FORMAT(4I1)
  204 CONTINUE
!*******************************************************
!*  USE RHO AND SIGMA TO CONSTRUCT COVARIANCE MATRIX.  *
!*  TAKE CHOLESKY DECOMPOSITION OF COVARIANCE MATRIX.  *
!*******************************************************
      DO 105 J=2,4
      DO 106 K=1,J-1
       RHO(K,J) = RHO(J,K)
  106 CONTINUE
  105 CONTINUE
      DO 205 J=1,4
      DO 206 K=1,4
       RHO(J,K) = RHO(J,K)*SIGMA(J)*SIGMA(K)
  206 CONTINUE
  205 CONTINUE
      CALL LFCDS(4,RHO,4,A,4,COND)
      DO 107 J=2,4
      DO 108 K=1,J-1
       A(J,K) = A(K,J)
  108 CONTINUE
  107 CONTINUE
!************************************
!*  SET UP THE INITIAL PARM VECTOR  *
!************************************
      NP = 0
      DO 3 J=1,2
      DO 4 K=1,6
       NP = NP + 1
       CPARM(NP) = BETA(J,K)
    4 CONTINUE
    3 CONTINUE
       NP = NP + 1
       CPARM(NP) = CBAR1
       NP = NP + 1
       CPARM(NP) = CBAR2
       NP = NP + 1
       CPARM(NP) = CS
       NP = NP + 1
       CPARM(NP) = VHOME
       NP = NP + 1
       CPARM(NP) = DELTA
      DO 5 J=1,4
      DO 6 K=1,J
       NP = NP + 1
       CPARM(NP) = A(J,K)
    6 CONTINUE
    5 CONTINUE
!********************************************
!*  WRITE OUT TRANSFORMED PARAMETER VECTOR  *
!********************************************
      DO 7000 JJ = 1,NPARM
       IF(TRANS.EQ.'YES') THEN
        IF(JJ.EQ.17) THEN
          CPARMS(JJ) = (1.0/CPARM(JJ)) - 1.0
          GOTO 7000
        ENDIF
        CPARMS(JJ) = CPARM(JJ)
       ELSE
        CPARMS(JJ) = CPARM(JJ)
       ENDIF
 7000 CONTINUE
      WRITE(11,7100)
 7100 FORMAT(' TRANSFORMED STARTING PARAMETER VECTOR:')
      WRITE(11,4001) (CPARMS(JJ),JJ=1,NPARM)
!******************
!*  READ IN DATA  *
!******************
!C***SKIP TO THE PROPER PLACE IN THE DATA FILE FOR THIS REPLICATION
      DO 7777 J=1,4000*(IRUN-1)
       READ(10,*)
 7777 CONTINUE
      DO 7 I=1,NPOP
      READ(10,1000) IPPP,TIME(I),STATE(I,1),WAGE(I,1),EXPER(I,1,1),EXPER(I,1,2),EDUC(I,1),LSCHL(I,1)
       IF(I.eq.1) WRITE(11,1002) IRUN,IPPP,TIME(I)
 1002  FORMAT(' REPLICATION ',I4,' STARTS WITH PERSON ',I4,' PERIOD ',I2,/)
       DO 8 T=2,TIME(I)
       READ(10,1001)STATE(I,T),WAGE(I,T),EXPER(I,T,1),EXPER(I,T,2),EDUC(I,T),LSCHL(I,T)
    8  CONTINUE
    7  CONTINUE
 1000 FORMAT(1X,I5,1X,I3,1X,I1,1X,F10.2,4(1X,I3))
 1001 FORMAT(11X,I1,1X,F10.2,4(1X,I3))
!*****************************************************************
!*  DETERMINE NUMBER OF PEOPLE IN EACH STATE MAKING EACH CHOICE  *
!*****************************************************************
!CCC   DO 707 T=1,NPER
!CCC     K=0
!CCC     DO 708 E=10,20
!CCC      IF(E.GT.10+T-1) GOTO 708
!CCC     DO 709 X1=0,T-1
!CCC     DO 710 X2=0,T-1
!CCC      IF(X1+X2+E-10.LT.T) THEN
!CCC        DO 712 LS=0,1
!CCC        IF((LS.EQ.0).AND.((E-T).EQ.9)) GOTO 712
!CCC        IF((LS.EQ.1).AND.(E.EQ.10).AND.(T.GT.1)) GOTO 712
!CCC        K=K+1
!CCC        KSTATE(K,1)=X1
!CCC        KSTATE(K,2)=X2
!CCC        KSTATE(K,3)=E
!CCC        KSTATE(K,4)=LS
!CCC        FSTATE(X1+1,X2+1,E-9,LS+1)=K
!CCC        ICOUNT0(K)=0
!CCC        DO 711 J=1,4
!CCC         ICOUNT(K,J)=0
!CC711      CONTINUE
!CC712      CONTINUE
!CCC      ENDIF
!CC710   CONTINUE
!CC709   CONTINUE
!CC708   CONTINUE
!CCC     KMAX=K
!CCC     DO 712 I=1,NPOP
!CCC      X1=EXPER(I,T,1)
!CCC      X2=EXPER(I,T,2)
!CCC      E=EDUC(I,T)
!CCC      LS=LSCHL(I,T)
!CCC      K=FSTATE(X1+1,X2+1,E-9,LS+1)
!CCC      J=STATE(I,T)
!CCC      ICOUNT(K,J)=ICOUNT(K,J)+1
!CCC      ICOUNT0(K)=ICOUNT0(K)+1
!CC712   CONTINUE
!CCC     DO 713 K=1,KMAX
!CCC      IF(ICOUNT0(K).GT.0) THEN
!CCC       WRITE(11,1715) T,K,(ICOUNT(K,J),J=1,4)
!C1715     FORMAT(' T=',I2,' K=',I4,' COUNTS=',4I6)
!CCC      ENDIF
!CC713   CONTINUE
!CC707 CONTINUE
!***************************
!*  DRAW RANDOM VARIABLES  *
!***************************
!C***SEED IS IN RANGE (1,2147483646)
!C     CALL RNSET(1010101010)
      CALL RNSET(ISEED)
!C***DRAW THE RANDOM VARIABLES FOR SIMULATING THE DP SOLUTION
      DO 9 T=1,NPER
      DO 10 J=1,DRAW
!C     DO 10 T=1,NPER
       CALL RNNOR(4,RV)
       RNN(J,T,1)=RV(1)
       RNN(J,T,2)=RV(2)
       RNN(J,T,3)=RV(3)
       RNN(J,T,4)=RV(4)
   10 CONTINUE
    9 CONTINUE
      CALL RNGET(ISEED)
!C     CALL RNSET(2020202020)
      CALL RNSET(ISEED1)
!C***DRAW THE RANDOM VARIABLES FOR SIMULATING THE LIKELIHOOD
      DO 11 T=1,NPER
      DO 12 J=1,DRAW1
       CALL RNNOR(4,RV)
       RNNL(J,T,1)=RV(1)
       RNNL(J,T,2)=RV(2)
       RNNL(J,T,3)=RV(3)
       RNNL(J,T,4)=RV(4)
   12 CONTINUE
   11 CONTINUE
      CALL RNGET(ISEED1)
!********************
!*  ITERATION LOOP  *
!********************
      BUMP = 0.00001
!C     BUMP = 0.00100
      DO 5000 ITER=1,MAXIT
      SSIZE = 0.50
!****************
!*  step loop   *
!****************
       ISTEP = -1
       IGOOD = 0
       IBAD = 0
 5001 CONTINUE
       ISTEP = ISTEP + 1
       WRITE(11,4000) ITER,ISTEP
 4000  FORMAT(/,' ITERATION = ',I2,'  STEP =',I2,/)
!**********************************************
!*  IF ISTEP IS 0 THEN CALCULATE DERIVATIVES  *
!**********************************************
      IP = 0
      IF(ISTEP.EQ.0) NP = 0
      IF(ISTEP.GT.0) NP = NPARM
 6000 CONTINUE
      NP = NP + 1
!C     WRITE(11,4005) NP
!C4005 FORMAT(' NP = ',I2)
!C***WHEN NP=NPARM+1 CALCULATE THE LOG LIKELIHOOD
      IF(NP.EQ.NPARM+1) GOTO 6001
!C***JUMP TO 6003 AFTER DERIVS AND LIKELIHOOD HAVE BEEN CONSTRUCTED
      IF(NP.EQ.NPARM+2) GOTO 6003
!C***WHEN NP.LE.NPARM CALCULATE NUMERICAL DERIVATIVE FOR NP
      IF(IPARM(NP).EQ.0) GOTO 6000
      IP = IP + 1
!**********************************************
!*  IF ISTEP IS 0 THEN CALCULATE DERIVATIVES  *
!**********************************************
!C     WRITE(11,4006) NP,IP
!C4006 FORMAT(' WORKING ON DERIV FOR NP = ',I2,' IP = ',I2)
      DO 100 JJ = 1,NPARM
!C      IF(JJ.EQ.NP) CPARMB(JJ) = CPARM(JJ)*(1.0 + BUMP) + BUMP
!C      IF(JJ.EQ.NP) CPARMB(JJ) = CPARM(JJ) + BUMP
       IF(JJ.EQ.NP) THEN
        IF(TRANS.EQ.'YES') THEN
         IF(JJ.EQ.17) THEN
           CT = (1.0/CPARM(JJ)) - 1.0
           CT = CT + BUMP
           CPARMB(JJ) = 1.0/(1.0+CT)
           GOTO 100
         ENDIF
         CPARMB(JJ) = CPARM(JJ) + BUMP
        ELSE
         CPARMB(JJ) = CPARM(JJ) + BUMP
        ENDIF
       ENDIF
       IF(JJ.NE.NP) CPARMB(JJ) = CPARM(JJ)
  100 CONTINUE
      L = 0
      DO 101 J=1,2
      DO 102 K=1,6
       L = L + 1
       BETA(J,K) = CPARMB(L)
  102 CONTINUE
  101 CONTINUE
       L = L + 1
       CBAR1 = CPARMB(L)
       L = L + 1
       CBAR2 = CPARMB(L)
       L = L + 1
       CS = CPARMB(L)
       L = L + 1
       VHOME = CPARMB(L)
       L = L + 1
       DELTA = CPARMB(L)
      DO 103 J=1,4
      DO 104 K=1,J
       L = L + 1
       A(J,K) = CPARMB(L)
  104 CONTINUE
  103 CONTINUE
      GOTO 6002
 6001 CONTINUE
      IP = NITER + 1
!*********************************************************
!*  IF STEP IS GREATER THAN 0 THEN SET PARAMETER VALUES  *
!*********************************************************
!C     WRITE(11,4007) NP,IP
!C4007 FORMAT(' WORKING ON LIKELIHOOD FUNCTION,'
!C    *    ' NP = ',I2,'IP=',I2)
!CCC   WRITE(11,4407)
!C4407 FORMAT(' EVALUATING LIKELIHOOD FUNCTION AT:')
!CCC   WRITE(11,4001) (CPARM(IP1),IP1=1,NPARM)
      L = 0
      DO 109 J=1,2
      DO 110 K=1,6
       L = L + 1
       BETA(J,K) = CPARM(L)
  110 CONTINUE
  109 CONTINUE
       L = L + 1
       CBAR1 = CPARM(L)
       L = L + 1
       CBAR2 = CPARM(L)
       L = L + 1
       CS = CPARM(L)
       L = L + 1
       VHOME = CPARM(L)
       L = L + 1
       DELTA = CPARM(L)
      DO 111 J=1,4
      DO 112 K=1,J
       L = L + 1
       A(J,K) = CPARM(L)
  112 CONTINUE
  111 CONTINUE
 6002 CONTINUE
!*******************************
!*  SET SCALING OF PARAMETERS  *
!*******************************
      VHOME = VHOME*1000.0
      CBAR1 = CBAR1*1000.0
      CBAR2 = CBAR2*1000.0
      CS = CS*1000.0
      SIGMA(1) = A(1,1)
      SIGMA(2) = SQRT(A(2,1)**2 + A(2,2)**2)
      SIGMA(3) = SQRT(A(3,1)**2 + A(3,2)**2 + A(3,3)**2)
      SIGMA(4) = SQRT(A(4,1)**2 + A(4,2)**2 + A(4,3)**2+A(4,4)**2)
!*****************************************************
!*  WRITE OUT THE CORRELATION MATRIX AND THE SIGMAS  *
!*  IMPLIED BY THE A MATRIX                          *
!*****************************************************
      DO 7002 J=1,4
      DO 7003 K=1,J
       RHO(J,K) = 0.0
      DO 7004 L=1,K
       RHO(J,K) = RHO(J,K) + A(J,L)*A(K,L)
 7004 CONTINUE
 7003 CONTINUE
 7002 CONTINUE
      DO 7005 J=1,4
      DO 7006 K=1,J
       RHO(J,K) = RHO(J,K)/(SIGMA(J)*SIGMA(K))
 7006 CONTINUE
 7005 CONTINUE
      IF(NP.EQ.NPARM+1) THEN
!C     DO 1120 J=1,4
!C     WRITE(11,1121) (A(J,K),K=1,J)
!C1121 FORMAT(' A=',4(F15.10))
!C1120 CONTINUE
      DO 7007 J=1,4
      WRITE(11,7008) (RHO(J,K),K=1,J)
 7008 FORMAT(' RHO=',4F10.3)
 7007 CONTINUE
      WRITE(11,7009) (SIGMA(J),J=1,4)
 7009 FORMAT(' SIGMA=',4F10.3,/)
      ENDIF
      DO 1115 J=3,4
      DO 1116 K=1,J
        A(J,K) = A(J,K)*1000.0
 1116 CONTINUE
 1115 CONTINUE
!C****INITIALIZE LOG LIKELIHOOD FUNCTION
      RLOGLF(IP)=0.0
      DO 117 I=1,NPOP
        RLLFI(I,IP)=0.0
  117 CONTINUE
!C     WRITE(11,4008) NP,IP
!C4008 FORMAT(' EVALUATING LIKELIHOOD, NP = ',I2,' IP = ',I2)
!*****************************************************************
!*  CONSTRUCT THE EXPECTED MAX OF THE TIME NPER VALUE FUNCTIONS  *
!*****************************************************************
!C****CREATE THE STATE INDEX FOR TIME = NPER
      K=0
      DO 15 E=10,20
         IF(E.GT.10+NPER-1)  GO TO 15
      DO 16 X1=0,NPER-1
      DO 17 X2=0,NPER-1
        IF(X1+X2+E-10.LT.NPER) THEN
          DO 18 LS=0,1
          IF((LS.EQ.0).AND.((E-NPER).EQ.9)) GOTO 18
          IF((LS.EQ.1).AND.(E.EQ.10)) GOTO 18
           K=K+1
           KSTATE(K,1)=X1
           KSTATE(K,2)=X2
           KSTATE(K,3)=E
           KSTATE(K,4)=LS
           FSTATE(X1+1,X2+1,E-9,LS+1)=K
   18     CONTINUE
        ENDIF
   17 CONTINUE
   16 CONTINUE
   15 CONTINUE
      KMAX=K
!****************************************
!*  SET NUMBER OF INTERPOLATING POINTS  *
!****************************************
      IF(NPER.GE.INTPER) THEN
        IF(KMAX.GE.INTP) KKMAX=INTP
        IF(KMAX.LT.INTP) KKMAX=KMAX
       ELSE
        KKMAX=KMAX
      ENDIF
!C     write(11,1511) kkmax
!C1511 format(' kkmax=',i5)
!************************************
!*  chose the interpolating points  *
!************************************
      IF((ITER.EQ.1).AND.(ISTEP.EQ.0).AND.(IP.EQ.1)) THEN
      IF(NPER.GE.INTPER) THEN
!C      write(11,1512) NPER,INTPER
!C1512  format(' A: NPER=',I2,' INTPER=',I2)
       CALL RNOPT(1)
       CALL RNSET(ISEED2)
!C      write(11,9666) NPER,KMAX,ISEED2
!C9666  format(' CALLING RNSRI, T= ',I2,' KMAX=',I5,' SEED=',I10)
       CALL RNSRI(KKMAX,KMAX,TK1)
!C      write(11,9667) (TK1(k),k=1,50)
!C9667  format(8I7)
       CALL RNGET(ISEED2)
       DO 715 K=1,KKMAX
        TK(K,NPER) = TK1(K)
  715  CONTINUE
      ELSE
!C     write(11,1513) NPER,INTPER
!C1513 format(' B: NPER=',I2,' INTPER=',I2)
       DO 716 K=1,KMAX
        TK(K,NPER) = K
  716  CONTINUE
      ENDIF
      ENDIF
!C****INITIALIZE THE MATRICES FOR THE INTERPOLATING REGRESSION
      DO 520 IQ=1,NPTA
        XY(IQ) = 0.0
       DO 521 IR=1,NPTA
         XX(IQ,IR) = 0.0
  521  CONTINUE
  520 CONTINUE
!C***CONSTRUCT THE RANDOM VARIABLES FOR EACH ALTERNATIVE
      DO 122 J=1,DRAW
        RNN1=RNN(J,NPER,1)
        RNN2=RNN(J,NPER,2)
        RNN3=RNN(J,NPER,3)
        RNN4=RNN(J,NPER,4)
        U1        = A(1,1)*RNN1
        U2        = A(2,1)*RNN1+A(2,2)*RNN2
        C(J,NPER) = A(3,1)*RNN1+A(3,2)*RNN2+A(3,3)*RNN3
        B(J,NPER) = A(4,1)*RNN1+A(4,2)*RNN2+A(4,3)*RNN3+A(4,4)*RNN4
        EU1(J,NPER)=EXP(U1)
        EU2(J,NPER)=EXP(U2)
  122 CONTINUE
!****************************************************************
!*  SIMULATE THE EXPECTED MAX ONLY AT THE INTERPOLATING POINTS  *
!****************************************************************
      YBAR = 0.0
      DO 7021 KK=1,KKMAX
       K = TK(KK,NPER)
!C      IF(NP.EQ.NPARM+1) WRITE(11,4016) NPER,KK,K,E-9,X1,X2,LS
!C4016  FORMAT(' T=',I2,' KK=',I2,' K=',I5,' E=',I2,' X1=',I2,
!C    *    ' X2=',I2,' LS=',I2)
       X1=KSTATE(K,1)
       X2=KSTATE(K,2)
       E=KSTATE(K,3)
       LS=KSTATE(K,4)
       W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
       W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
       IF(E.GE.12) THEN
         CBAR = CBAR1 - CBAR2
        ELSE
         CBAR = CBAR1
       ENDIF
       IF(LS.EQ.0) CBAR = CBAR - CS
       IF(E.GT.19) CBAR = CBAR - 50000.0
      EMAX(K) = 0.0
      DO 22 J=1,DRAW
        V1=W1*EU1(J,NPER)
        V2=W2*EU2(J,NPER)
        V3=CBAR+C(J,NPER)
        V4=VHOME+B(J,NPER)
!C       WRITE(11,4148) J,RNN(J,NPER,1),RNN(J,NPER,2),
!C    *       RNN(J,NPER,3),RNN(J,NPER,4)
!C4148   FORMAT(/,' DRAW=',I2,' N1=',F10.4,' N2=',F10.4,' N3=',f10.4,
!C    *     ' N4=',F10.4)
!C       WRITE(11,4149) J,EU1(J,NPER),EU2(J,NPER),C(J,NPER),B(J,NPER)
!C4149   FORMAT(/,' DRAW=',I2,' E1=',F10.4,' E2=',F10.4,' E3=',f10.4,
!C    *     ' E4=',F10.4)
!C       WRITE(11,4150) J,V1,V2,V3,V4
!C4150   FORMAT(/,' DRAW=',I2,' V1=',F10.2,' V2=',F10.2,' V3=',f10.2,
!C    *     ' V4=',F10.2)
        VMAX=AMAX1(V1,V2,V3,V4)
!C       SUMV=EXP((V1-VMAX)/TAU)+EXP((V2-VMAX)/TAU)
!C    *      +EXP((V3-VMAX)/TAU)+EXP((V4-VMAX)/TAU)
        EMAX(K)=EMAX(K)+VMAX
!C       EMAX(K)=EMAX(K)
!C    *       + TAU*(GAMA+LOG(SUMV)+VMAX/TAU)
   22 CONTINUE
      EMAX(K) = EMAX(K)/DRAW
      EMAXS(KK) = EMAX(K)
!C     write(17,9921) KK,K,EMAX(K),EMAXS(KK)
      IF(NPER.GE.INTPER) THEN
      EV1 = W1*EXP(SIGMA(1)**2/2.0)
      EV2 = W2*EXP(SIGMA(2)**2/2.0)
      RMAXV(K) = AMAX1(EV1,EV2,CBAR,VHOME)
!C     write(11,1190) k,X1,X2,E,LS
!C1190 format(' k=',I5,' X1=',I2,' X2=',I2,' E=',I2,' LS=',I2)
!C     WRITE(11,1191) EV1,EV2,CBAR,VHOME
!C1191 format(' EV1=',f10.2,' EV2=',f10.2,' EV3=',f10.2,' EV4=',F10.2)
      IF(RMAXV(K).GT.EMAX(K)) EMAX(K) = RMAXV(K)
!C     ARG = (EMAX(K)-RMAXV(K))/RMAXV(K)
!C     write(11,1122) K,EMAX(K),RMAXV(K),ARG
!C1122 format(' K=',I5,' EMAX=',f10.2,' maxe=',f10.2,' y=',f10.6)
      Y(KK) = EMAX(K)-RMAXV(K)
      YBAR = YBAR + Y(KK)/KKMAX
      XR(1) = 1.0
      XR(2) = PEI_SQRT(0.00001 + RMAXV(K) - EV1)
      XR(3) = PEI_SQRT(0.00001 + RMAXV(K) - EV2)
      XR(4) = PEI_SQRT(0.00001 + RMAXV(K) - CBAR)
      XR(5) = PEI_SQRT(0.00001 + RMAXV(K) - VHOME)
      XR(6) = (RMAXV(K) - EV1)/100.0
      XR(7) = (RMAXV(K) - EV2)/100.0
!C     XR(8) = (RMAXV(K) - CBAR)/100.0
      XR(8) = (RMAXV(K) - VHOME)/100.0
      DO 522 IQ = 1,NPTA
       XY(IQ) = XY(IQ) + XR(IQ)*Y(KK)
       DO 523 IR = 1,NPTA
        XX(IQ,IR) = XX(IQ,IR) + XR(IQ)*XR(IR)
  523  CONTINUE
  522 CONTINUE
      ENDIF
 7021 CONTINUE
!**************************************
!*  RUN THE INTERPOLATING REGRESSION  *
!**************************************
      IF(NPER.GE.INTPER) THEN
!C     DO 5528 IQ=1,NPTA
!C     WRITE(11,5557) (XX(IQ,IR),IR=1,IQ)
!C5557 FORMAT(' XX=',5f12.1)
!C5528 CONTINUE
      CALL LINDS(NPTA,XX,10,XXI,10)
      DO 526 IQ=2,NPTA
      DO 527 IR=1,IQ-1
        XXI(IQ,IR) = XXI(IR,IQ)
  527 CONTINUE
  526 CONTINUE
      J=0
      DO 626 IQ=1,NPTA
      DO 627 IR=1,NPTA
        J=J+1
        XXIV(J) = XXI(IQ,IR)
  627 CONTINUE
  626 CONTINUE
!C     DO 5529 IQ=1,NPTA
!C     WRITE(11,5530) (XXI(IQ,IR),IR=1,IQ)
!C5530 FORMAT(' XXI=',8f9.6)
!C5529 CONTINUE
      DO 524 IQ=1,NPTA
         RGAMA(IQ) = 0.0
       DO 525 IR=1,NPTA
         RGAMA(IQ) = RGAMA(IQ) + XXI(IQ,IR)*XY(IR)
  525  CONTINUE
  524 CONTINUE
!C     WRITE(11,5553)
!C5553 FORMAT(20X,'  C     MAXE-EV1   MAXE-EV2   MAXE-EV3   MAXE-EV4',/,
!C    *       '    MAXE-EV1   MAXE-EV2   MAXE-EV3   MAXE-EV4')
!C     WRITE(11,5554) NPER,(RGAMA(IQ),IQ=1,NPTA)
!C***CALCULATE STANDARD ERRORS
!C     SSE = 0.0
!C     SST = 0.0
!C     DO 528 KK=1,KKMAX
!C      K=TK(KK,NPER)
!C      KKN=0
!C      DO 928 KKK=1,NPTA
!C       KKN = KKN + 1
!C 928  CONTINUE
!C      X1=KSTATE(K,1)
!C      X2=KSTATE(K,2)
!C      E=KSTATE(K,3)
!C      LS=KSTATE(K,4)
!C      W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2
!C    *        +BETA(1,5)*X2+BETA(1,6)*X2**2)
!C      W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2
!C    *        +BETA(2,5)*X2+BETA(2,6)*X2**2)
!C      IF(E.GE.12) THEN
!C       CBAR = CBAR1 - CBAR2
!C      ELSE
!C       CBAR = CBAR1
!C      ENDIF
!C      IF(LS.EQ.0) CBAR = CBAR - CS
!C      IF(E.GT.19) CBAR = CBAR - 50000.0
!C      RMAXV(K) = AMAX1(EV1,EV2,CBAR,VHOME)
!C      XR2 = SQRT(0.00001 + RMAXV(K) - EV1)
!C      XR3 = SQRT(0.00001 + RMAXV(K) - EV2)
!C      XR4 = SQRT(0.00001 + RMAXV(K) - CBAR)
!C      XR5 = SQRT(0.00001 + RMAXV(K) - VHOME)
!C      XR6 = (RMAXV(K) - EV1)/100.0
!C      XR7 = (RMAXV(K) - EV2)/100.0
!CC     XR8 = (RMAXV(K) - CBAR)/100.0
!C      XR8 = (RMAXV(K) - VHOME)/100.0
!C      XGAMA = RGAMA(1) + RGAMA(2)*XR2  + RGAMA(3)*XR3
!C    *                  + RGAMA(4)*XR4  + RGAMA(5)*XR5
!C    *                  + RGAMA(6)*XR6  + RGAMA(7)*XR7
!C    *                  + RGAMA(8)*XR8
!CC   *                  + RGAMA(8)*XR8  + RGAMA(9)*XR9
!C      EMAX(K) = RMAXV(K) + AMAX1(0.0,XGAMA)
!C      SSE = SSE + ( Y(KK) - XGAMA )**2
!C      SST = SST + ( Y(KK) - YBAR )**2
!C      WRITE(11,5700) NPER,KK,K,EMAX(K),RMAXV(K)
!C      WRITE(11,6554) KK,Y(KK),XGAMA,YBAR,SSE,SST
!C 528 CONTINUE
!C     RSQ=(SST-SSE)/SST
!C     DO 561 IQ=1,NPTA
!C       IR=IQ*(NPTA+1)-NPTA
!C       VGAMA(IQ) = XXIV(IR)*SSE/(KKMAX-NPTA)
!C 561 CONTINUE
!C     DO 571 IR=1,NPTA
!C       SEGAMA(IR) = SQRT(VGAMA(IR))
!C 571 CONTINUE
!C     WRITE(11,5555) NPER,(SEGAMA(IQ),IQ=1,NPTA)
!C     WRITE(11,6555) SSE,SST,RSQ,YBAR,INTP
!C6555 FORMAT(' SSE=',f12.3,' SST=',F12.3,' RSQ=',F6.4,' YBAR=',f8.2,
!C    *     ' INTP=',I5)
!*******************************************************
!*  USE THE INTERPOLATING REGRESSION TO CONSTRUCT THE  *
!*  FITTED EXPECTED MAXIMA                             *
!*******************************************************
      DO 529 K=1,KMAX
       X1=KSTATE(K,1)
       X2=KSTATE(K,2)
       E=KSTATE(K,3)
       LS=KSTATE(K,4)
       W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
       W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
       IF(E.GE.12) THEN
         CBAR = CBAR1 - CBAR2
        ELSE
         CBAR = CBAR1
       ENDIF
       IF(LS.EQ.0) CBAR = CBAR - CS
       IF(E.GT.19) CBAR = CBAR - 50000.0
       EV1 = W1*EXP(SIGMA(1)**2/2.0)
       EV2 = W2*EXP(SIGMA(2)**2/2.0)
       RMAXV(K) = AMAX1(EV1,EV2,CBAR,VHOME)
       XR2 = PEI_SQRT(0.00001 + RMAXV(K) - EV1)
       XR3 = PEI_SQRT(0.00001 + RMAXV(K) - EV2)
       XR4 = PEI_SQRT(0.00001 + RMAXV(K) - CBAR)
       XR5 = PEI_SQRT(0.00001 + RMAXV(K) - VHOME)
       XR6 = (RMAXV(K) - EV1)/100.0
       XR7 = (RMAXV(K) - EV2)/100.0
!C      XR8 = (RMAXV(K) - CBAR)/100.0
       XR8 = (RMAXV(K) - VHOME)/100.0
       XGAMA = RGAMA(1) + RGAMA(2)*XR2  + RGAMA(3)*XR3 &
                        + RGAMA(4)*XR4  + RGAMA(5)*XR5 &
                        + RGAMA(6)*XR6  + RGAMA(7)*XR7 &
                        + RGAMA(8)*XR8
!C    *                  + RGAMA(8)*XR8  + RGAMA(9)*XR9
       EMAX(K) = RMAXV(K) + AMAX1(0.0,XGAMA)
!CC     IF(NP.EQ.NPARM+1) write(11,4100) NPER,k,emax(k),RMAXV(K)
  529 CONTINUE
      DO 9021 KK=1,KKMAX
       K = TK(KK,NPER)
!C      write(17,9921) KK,K,EMAX(K),EMAXS(KK)
!C9921  format(' KK=',I2,' K=',I5,' EMAX=',F10.2,' EMAXS=',F10.2)
       EMAX(K) = EMAXS(KK)
 9021 CONTINUE
      ENDIF
!**************************************************************
!*  CONSTRUCT THE LOG LIKELIHOOD FUNCTION CONTIBUTIONS FOR    *
!*  ALL PEOPLE AT PERIOD NPER                                 *
!**************************************************************
!C     GOTO 998
      T = NPER
      DO 7160 K=1,KMAX
       ICOUNT0(K) = 0
 7160 CONTINUE
      DO 160 I=1,NPOP
         E=EDUC(I,T)
         X1=EXPER(I,T,1)
         X2=EXPER(I,T,2)
         J=STATE(I,T)
         LS=LSCHL(I,T)
         K=FSTATE(X1+1,X2+1,E-9,LS+1)
!C        WRITE(11,3160) I,T,J,E,X1,X2,LS
!C3160    FORMAT(' I=',I4,' T=',I2,' J=',I1,' E=',I2,' X1=',I2,
!C    *      ' X2=',I2,' LS=',I2)
         IF(E.GE.12) THEN
           CBAR = CBAR1 - CBAR2
          ELSE
           CBAR = CBAR1
         ENDIF
         IF(LS.EQ.0) CBAR = CBAR - CS
         IF(E.GT.19) CBAR = CBAR - 50000.0
!**********************************************************
!*  PROBS OF STATES 3 and 4 NEED ONLY BE SIMULATED ONCE   *
!*  (THAT IS, FOR ONE PERSON) AT EACH VALUE OF THE STATE  *
!*  SPACE FOR WHICH 3 OR 4 OCCURS                         *
!**********************************************************
         IF((J.EQ.3).OR.(J.EQ.4)) THEN
          ICOUNT0(K)=ICOUNT0(K)+1
          IF(ICOUNT0(K).EQ.1) THEN
           PROBI3(K)=0.0
           PROBI4(K)=0.0
          ENDIF
         ENDIF
!C****RETURN TO MAIN SECTION
         W1=EXP(BETA(1,1)+BETA(1,2)*E &
             +BETA(1,3)*X1+BETA(1,4)*X1**2 &
             +BETA(1,5)*X2+BETA(1,6)*X2**2)
         W2=EXP(BETA(2,1)+BETA(2,2)*E &
             +BETA(2,3)*X1+BETA(2,4)*X1**2 &
             +BETA(2,5)*X2+BETA(2,6)*X2**2)
!*****************************************************
!*  wage(1) = w1*eu1 = exp(xbeta)*exp(a(1,1)*rnnl1)  *
!*  w1               = exp(xbeta)                    *
!*  log(wage(1))     = xbeta + a(1,1)*rnnl1          *
!*  log(w1)          = xbeta                         *
!*  log(wage(1)) - log(w1) = a(1,1)*rnnl1            *
!*****************************************************
        IF(STATE(I,T).EQ.1) THEN
          U1L = LOG(WAGE(I,T))-LOG(W1)
          U1J = U1L
          RNNL1 = U1L/SIGMA(1)
          U2L   = A(2,1)*RNNL1
          CL    = A(3,1)*RNNL1
          BL    = A(4,1)*RNNL1
          V1=WAGE(I,T)
          PROBI=0.0
        ENDIF
        IF(STATE(I,T).EQ.2) THEN
          U2L=LOG(WAGE(I,T))-LOG(W2)
          U2J=U2L
          V2=WAGE(I,T)
          PROBI=0.0
        ENDIF
       IF(STATE(I,T).EQ.1) THEN
        DO 162 J=1,DRAW1
         RNNL2=RNNL(J,T,2)
         RNNL3=RNNL(J,T,3)
         RNNL4=RNNL(J,T,4)
         U2J = U2L + A(2,2)*RNNL2
         CJ  = CL  + A(3,2)*RNNL2+A(3,3)*RNNL3
         BJ  = BL  + A(4,2)*RNNL2+A(4,3)*RNNL3 + A(4,4)*RNNL4
         V2 = W2*EXP(U2J)
         V3 = CBAR+CJ
         V4 = VHOME+BJ
         VMAX=AMAX1(V1,V2,V3,V4)
         A1=EXP((V1-VMAX)/TAU)
         A2=EXP((V2-VMAX)/TAU)
         A3=EXP((V3-VMAX)/TAU)
         A4=EXP((V4-VMAX)/TAU)
         SUMV=A1+A2+A3+A4
         PROB = A1/SUMV
         PROBI = PROBI + PROB
  162   CONTINUE
        PROBI = PROBI/DRAW1
       ENDIF
       IF(STATE(I,T).EQ.2) THEN
        DO 163 J=1,DRAW1
         RNNL1=RNNL(J,T,1)
         RNNL3=RNNL(J,T,3)
         RNNL4=RNNL(J,T,4)
         U1J   = A(1,1)*RNNL1
         RNNJ2 = (U2J-A(2,1)*RNNL1)/A(2,2)
         CJ    = A(3,1)*RNNL1+A(3,2)*RNNJ2+A(3,3)*RNNL3
         BJ    = A(4,1)*RNNL1+A(4,2)*RNNJ2+A(4,3)*RNNL3+A(4,4)*RNNL4
         V1 = W1*EXP(U1J)
         V3 = CBAR+CJ
         V4 = VHOME+BJ
         VMAX=AMAX1(V1,V2,V3,V4)
         A1=EXP((V1-VMAX)/TAU)
         A2=EXP((V2-VMAX)/TAU)
         A3=EXP((V3-VMAX)/TAU)
         A4=EXP((V4-VMAX)/TAU)
         SUMV=A1+A2+A3+A4
         PROB = A2/SUMV
         DENW = 0.3989423*EXP(-0.5*RNNJ2**2)/A(2,2)
         PROB = PROB*DENW
         PROBI = PROBI + PROB
  163   CONTINUE
        PROBI = PROBI/DRAW1
       ENDIF
       IF((STATE(I,T).EQ.3).OR.(STATE(I,T).EQ.4)) THEN
       IF(ICOUNT0(K).EQ.1) THEN
        DO 164 J=1,DRAW1
         RNNL1=RNNL(J,T,1)
         RNNL2=RNNL(J,T,2)
         RNNL3=RNNL(J,T,3)
         RNNL4=RNNL(J,T,4)
         U1J = A(1,1)*RNNL1
         U2J = A(2,1)*RNNL1+A(2,2)*RNNL2
         CJ  = A(3,1)*RNNL1+A(3,2)*RNNL2+A(3,3)*RNNL3
         BJ  = A(4,1)*RNNL1+A(4,2)*RNNL2+A(4,3)*RNNL3+A(4,4)*RNNL4
         V1 = W1*EXP(U1J)
         V2 = W2*EXP(U2J)
         V3 = CBAR+CJ
         V4 = VHOME+BJ
         VMAX=AMAX1(V1,V2,V3,V4)
         A1=EXP((V1-VMAX)/TAU)
         A2=EXP((V2-VMAX)/TAU)
         A3=EXP((V3-VMAX)/TAU)
         A4=EXP((V4-VMAX)/TAU)
         SUMV=A1+A2+A3+A4
         PROB3 = A3/SUMV
         PROB4 = A4/SUMV
         PROBI3(K) = PROBI3(K) + PROB3
         PROBI4(K) = PROBI4(K) + PROB4
  164   CONTINUE
        PROBI3(K) = PROBI3(K)/DRAW1
        PROBI4(K) = PROBI4(K)/DRAW1
       ENDIF
       ENDIF
      IF(STATE(I,T).eq.1) THEN
        DENW = 0.3989423*EXP(-0.5*(U1L/SIGMA(1))**2)/SIGMA(1)
        PROBI = PROBI*DENW
      ENDIF
      IF((STATE(I,T).eq.1).OR.(STATE(I,T).EQ.2)) THEN
      IF(PROBI.GT.PMIN) THEN
!C       WRITE(11,6656) I,T,STATE(I,T),PROBI,DENW
!C6656   FORMAT(' I=',I2,' T=',I2,' STATE=',I2,' PROBI=',F6.3,
!C    *     ' DENW=',F6.3)
        RLLFI(I,IP) = RLLFI(I,IP) + LOG(PROBI)
        RLOGLF(IP) = RLOGLF(IP) + LOG(PROBI)
       ELSE
        IF(NP.EQ.NPARM+1) WRITE(11,6655) I,T,STATE(I,T),PROBI
        RLLFI(I,IP) = RLLFI(I,IP) + LOG(PMIN)
        RLOGLF(IP) = RLOGLF(IP) + LOG(PMIN)
      ENDIF
      ENDIF
      IF(STATE(I,T).eq.3) THEN
      IF(PROBI3(K).GT.PMIN) THEN
!C       WRITE(11,6655) I,T,STATE(I,T),PROBI3(K)
        RLLFI(I,IP) = RLLFI(I,IP) + LOG(PROBI3(K))
        RLOGLF(IP) = RLOGLF(IP) + LOG(PROBI3(K))
       ELSE
        IF(NP.EQ.NPARM+1) WRITE(11,6655) I,T,STATE(I,T),PROBI3(K)
        RLLFI(I,IP) = RLLFI(I,IP) + LOG(PMIN)
        RLOGLF(IP) = RLOGLF(IP) + LOG(PMIN)
      ENDIF
      ENDIF
      IF(STATE(I,T).eq.4) THEN
      IF(PROBI4(K).GT.PMIN) THEN
!C       WRITE(11,6655) I,T,STATE(I,T),PROBI3(K)
        RLLFI(I,IP) = RLLFI(I,IP) + LOG(PROBI4(K))
        RLOGLF(IP) = RLOGLF(IP) + LOG(PROBI4(K))
       ELSE
        IF(NP.EQ.NPARM+1) WRITE(11,6655) I,T,STATE(I,T),PROBI4(K)
        RLLFI(I,IP) = RLLFI(I,IP) + LOG(PMIN)
        RLOGLF(IP) = RLOGLF(IP) + LOG(PMIN)
      ENDIF
      ENDIF
!C     IF(NP.EQ.NPARM+1) WRITE(11,4010) I,NPER,STATE(I,T),
!C    *   PROBI,RLOGLF(IP)
!C4010 FORMAT(' I=',I2,' T=',I2,' STATE=',i2,' PROB=',
!C    *   F11.8,' RLOGLF=',F12.3)
!C***END OF LOOP OVER PEOPLE
  160 CONTINUE
!C 998 CONTINUE
!***********************************************************
!*  CONSTRUCT THE EXPECTED MAX OF THE VALUE FUNCTIONS FOR  *
!*  PERIODS 2 THROUGH NPER-1                               *
!***********************************************************
      DO 30 IS=1,NPER-1
      T=NPER-IS
!C     WRITE(11,6330) T
!C6330 FORMAT(' WORKING ON PERIOD ',I2)
!C****SAVE THE EXPECTED MAXIMA AND FSTATE MATRIX FOR PERIOD T+1
      DO 35 K=1,KMAX
        EMAX1(K) = EMAX(K)
!C       IF(NP.EQ.NPARM+1) WRITE(11,4105) T,K,EMAX1(K),EMAX(K)
!C4105   FORMAT(' t=',I2,' k=',I2,' EMAX1=',F10.2,' EMAX=',F10.2)
   35 CONTINUE
      DO 36 E=10,20
        IF(E.GT.10+T) GOTO 36
      DO 37 X1=0,T
      DO 38 X2=0,T
        IF(X1+X2+E-10.LT.T+1) THEN
        DO 39 LS=0,1
        IF((LS.EQ.0).AND.((E-T-1).EQ.9)) GOTO 39
        IF((LS.EQ.1).AND.(E.EQ.10)) GOTO 39
          FSTATE1(X1+1,X2+1,E-9,LS+1) = FSTATE(X1+1,X2+1,E-9,LS+1)
!C         IF((X1.LE.2).and.(X2.LE.2)) WRITE(11,4106) X1,X2,E,LS,
!C    *        FSTATE1(X1+1,X2+1,E-9,LS+1),FSTATE(X1+1,X2+1,E-9,LS+1)
!C4106   FORMAT(' X1=',I2,' X2=',I2,' E=',I2,' LS=',I2,
!C    *        ' Fstate1=',I5,' Fstate=',I5)
   39   CONTINUE
        ENDIF
   38 CONTINUE
   37 CONTINUE
   36 CONTINUE
!CC    IF(T.EQ.1) GOTO 3000
!C****CREATE THE STATE INDEX FOR TIME = T
      K=0
      DO 40 E=10,20
         IF(E.GT.10+T-1)  GOTO 40
      DO 41 X1=0,T-1
      DO 42 X2=0,T-1
        IF(X1+X2+E-10.LT.T) THEN
         DO 43 LS=0,1
         IF((LS.EQ.0).AND.((E-T).EQ.9)) GOTO 43
         IF((LS.EQ.1).AND.(E.EQ.10).AND.(T.GT.1)) GOTO 43
           K=K+1
           KSTATE(K,1)=X1
           KSTATE(K,2)=X2
           KSTATE(K,3)=E
           KSTATE(K,4)=LS
           FSTATE(X1+1,X2+1,E-9,LS+1)=K
   43    CONTINUE
        ENDIF
   42 CONTINUE
   41 CONTINUE
   40 CONTINUE
      KMAX=K
!C     WRITE(11,5015) T,KMAX
!C5015 FORMAT(' T=',I2,' KMAX=',I5)
      IF(T.GE.INTPER) THEN
        IF(KMAX.GE.INTP) KKMAX=INTP
        IF(KMAX.LT.INTP) KKMAX=KMAX
      ELSE
        KKMAX=KMAX
      ENDIF
      IF((ITER.EQ.1).AND.(ISTEP.EQ.0).AND.(IP.EQ.1)) THEN
      IF(T.GE.INTPER) THEN
       CALL RNOPT(1)
       CALL RNSET(ISEED2)
       CALL RNSRI(KKMAX,KMAX,TK1)
       CALL RNGET(ISEED2)
!C      write(11,9667) (TK1(k),k=1,KKMAX)
       DO 717 K=1,KKMAX
        TK(K,T) = TK1(K)
  717  CONTINUE
      ELSE
       DO 718 K=1,KMAX
        TK(K,T) = K
  718  CONTINUE
      ENDIF
      ENDIF
!C     IF(T.eq.1) WRITE(11,7200) T,KMAX,FSTATE(1,1,1)
!C7200 FORMAT(' T=',I2,' KMAX=',I2,' FSTATE(1,1,1)=',I2)
      IF(T.EQ.1) GOTO 3000
!C     IF(NP.EQ.NPARM+1) WRITE(11,4015) T,KMAX
!C****INITIALIZE THE MATRICES FOR THE INTERPOLATING REGRESSION
      DO 550 IQ=1,NPT
        XY(IQ) = 0.0
       DO 551 IR=1,NPT
         XX(IQ,IR) = 0.0
  551  CONTINUE
  550 CONTINUE
!C****CONSTRUCT THE RANDOM VARIABLES FOR EACH ALTERNATIVE
      DO 153 J=1,DRAW
        RNN1=RNN(J,T,1)
        RNN2=RNN(J,T,2)
        RNN3=RNN(J,T,3)
        RNN4=RNN(J,T,4)
        U1     = A(1,1)*RNN1
        U2     = A(2,1)*RNN1+A(2,2)*RNN2
        C(J,T) = A(3,1)*RNN1+A(3,2)*RNN2+A(3,3)*RNN3
        B(J,T) = A(4,1)*RNN1+A(4,2)*RNN2+A(4,3)*RNN3+A(4,4)*RNN4
        EU1(J,T)=EXP(U1)
        EU2(J,T)=EXP(U2)
  153 CONTINUE
!*******************************************************************
!*  SIMULATE THE EXPECTED MAXIMA ONLY AT THE INTERPOLATING POINTS  *
!*******************************************************************
      YBAR = 0.0
      DO 7052 KK=1,KKMAX
      K = TK(KK,T)
!C     WRITE(11,5018) KK,K
!C5018 FORMAT(' KK=',I2,' K=',I5)
      X1=KSTATE(K,1)
      X2=KSTATE(K,2)
      E=KSTATE(K,3)
      LS=KSTATE(K,4)
      W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
      W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
      IF(E.GE.12) THEN
        CBAR = CBAR1 - CBAR2
       ELSE
        CBAR = CBAR1
      ENDIF
      IF(LS.EQ.0) CBAR = CBAR - CS
!C     IF(NP.EQ.NPARM+1) WRITE(11,4016) T,KK,K,E-9,X1,X2,LS
      E1 = DELTA*EMAX1(FSTATE1(X1+2,X2+1,E-9,1))
      E2 = DELTA*EMAX1(FSTATE1(X1+1,X2+2,E-9,1))
      IF(E.LE.19) E3 = CBAR + DELTA*EMAX1(FSTATE1(X1+1,X2+1,E-8,2))
      IF(E.GT.19) E3 = CBAR - 50000.0
      E4 = VHOME + DELTA*EMAX1(FSTATE1(X1+1,X2+1,E-9,1))
      EMAX(K) = 0.0
      DO 53 J=1,DRAW
!C       IF((T.EQ.36).and.(KK.GE.48)) THEN
!C         WRITE(11,1054) J,DRAW
!C       ENDIF
!C1054   FORMAT(' DRAW=',I4,' DRAWS=',F5.0)
        V1=W1*EU1(J,T)  + E1
        V2=W2*EU2(J,T)  + E2
        V3=C(J,T)       + E3
        V4=B(J,T)       + E4
        VMAX=AMAX1(V1,V2,V3,V4)
!C       SUMV=EXP((V1-VMAX)/TAU)+EXP((V2-VMAX)/TAU)
!C    *      +EXP((V3-VMAX)/TAU)+EXP((V4-VMAX)/TAU)
        EMAX(K)=EMAX(K)+VMAX
!C       EMAX(K)=EMAX(K)+TAU*(GAMA+LOG(SUMV)+VMAX/TAU)
!C     IF((T.EQ.37).and.(KK.GE.48))
!C    *      write(11,1053) T,K,J,V1,V2,V3,V4,EMAX(K)
!C1053 format(/,' T=',I2,' STATE=',I5,' DRAW=',I3,
!C    *       '      V1=',F13.3,' V2=',F13.3,' V3=',F13.3,
!C    *       ' V4=',F13.4,' EMAX=',F13.4)
   53 CONTINUE
      EMAX(K) = EMAX(K)/DRAW
      EMAXS(KK) = EMAX(K)
!C     write(17,9921) KK,K,EMAX(K),EMAXS(KK)
      IF(T.GE.INTPER) THEN
      EW1 = W1*EXP(SIGMA(1)**2/2.0)
      EW2 = W2*EXP(SIGMA(2)**2/2.0)
      RMAXV(K) = AMAX1(EW1+E1,EW2+E2,E3,E4)
      IF(RMAXV(K).GT.EMAX(K)) EMAX(K) = RMAXV(K)
!C     ARG = (EMAX(K)-RMAXV(K))/RMAXV(K)
!C     write(11,1190) k,X1,X2,E,LS
!C     WRITE(11,1191) E1+EW1,E2+EW2,E3,E4
!C     write(11,1122) K,EMAX(K),RMAXV(K),ARG
      Y(KK) = EMAX(K)-RMAXV(K)
      YBAR = YBAR + Y(KK)/KKMAX
      XR(1) = 1.0
      XR(2) = PEI_SQRT(0.00001 + RMAXV(K) - EW1 - E1)
      XR(3) = PEI_SQRT(0.00001 + RMAXV(K) - EW2 - E2)
      XR(4) = PEI_SQRT(0.00001 + RMAXV(K) - E3)
      XR(5) = PEI_SQRT(0.00001 + RMAXV(K) - E4)
      XR(6) = (RMAXV(K) - EW1 - E1)/100.0
      XR(7) = (RMAXV(K) - EW2 - E2)/100.0
!C     XR(8) = (RMAXV(K) - E3)/100.0
      XR(8) = (RMAXV(K) - E4)/100.0
      DO 552 IQ = 1,NPT
       XY(IQ) = XY(IQ) + XR(IQ)*Y(KK)
       DO 553 IR = 1,NPT
        XX(IQ,IR) = XX(IQ,IR) + XR(IQ)*XR(IR)
  553  CONTINUE
  552 CONTINUE
      ENDIF
 7052 CONTINUE
!**************************************
!*  RUN THE INTERPOLATING REGRESSION  *
!**************************************
      IF(T.GE.INTPER) THEN
!C     DO 5558 IQ=1,NPT
!C     WRITE(11,5557) (XX(IQ,IR),IR=1,IQ)
!C5558 CONTINUE
      CALL LINDS(NPT,XX,10,XXI,10)
      DO 557 IQ=2,NPT
      DO 558 IR=1,IQ-1
        XXI(IQ,IR) = XXI(IR,IQ)
  558 CONTINUE
  557 CONTINUE
      J=0
      DO 628 IQ=1,NPT
      DO 629 IR=1,NPT
        J=J+1
        XXIV(J) = XXI(IQ,IR)
  629 CONTINUE
  628 CONTINUE
!C     DO 5559 IQ=1,NPT
!C     WRITE(11,5530) (XXI(IQ,IR),IR=1,NPT)
!C5559 CONTINUE
      DO 554 IQ=1,NPT
        RGAMA(IQ) = 0.0
       DO 555 IR=1,NPT
         RGAMA(IQ) = RGAMA(IQ) + XXI(IQ,IR)*XY(IR)
  555  CONTINUE
  554 CONTINUE
!C     WRITE(11,5553)
!C     WRITE(11,5554) T,(RGAMA(IQ),IQ=1,NPT)
!C5554 FORMAT(' T=',I2,' GAMA=',5F12.6,/,11x,5F12.6)
!C***CALCULATE STANDARD ERRORS
!C     SSE = 0.0
!C     SST = 0.0
!C     DO 562 KK=1,KKMAX
!C      K=TK(KK,T)
!C      KKN=0
!C      DO 929 KKK=1,NPT
!C       KKN = KKN + 1
!C 929  CONTINUE
!C      K=TK(KK,T)
!C      X1=KSTATE(K,1)
!C      X2=KSTATE(K,2)
!C      E=KSTATE(K,3)
!C      LS=KSTATE(K,4)
!C      W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2
!C    *      +BETA(1,5)*X2+BETA(1,6)*X2**2)
!C      W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2
!C    *      +BETA(2,5)*X2+BETA(2,6)*X2**2)
!C      IF(E.GE.12) THEN
!C        CBAR = CBAR1 - CBAR2
!C       ELSE
!C        CBAR = CBAR1
!C      ENDIF
!C      IF(LS.EQ.0) CBAR = CBAR - CS
!C      EW1 = W1*EXP(SIGMA(1)**2/2.0)
!C      EW2 = W2*EXP(SIGMA(2)**2/2.0)
!C      V1=EW1   + DELTA*EMAX1(FSTATE1(X1+2,X2+1,E-9,1))
!C      V2=EW2   + DELTA*EMAX1(FSTATE1(X1+1,X2+2,E-9,1))
!C      IF(E.LE.19) THEN
!C        V3=CBAR  + DELTA*EMAX1(FSTATE1(X1+1,X2+1,E-8,2))
!C       ELSE
!C        V3=CBAR  - 50000.0
!C      ENDIF
!C      V4=VHOME + DELTA*EMAX1(FSTATE1(X1+1,X2+1,E-9,1))
!C      RMAXV(K) = AMAX1(V1,V2,V3,V4)
!C      XR2 = SQRT(0.00001 + RMAXV(K) - V1)
!C      XR3 = SQRT(0.00001 + RMAXV(K) - V2)
!C      XR4 = SQRT(0.00001 + RMAXV(K) - V3)
!C      XR5 = SQRT(0.00001 + RMAXV(K) - V4)
!C      XR6 = (RMAXV(K) - V1)/100.0
!C      XR7 = (RMAXV(K) - V2)/100.0
!CC     XR8 = (RMAXV(K) - V3)/100.0
!C      XR8 = (RMAXV(K) - V4)/100.0
!C      XGAMA = RGAMA(1) + RGAMA(2)*XR2  + RGAMA(3)*XR3
!C    *                  + RGAMA(4)*XR4  + RGAMA(5)*XR5
!C    *                  + RGAMA(6)*XR6  + RGAMA(7)*XR7
!C    *                  + RGAMA(8)*XR8
!CC   *                  + RGAMA(8)*XR8  + RGAMA(9)*XR9
!C      EMAX(K) = RMAXV(K) + AMAX1(0.0,XGAMA)
!C      SSE = SSE + ( Y(KK) - XGAMA )**2
!C      SST = SST + ( Y(KK) - YBAR )**2
!C      WRITE(11,6551) KK,K,X1,X2,E,LS
!C      WRITE(11,5700) T,KK,K,EMAX(K),RMAXV(K)
!C      WRITE(11,6554) KK,Y(KK),XGAMA,YBAR,SSE,SST
!C6554  FORMAT(' KK=',I2,' Y=',F8.3,' YHAT=',F8.3,' YBAR=',F8.3,
!C    *     ' SSE=',F8.3,' SST=',F8.3)
!C 562 CONTINUE
!C     RSQ = (SST - SSE)/SST
!C     DO 560 IQ=1,NPT
!C       IR=IQ*(NPT+1)-NPT
!C       VGAMA(IQ) = XXIV(IR)*SSE/(KKMAX-NPT)
!C 560 CONTINUE
!C     DO 570 IR=1,NPT
!C       SEGAMA(IR) = SQRT(VGAMA(IR))
!C 570 CONTINUE
!C     WRITE(11,5555) T,(SEGAMA(IQ),IQ=1,NPT)
!C5555 FORMAT(' T=',I2,' SE  =',5F12.6,/,11x,5F12.6)
!C     WRITE(11,6555) SSE,SST,RSQ,YBAR,INTP
!*******************************************************
!*  USE THE INTERPOLATING REGRESSION TO CONSTRUCT THE  *
!*  FITTED EXPECTED MAXIMA                             *
!*******************************************************
      DO 559 K=1,KMAX
       X1=KSTATE(K,1)
       X2=KSTATE(K,2)
       E=KSTATE(K,3)
       LS=KSTATE(K,4)
       W1=EXP(BETA(1,1)+BETA(1,2)*E+BETA(1,3)*X1+BETA(1,4)*X1**2+BETA(1,5)*X2+BETA(1,6)*X2**2)
       W2=EXP(BETA(2,1)+BETA(2,2)*E+BETA(2,3)*X1+BETA(2,4)*X1**2+BETA(2,5)*X2+BETA(2,6)*X2**2)
       IF(E.GE.12) THEN
         CBAR = CBAR1 - CBAR2
        ELSE
         CBAR = CBAR1
       ENDIF
       IF(LS.EQ.0) CBAR = CBAR - CS
       EW1 = W1*EXP(SIGMA(1)**2/2.0)
       EW2 = W2*EXP(SIGMA(2)**2/2.0)
       V1=EW1   + DELTA*EMAX1(FSTATE1(X1+2,X2+1,E-9,1))
       V2=EW2   + DELTA*EMAX1(FSTATE1(X1+1,X2+2,E-9,1))
       IF(E.LE.19) THEN
         V3=CBAR  + DELTA*EMAX1(FSTATE1(X1+1,X2+1,E-8,2))
        ELSE
         V3=CBAR  - 50000.0
       ENDIF
       V4=VHOME + DELTA*EMAX1(FSTATE1(X1+1,X2+1,E-9,1))
       RMAXV(K) = AMAX1(V1,V2,V3,V4)
       XR2 = PEI_SQRT(0.00001 + RMAXV(K) - V1)
       XR3 = PEI_SQRT(0.00001 + RMAXV(K) - V2)
       XR4 = PEI_SQRT(0.00001 + RMAXV(K) - V3)
       XR5 = PEI_SQRT(0.00001 + RMAXV(K) - V4)
       XR6 = (RMAXV(K) - V1)/100.0
       XR7 = (RMAXV(K) - V2)/100.0
!C      XR8 = (RMAXV(K) - V3)/100.0
       XR8 = (RMAXV(K) - V4)/100.0
       XGAMA = RGAMA(1) + RGAMA(2)*XR2  + RGAMA(3)*XR3 &
                        + RGAMA(4)*XR4  + RGAMA(5)*XR5 &
                        + RGAMA(6)*XR6  + RGAMA(7)*XR7 &
                        + RGAMA(8)*XR8
!C    *                  + RGAMA(8)*XR8  + RGAMA(9)*XR9
       EMAX(K) = RMAXV(K) + AMAX1(0.0,XGAMA)
!CC     IF(NP.EQ.NPARM+1) write(11,4100) T,k,emax(k),RMAXV(K)
!C4100 FORMAT(' T=',i2,' K=',i3,' EMAX=',f12.2,' RMAXV=',F12.2)
  559 CONTINUE
      DO 9052 KK=1,KKMAX
       K = TK(KK,T)
!C      write(17,9921) KK,K,EMAX(K),EMAXS(KK)
       EMAX(K) = EMAXS(KK)
 9052 CONTINUE
      ENDIF
 3000 CONTINUE
!**************************************************************
!*  CONSTRUCT THE LOG LIKELIHOOD FUNCTION CONTIBUTIONS FOR    *
!*  ALL PEOPLE AT PERIOD T                                    *
!**************************************************************
!C     IF(T.GT.20) GOTO 997
      DO 7161 K=1,KMAX
       ICOUNT0(K) = 0
 7161 CONTINUE
      DO 60 I=1,NPOP
         E=EDUC(I,T)
         X1=EXPER(I,T,1)
         X2=EXPER(I,T,2)
         LS=LSCHL(I,T)
         J=STATE(I,T)
         IF(E.GE.12) THEN
           CBAR = CBAR1 - CBAR2
          ELSE
           CBAR = CBAR1
         ENDIF
         IF(LS.EQ.0) CBAR = CBAR - CS
         E1 = DELTA*EMAX1(FSTATE1(X1+2,X2+1,E-9,1))
         E2 = DELTA*EMAX1(FSTATE1(X1+1,X2+2,E-9,1))
         IF(E.LE.19) E3 = CBAR + DELTA*EMAX1(FSTATE1(X1+1,X2+1,E-8,2))
         IF(E.GT.19) E3 = CBAR - 50000.0
         E4 = VHOME + DELTA*EMAX1(FSTATE1(X1+1,X2+1,E-9,1))
!**********************************************************
!*  PROBS OF STATES 3 and 4 NEED ONLY BE SIMULATED ONCE   *
!*  (THAT IS, FOR ONE PERSON) AT EACH VALUE OF THE STATE  *
!*  SPACE FOR WHICH 3 OR 4 OCCURS                         *
!**********************************************************
         IF((J.EQ.3).OR.(J.EQ.4)) THEN
          K=FSTATE(X1+1,X2+1,E-9,LS+1)
          ICOUNT0(K)=ICOUNT0(K)+1
          IF(ICOUNT0(K).EQ.1) THEN
           PROBI3(K)=0.0
           PROBI4(K)=0.0
          ENDIF
         ENDIF
!C****RETURN TO MAIN SECTION
         W1=EXP(BETA(1,1)+BETA(1,2)*E       &
             +BETA(1,3)*X1+BETA(1,4)*X1**2  &
             +BETA(1,5)*X2+BETA(1,6)*X2**2)
         W2=EXP(BETA(2,1)+BETA(2,2)*E       &
             +BETA(2,3)*X1+BETA(2,4)*X1**2  &
             +BETA(2,5)*X2+BETA(2,6)*X2**2)
       IF(STATE(I,T).EQ.1) THEN
         U1L = LOG(WAGE(I,T))-LOG(W1)
         U1J = U1L
         RNNL1 = U1L/SIGMA(1)
         U2L = A(2,1)*RNNL1
         CL = A(3,1)*RNNL1
         BL = A(4,1)*RNNL1
         V1 = WAGE(I,T) + E1
         PROBI=0.0
       ENDIF
       IF(STATE(I,T).EQ.2) THEN
         U2L = LOG(WAGE(I,T))-LOG(W2)
         U2J = U2L
         V2  = WAGE(I,T) + E2
         PROBI=0.0
       ENDIF
       IF(STATE(I,T).EQ.1) THEN
        DO 62 J=1,DRAW1
         RNNL2=RNNL(J,T,2)
         RNNL3=RNNL(J,T,3)
         RNNL4=RNNL(J,T,4)
         U2J = U2L + A(2,2)*RNNL2
         CJ = CL   + A(3,2)*RNNL2+A(3,3)*RNNL3
         BJ = BL   + A(4,2)*RNNL2+A(4,3)*RNNL3 + A(4,4)*RNNL4
         V2 = W2*EXP(U2J) + E2
         V3 = CJ          + E3
         V4 = BJ          + E4
         VMAX=AMAX1(V1,V2,V3,V4)
         A1 = EXP((V1-VMAX)/TAU)
         A2 = EXP((V2-VMAX)/TAU)
         A3 = EXP((V3-VMAX)/TAU)
         A4 = EXP((V4-VMAX)/TAU)
         SUMV = A1+A2+A3+A4
         PROB = A1/SUMV
         PROBI = PROBI + PROB
!C        IF(NP.EQ.NPARM+1)
!C    *   write(11,1061) I,T,STATE(I,T),J,PROB,V1,V2,V3,V4
!C1061    format(' I=',I3,' t=',I2,' STATE=',I2,' DRAW=',I3,
!C    *      ' PROB=',F16.12,/,
!C    *      '      V1=',F13.3,' V2=',F13.3,' V3=',F13.3,
!C    *      ' V4=',F13.4)
   62   CONTINUE
        PROBI = PROBI/DRAW1
       ENDIF
       IF(STATE(I,T).EQ.2) THEN
        DO 63 J=1,DRAW1
         RNNL1=RNNL(J,T,1)
         RNNL3=RNNL(J,T,3)
         RNNL4=RNNL(J,T,4)
         U1J = A(1,1)*RNNL1
         RNNJ2 = (U2J-A(2,1)*RNNL1)/A(2,2)
         CJ = A(3,1)*RNNL1+A(3,2)*RNNJ2+A(3,3)*RNNL3
         BJ = A(4,1)*RNNL1+A(4,2)*RNNJ2+A(4,3)*RNNL3+A(4,4)*RNNL4
         V1 = W1*EXP(U1J) + E1
         V3 = CJ          + E3
         V4 = BJ          + E4
         VMAX=AMAX1(V1,V2,V3,V4)
         A1 = EXP((V1-VMAX)/TAU)
         A2 = EXP((V2-VMAX)/TAU)
         A3 = EXP((V3-VMAX)/TAU)
         A4 = EXP((V4-VMAX)/TAU)
         SUMV = A1+A2+A3+A4
         PROB = A2/SUMV
!C        IF(NP.EQ.NPARM+1)
!C    *     write(11,1061) I,T,STATE(I,T),J,PROB,V1,V2,V3,V4
         DENW = 0.3989423*EXP(-0.5*RNNJ2**2)/A(2,2)
         PROB = PROB*DENW
         PROBI = PROBI + PROB
!C        write(11,1061) I,T,STATE(I,T),J,PROB,V1,V2,V3,V4
   63   CONTINUE
        PROBI = PROBI/DRAW1
       ENDIF
       IF((STATE(I,T).EQ.3).OR.(STATE(I,T).EQ.4)) THEN
       IF(ICOUNT0(K).EQ.1) THEN
        DO 64 J=1,DRAW1
         RNNL1=RNNL(J,T,1)
         RNNL2=RNNL(J,T,2)
         RNNL3=RNNL(J,T,3)
         RNNL4=RNNL(J,T,4)
         U1J = A(1,1)*RNNL1
         U2J = A(2,1)*RNNL1+A(2,2)*RNNL2
         CJ  = A(3,1)*RNNL1+A(3,2)*RNNL2+A(3,3)*RNNL3
         BJ  = A(4,1)*RNNL1+A(4,2)*RNNL2+A(4,3)*RNNL3+A(4,4)*RNNL4
         V1 = W1*EXP(U1J) + E1
         V2 = W2*EXP(U2J) + E2
         V3 = CJ          + E3
         V4 = BJ          + E4
         VMAX=AMAX1(V1,V2,V3,V4)
         A1 = EXP((V1-VMAX)/TAU)
         A2 = EXP((V2-VMAX)/TAU)
         A3 = EXP((V3-VMAX)/TAU)
         A4 = EXP((V4-VMAX)/TAU)
         SUMV = A1+A2+A3+A4
         PROB3 = A3/SUMV
         PROB4 = A4/SUMV
         PROBI3(K) = PROBI3(K) + PROB3
         PROBI4(K) = PROBI4(K) + PROB4
!C        IF(NP.EQ.NPARM+1) THEN
!C        IF(STATE(I,T).EQ.3) THEN
!C           write(11,1061) I,T,STATE(I,T),J,PROB3,V1,V2,V3,V4
!C         ELSE
!C           write(11,1061) I,T,STATE(I,T),J,PROB4,V1,V2,V3,V4
!C        ENDIF
!C        ENDIF
   64   CONTINUE
        PROBI3(K) = PROBI3(K)/DRAW1
        PROBI4(K) = PROBI4(K)/DRAW1
       ENDIF
       ENDIF
      IF(STATE(I,T).eq.1) THEN
        DENW = 0.3989423*EXP(-0.5*(U1L/SIGMA(1))**2)/SIGMA(1)
        PROBI = PROBI*DENW
      ENDIF
      IF((STATE(I,T).eq.1).OR.(STATE(I,T).EQ.2)) THEN
      IF(PROBI.GT.PMIN) THEN
!C       WRITE(11,6656) I,T,STATE(I,T),PROBI,DENW
        RLLFI(I,IP) = RLLFI(I,IP) + LOG(PROBI)
        RLOGLF(IP) = RLOGLF(IP) + LOG(PROBI)
       ELSE
        IF(NP.EQ.NPARM+1) WRITE(11,6655) I,T,STATE(I,T),PROBI
 6655   FORMAT(' *** WARNING: PERSON',I5,' T=',I2,' J=',I2,' PROB=',F16.14)
        RLLFI(I,IP) = RLLFI(I,IP) + LOG(PMIN)
        RLOGLF(IP) = RLOGLF(IP) + LOG(PMIN)
      ENDIF
      ENDIF
      IF(STATE(I,T).eq.3) THEN
      IF(PROBI3(K).GT.PMIN) THEN
!C       WRITE(11,6656) I,T,STATE(I,T),PROBI,DENW
        RLLFI(I,IP) = RLLFI(I,IP) + LOG(PROBI3(K))
        RLOGLF(IP) = RLOGLF(IP) + LOG(PROBI3(K))
       ELSE
        IF(NP.EQ.NPARM+1) WRITE(11,6655) I,T,STATE(I,T),PROBI3(K)
        RLLFI(I,IP) = RLLFI(I,IP) + LOG(PMIN)
        RLOGLF(IP) = RLOGLF(IP) + LOG(PMIN)
      ENDIF
      ENDIF
      IF(STATE(I,T).eq.4) THEN
      IF(PROBI4(K).GT.PMIN) THEN
!C       WRITE(11,6656) I,T,STATE(I,T),PROBI,DENW
        RLLFI(I,IP) = RLLFI(I,IP) + LOG(PROBI4(K))
        RLOGLF(IP) = RLOGLF(IP) + LOG(PROBI4(K))
       ELSE
        IF(NP.EQ.NPARM+1) WRITE(11,6655) I,T,STATE(I,T),PROBI4(K)
        RLLFI(I,IP) = RLLFI(I,IP) + LOG(PMIN)
        RLOGLF(IP) = RLOGLF(IP) + LOG(PMIN)
      ENDIF
      ENDIF
!C     IF(NP.EQ.NPARM+1) WRITE(11,4010) I,T,STATE(I,T),PROBI,
!C    *     RLOGLF(IP)
!C***END OF LOOP OVER PEOPLE
   60 CONTINUE
!C 997 CONTINUE
!C***END OF LOOP OVER TIME PERIODS
   30 CONTINUE
!C***END OF LOOP OVER PARAMETERS (NP) FOR CALCULATION OF NUMERICAL
!C***DERIVS
      GOTO 6000
!*************************************************************
!* JUMP TO 6003 AFTER DERIVS AND LIKELIHOOD ARE CONSTRUCTED  *
!*************************************************************
 6003 CONTINUE
      IF((ITER.EQ.1).AND.(ISTEP.EQ.0)) SRLOGLF = RLOGLF(NITER+1)
      IF(ISTEP.EQ.0) RLOGLF0 = RLOGLF(NITER+1)
!CCC   WRITE(11,1100) ITER,ISTEP,RLOGLF(NITER+1)
!C1100 FORMAT(/,' ITER=',I2,' STEP=',I2,
!CCC  *       ' SIMULATED LOG LIKELIHOOD = ',F20.6,/)
!***********************************************************
!*  IF STEP IS 0 THEN CONSTRUCT THE NUMERICAL DERIVATIVES  *
!*  AND THE APPROXIMATE HESSIAN AND ITS INVERSE            *
!***********************************************************
      IF(ISTEP.EQ.0) THEN
      IP=0
      DO 70 JJ = 1,NPARM
       IF(IPARM(JJ).eq.0) GOTO 70
       IP=IP+1
       FOC(IP) = (RLOGLF(IP) - RLOGLF(NITER+1))/BUMP
!CCC    FOC(IP) = (RLOGLF(IP) - RLOGLF(NITER+1))/(BUMP*CPARM(JJ)+BUMP)
!CCC    WRITE(11,1200) IP,FOC(IP)
!C1200  FORMAT(' IP = ',I2,' FOC = ',f20.5)
   70 CONTINUE
      IP1=0
      DO 71 JJ1 = 1,NPARM
        IF(IPARM(JJ1).eq.0) GOTO 71
        IP1=IP1+1
!C       BUMP1=BUMP*CPARM(JJ1)+BUMP
        BUMP1=BUMP
        IP2=0
      DO 72 JJ2 = 1,NPARM
        IF(IPARM(JJ2).eq.0) GOTO 72
        IP2=IP2+1
!C       BUMP2=BUMP*CPARM(JJ2)+BUMP
        BUMP2=BUMP
        SPDM(IP1,IP2) = 0.0
      DO 73 I=1,NPOP
        FOC1 = (RLLFI(I,IP1)-RLLFI(I,NITER+1))/BUMP1
        FOC2 = (RLLFI(I,IP2)-RLLFI(I,NITER+1))/BUMP2
        SPDM(IP1,IP2) = SPDM(IP1,IP2) + FOC1*FOC2
   73 CONTINUE
   72 CONTINUE
   71 CONTINUE
      DO 1071 IP1=1,NITER
      DO 1072 IP2=1,NITER
        IF(IP1.EQ.IP2) SPDMR(IP1,IP2) = SPDM(IP1,IP2)*(1.0 + ALPHA)
        IF(IP1.NE.IP2) SPDMR(IP1,IP2) = SPDM(IP1,IP2)
 1072 CONTINUE
 1071 CONTINUE
!CCC   IF(ITER.EQ.1) THEN
!CCC    WRITE(11,4002)
!C4002  FORMAT(/,' OUTER PRODUCT APPROXIMATION OF HESSIAN:')
!CCC   DO 74 IP1 = 1,NITER
!CCC    WRITE(11,4003) (SPDM(IP1,IP2),IP2=1,NITER)
!C4003  FORMAT(6F12.2)
!CCC74 CONTINUE
!CCC   ENDIF
!C***INVERT PDM
      CALL LINDS(NITER,SPDM,27,SPDMI,27)
      CALL LINDS(NITER,SPDMR,27,SPDMRI,27)
      DO 75 IP1 = 2,NITER
      DO 76 IP2 = 1,IP1-1
        SPDMI(IP1,IP2) = SPDMI(IP2,IP1)
        SPDMRI(IP1,IP2) = SPDMRI(IP2,IP1)
   76 CONTINUE
   75 CONTINUE
!C      WRITE(11,4004)
!C4004  FORMAT(/,' COVARIANCE MATRIX :')
!C     DO 77 IP1 = 1,NITER
!C      WRITE(11,4003) (SPDMI(IP1,IP2),IP2=1,NITER)
!C  77 CONTINUE
      DO 80 IP1 = 1,NITER
        DELTAP(IP1) = 0.0
      DO 81 IP2 = 1,NITER
        DELTAP(IP1) = DELTAP(IP1) + SPDMRI(IP1,IP2)*FOC(IP2)
   81 CONTINUE
   80 CONTINUE
!CCC   DO 380 IP=1,NITER
!CCC     WRITE(11,4380) IP,DELTAP(IP)
!C4380   FORMAT(' DELTAP(',I2,') = ',f8.3)
!CC380 CONTINUE
      ENDIF
!*******************************************************
!*  IF THIS STEP IMPROVED THE LOGLIKELIHOOD THEN SAVE  *
!*  THE NEW PARAMETER VECTOR IN SPARM AND SAVE THE     *
!*  NEW LOGLIKELIHOOD FUNCTION IN SRLOGLF. OTHERWISE,  *
!*  RESTORE THE OLD PARMETER VECTOR                    *
!*******************************************************
      IF(ISTEP.GT.0) THEN
      WRITE(11,4020) ITER,ISTEP,SRLOGLF,RLOGLF(NITER+1)
 4020 FORMAT(/,' ITER ',I2,' STEP ',I2,/, &
        ' CHECK IF STEP IMPROVED LIKELIHOOD:',/, &
        ' OLD LOGLF=',F20.12,' NEW LOGLF=',F20.12)
      IF(RLOGLF(NITER+1).GT.SRLOGLF) THEN
         SRLOGLF = RLOGLF(NITER+1)
         DO 83 NP = 1,NPARM
           SPARM(NP) = CPARM(NP)
   83    CONTINUE
         IGOOD = IGOOD + 1
         WRITE(11,4383) ISTEP
 4383    FORMAT(' STEP ',I2,' WAS GOOD. UPDATING PARM VECTOR')
!C4383    FORMAT(/,' STEP ',I2,' WAS GOOD. UPDATING PARM',
!C    *       ' VECTOR TO:')
!C        WRITE(11,4001) (CPARM(NP),NP=1,NPARM)
       ELSE
         DO 84 NP = 1,NPARM
           CPARM(NP) = SPARM(NP)
   84    CONTINUE
         SSIZE = 0.5*SSIZE
         IBAD = IBAD + 1
         WRITE(11,4384) ISTEP
 4384    FORMAT(/,' STEP ',I2,' WAS BAD')
         IF(IBAD.EQ.MAXSTP) THEN
           WRITE(11,6666) ITER
 6666      FORMAT(/,' ITER=',I2,' COULD NOT FIND INCREASING',' STEP ',/)
           GOTO 9999
         ENDIF
      ENDIF
!C***END THE ITERATION IF FOUR GOOD STEPS HAVE BEEN ACHEIVED
!C***OR IF THE MAX NUMBER OF STEPS HAS BEEN REACHED
      IF(IGOOD.EQ.4) GOTO 5002
      IF((IGOOD.GE.1).AND.(IBAD.GT.0)) GOTO 5002
      IF(ISTEP.EQ.MAXSTP) GOTO 5002
      ELSE
!C***WHEN STEP IS 0 SAVE THE PARM VECTOR
       DO 85 NP=1,Nparm
        SPARM(NP) = CPARM(NP)
   85  CONTINUE
!C      WRITE(11,4385) ITER,ISTEP
!C4385  FORMAT(/,'ITER ',I2,' STEP ',I2,': SAVING PARM',
!C    *      ' VECTOR AS:')
!C      WRITE(11,4001) (CPARM(NP),NP=1,NPARM)
      ENDIF
!*********************************
!*  UPDATE THE PARAMETER VECTOR  *
!*********************************
      IP = 0
      DO 86 NP = 1,NPARM
        IF(IPARM(NP).EQ.0) GOTO 86
        IP = IP + 1
        IF(TRANS.EQ.'YES') THEN
         IF(NP.EQ.17) THEN
          CT = (1.0/CPARM(NP)) - 1.0
          CT = CT + SSIZE*DELTAP(IP)
          CPARM(NP) = 1.0/(1.0+CT)
          GOTO 86
         ENDIF
         IF(NP.LE.17) THEN
          CPARM(NP) = CPARM(NP) + SSIZE*DELTAP(IP)
         ELSE
          CPARM(NP) = CPARM(NP) + 1.00*SSIZE*DELTAP(IP)
         ENDIF
        ELSE
         IF(NP.LE.17) THEN
          CPARM(NP) = CPARM(NP) + SSIZE*DELTAP(IP)
         ELSE
          CPARM(NP) = CPARM(NP) + 1.00*SSIZE*DELTAP(IP)
         ENDIF
        ENDIF
   86 CONTINUE
      WRITE(11,4386) ISTEP+1
 4386 FORMAT(/,' STEP ',I2,' - TRYING PARM VECTOR:')
      WRITE(11,4001) (CPARM(NP),NP=1,NPARM)
 4001 FORMAT(' CPARM=',6F9.3)
!*******************************
!*    GO BACK FOR NEXT STEP    *
!*******************************
      GOTO 5001
 5002 CONTINUE
!****************************************************************
!*  CONSTRUCT TRANSFORMED PARAMETER VECTOR AT END OF ITERATION  *
!****************************************************************
      DO 7001 JJ = 1,NPARM
       IF(TRANS.EQ.'YES') THEN
        IF(JJ.EQ.17) THEN
          CPARMT(JJ) = (1.0/CPARM(JJ)) - 1.0
          GOTO 7001
        ENDIF
        CPARMT(JJ) = CPARM(JJ)
       ELSE
        CPARMT(JJ) = CPARM(JJ)
       ENDIF
 7001 CONTINUE
!*********************************************************
!*  CONSTRUCT CHI SQUARED TEST FOR WHETHER PARAMS EQUAL  *
!*  THE STARTING VALUES (CPARMS VS. CPARMT).             *
!*********************************************************
!C     IP1 = 0
!C     DO 8000 NP1 = 1,NPARM
!C       IF(IPARM(NP1).eq.0) GOTO 8000
!C       IP1 = IP1 + 1
!C       C1(IP1) = 0.0
!C       IP2 = 0
!C       DO 8001 NP2 = 1,NPARM
!C         IF(IPARM(NP2).eq.0) GOTO 8001
!C         IP2 = IP2 + 1
!C         DC = CPARMS(NP2) - CPARMT(NP2)
!C         C1(IP1) = C1(IP1) + SPDM(IP1,IP2)*DC
!C8001   CONTINUE
!C8000 CONTINUE
!C     CHISQR = 0.0
!C     IP1 = 0
!C     DO 8002 NP1 = 1,NPARM
!C       IF(IPARM(NP1).eq.0) GOTO 8002
!C       IP1 = IP1 + 1
!C       DC = CPARMS(NP1) - CPARMT(NP1)
!C       CHISQR = CHISQR + DC*C1(IP1)
!C8002 CONTINUE
!*******************************************
!*  WRITE OUT RESULTS AT END OF ITERATION  *
!*******************************************
      IP = 0
      DO 90 NP = 1,NPARM
       IF(IPARM(NP).EQ.0) GOTO 90
       IP = IP + 1
       STDERR(IP) = SQRT(SPDMI(IP,IP))
       TSTAT(IP) = (CPARMT(NP)-CPARMS(NP))/STDERR(IP)
   90 CONTINUE
      WRITE(11,1111) ITER
 1111 FORMAT(//,' RESULTS AT END OF ITERATION ',I2,//, &
        ' PARAMETER   ESTIMATE      TRANSFORM   ', &
        ' TRUE VALUE    STD.ERR.      T-STAT')
      IP = 0
      DO 91 NP = 1,NPARM
       IF(IPARM(NP).EQ.0) GOTO 91
       IP = IP + 1
       WRITE(11,1091) NAME(NP),CPARM(NP),CPARMT(NP),CPARMS(NP),STDERR(IP),TSTAT(IP)
 1091  FORMAT(1x,A8,2f14.7,f10.4,2f14.7)
   91 CONTINUE
!CCC   WRITE(11,8005) CHISQR
!C8005 FORMAT(/,' CHISQR STAT FOR EQUALITY OF PARAMETER ',
!CCC  *         'VECTOR WITH TRUE PARAMETER VECTOR:',/,3x,F15.4,/)
!C***WRITE OUT THE CORRELATION MATRIX IMPLIED BY THE CPARM VECTOR
      L = 15
      DO 7010 J=1,4
      DO 7011 K=1,J
       L = L + 1
       A(J,K) = CPARM(L)
 7011 CONTINUE
 7010 CONTINUE
      DO 7012 J=1,4
      DO 7013 K=1,J
       RHO(J,K) = 0.0
      DO 7014 L=1,K
       RHO(J,K) = RHO(J,K) + A(J,L)*A(K,L)
 7014 CONTINUE
 7013 CONTINUE
 7012 CONTINUE
      SIGMA(1) = A(1,1)
      SIGMA(2) = SQRT(A(2,1)**2 + A(2,2)**2)
      SIGMA(3) = SQRT(A(3,1)**2 + A(3,2)**2 + A(3,3)**2)
      SIGMA(4) = SQRT(A(4,1)**2 + A(4,2)**2 + A(4,3)**2+A(4,4)**2)
      DO 7015 J=1,4
      DO 7016 K=1,J
       RHO(J,K) = RHO(J,K)/(SIGMA(J)*SIGMA(K))
 7016 CONTINUE
 7015 CONTINUE
!CCC   DO 1120 J=1,4
!CCC   WRITE(11,1121) (A(J,K),K=1,J)
!C1121 FORMAT(' A=',4(F15.10))
!C1120 CONTINUE
!CCC   DO 7017 J=1,4
!CCC   WRITE(11,7008) (RHO(J,K),K=1,J)
!C7017 CONTINUE
!CCC   WRITE(11,7009) (SIGMA(J),J=1,4)
!************************************************************
!*  IF LOGLF IMPROVEMENT WAS SMALL ENOUGH THEN JUMP OUT OF  *
!*  ITERATION LOOP                                          *
!************************************************************
       CLI = -(SRLOGLF - RLOGLF0)/RLOGLF0
       WRITE(11,5003) ITER,SRLOGLF,RLOGLF0,CLI
 5003  FORMAT(' ITER=',I2,' LF=',f12.6,' OLD LF=',F12.6,' % IMPROVE=',F10.8,/)
       IF(CLI.LT.0.000010) GOTO 9999
!C***END OF ITERATION LOOP
 5000 CONTINUE
 9999 CONTINUE
!**************************************************
!* WRITE OUT THE FINAL VALUES TO A NEW INPUT FILE *
!**************************************************
      L = 0
      DO 7109 J=1,2
      DO 7110 K=1,6
       L = L + 1
       BETA(J,K) = CPARM(L)
 7110 CONTINUE
 7109 CONTINUE
       L = L + 1
       CBAR1 = CPARM(L)
       L = L + 1
       CBAR2 = CPARM(L)
       L = L + 1
       CS = CPARM(L)
       L = L + 1
       VHOME = CPARM(L)
       L = L + 1
       DELTA = CPARM(L)
      DO 7111 J=1,4
      DO 7112 K=1,J
       L = L + 1
       A(J,K) = CPARM(L)
 7112 CONTINUE
 7111 CONTINUE
      DO 8012 J=1,4
      DO 8013 K=1,J
       RHO(J,K) = 0.0
      DO 8014 L=1,K
       RHO(J,K) = RHO(J,K) + A(J,L)*A(K,L)
 8014 CONTINUE
 8013 CONTINUE
 8012 CONTINUE
      SIGMA(1) = A(1,1)
      SIGMA(2) = SQRT(A(2,1)**2 + A(2,2)**2)
      SIGMA(3) = SQRT(A(3,1)**2 + A(3,2)**2 + A(3,3)**2)
      SIGMA(4) = SQRT(A(4,1)**2 + A(4,2)**2 + A(4,3)**2+A(4,4)**2)
      DO 8015 J=1,4
      DO 8016 K=1,J
       RHO(J,K) = RHO(J,K)/(SIGMA(J)*SIGMA(K))
 8016 CONTINUE
 8015 CONTINUE
      DO 7113 J=1,2
      WRITE(12,1501) (BETA(J,K),K=1,6)
 7113 CONTINUE
      WRITE(12,1502) CBAR1,CBAR2,CS,VHOME,DELTA
      DO 7114 J=1,4
      WRITE(12,1503) (RHO(J,K),K=1,J)
 7114 CONTINUE
      WRITE(12,1503) (SIGMA(J),J=1,4)
      DO 7115 J=1,2
       WRITE(12,1601) (NAME(K),K=(J-1)*6+1,J*6)
 7115 CONTINUE
      WRITE(12,1602) (NAME(K),K=2*6+1,2*6+5)
      DO 7116 J=1,4
       WRITE(12,1603) (NAME(K),K=2*6+5+J*(J-1)/2+1,2*6+5+J*(J+1)/2)
 7116 CONTINUE
      WRITE(12,1603) (NAME(K),K=2*6+5+4*(4+1)/2+1,2*6+5+4*(4+1)/2+4)
      DO 7203 J=1,2
       WRITE(12,1604) (IPARM(K),K=(J-1)*6+1,J*6)
 7203 CONTINUE
      WRITE(12,1605) (IPARM(K),K=2*6+1,2*6+5)
      DO 7204 J=1,4
      WRITE(12,1606)(IPARM(K),K=2*6+5+J*(J-1)/2+1,2*6+5+J*(J+1)/2)
 7204 CONTINUE
!********************************************************
!*  WRITE OUT ENDING PARAM VECTOR AND STD. ERROR VECTOR *
!*  FOR THIS REPLICATION                                *
!********************************************************
      DO 5005 J=1,5*(IRUN-1)
       READ(14,*)
       READ(15,*)
 5005 CONTINUE
      WRITE(14,1414) (CPARMT(J),J=1,16),(CPARMT(J),J=18,27),CLI
 1414 FORMAT(6F13.9)
      WRITE(15,1414) (STDERR(J),J=1,26)
!*******************************************************************
!* WRITE OUT REPLICATION NUMBER AND THE SEEDS FOR NEXT REPLICATION *
!*******************************************************************
      REWIND(UNIT=13)
      WRITE(13,1313) IRUN+1,ISEED,ISEED1,ISEED2
      STOP
      END
!**********************************
!*  NORMAL DISTRIBUTION FUNCTION  *
!**********************************
!C      FUNCTION PR(X)
!C      AX=ABS(X)
!C       IF AX .LE. 6 THEN
!C        T=1./(1.+0.2316419*AX)
!C         D=0.3989423*EXP(-X*X/2.0)
!C         PR=1.0-D*T*((((1.330274429*T-1.821255978)*T+
!C     *         1.781477937)*T-0.356563782)*T+0.319381530)
!C       ELSE
!C         PR=0.999999999009878099
!C      ENDIF
!C      IF(X.LT.0.0) PR=1.0-PR
!C      RETURN
!C      END
!*****************************
!*  NORMAL DENSITY FUNCTION  *
!*****************************
!C     FUNCTION DEN(X)
!C     AX=ABS(X)
!C     IF(AX.LE.6.0) THEN
!C        DEN=0.3989423*EXP(-X**2/2.0)
!C      ELSE
!C        DEN=0.00000000607588314830906838
!C     ENDIF
!C     RETURN
!C     END
