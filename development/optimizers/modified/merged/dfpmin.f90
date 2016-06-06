MODULE bfgs_function

	!/*	external modules	*/
	
	USE shared_constants

	IMPLICIT NONE

	!/*	setup	*/

	PRIVATE

	PUBLIC:: dfpmin

CONTAINS
!******************************************************************************
!******************************************************************************
SUBROUTINE dfpmin(func, dfunc, p, gtol, iter, fret)

    !/* external objects        */

	REAL(our_dble), INTENT(OUT) 	:: fret

	REAL(our_dble), INTENT(INOUT) 	:: p(:)
	REAL(our_dble), INTENT(IN) 		:: gtol

	INTEGER(our_int), INTENT(OUT) 	:: iter

	INTERFACE

		FUNCTION func(p)
	
			USE shared_constants

			IMPLICIT NONE
	
			REAL(our_dble), INTENT(IN) 	:: p(:)
			REAL(our_dble) 				:: func
	
		END FUNCTION  

		FUNCTION dfunc(p)
	
			USE shared_constants
	
			IMPLICIT NONE
	
			REAL(our_dble), INTENT(IN) 	:: p(:)
			REAL(our_dble)				:: dfunc(SIZE(p))
	
		END FUNCTION  

	END INTERFACE

    !/* internal objects        */

!	INTEGER(our_int) 				:: ITMAX = 200
	INTEGER(our_int) 				:: ITMAX = 200000000

	INTEGER(our_int) 				:: its
	INTEGER(our_int) 				:: i

	REAL(our_dble)					:: hessin(SIZE(p), SIZE(p))
	REAL(our_dble) 					:: STPMX = 100.0_our_dble
	REAL(our_dble), PARAMETER 		:: EPS = epsilon(p)
	REAL(our_dble) 					:: TOLX = 4.0_our_dble * EPS
	REAL(our_dble) 					:: pnew(SIZE(p))
	REAL(our_dble) 					:: hdg(SIZE(p))
	REAL(our_dble) 					:: xi(SIZE(p))
	REAL(our_dble) 					:: dg(SIZE(p))
	REAL(our_dble) 					:: g(SIZE(p))
	REAL(our_dble) 					:: stpmax
	REAL(our_dble) 					:: sumdg
	REAL(our_dble) 					:: sumxi
	REAL(our_dble) 					:: den
	REAL(our_dble) 					:: fac
	REAL(our_dble) 					:: fad
	REAL(our_dble) 					:: fae
	REAL(our_dble) 					:: fp

	LOGICAL 						:: check

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

	fp = func(p)
	g = dfunc(p)
	
	! Initialize Hessian
	hessin = zero_dble
	DO i = 1, SIZE(p)
		hessin(i, i) = one_dble
	END DO
	
	xi = -g
	
	stpmax = STPMX * MAX(SQRT(DOT_PRODUCT(p, p)), REAL(size(p), our_dble))
	
	DO its =1, ITMAX

		iter = its

		CALL lnsrch(p, fp, g, xi, pnew, fret, stpmax, check, func)

		fp = fret		
		xi = pnew - p
		p = pnew

		if (MAXVAL(ABS(xi) / max(ABS(p), one_dble)) < TOLX) RETURN

		dg = g
		g = dfunc(p)
		den = max(fret, one_dble)

		if (MAXVAL(ABS(g) * max(ABS(p), one_dble) / den) < gtol) RETURN

		dg =  g - dg
		hdg = matmul(hessin, dg)
		fac = DOT_PRODUCT(dg, xi)
		fae = DOT_PRODUCT(dg, hdg)
		sumdg = DOT_PRODUCT(dg, dg)
		sumxi = DOT_PRODUCT(xi, xi)

		IF (fac ** 2 > EPS * sumdg * sumxi) THEN

			fac = one_dble / fac
			fad = one_dble / fae
			dg = fac * xi - fad * hdg

			hessin = hessin + fac * outerprod(xi, xi) - fad * outerprod(hdg,hdg) + fae * outerprod(dg,dg)
		
		END IF

		xi = -matmul(hessin,g)
	
	END DO

	CALL nrerror('dfpmin: too many iterations')
	
END SUBROUTINE 
!******************************************************************************
!******************************************************************************
SUBROUTINE lnsrch(xold, fold, g, p, x, f, stpmax, check, func)

    !/* external objects        */

	REAL(our_dble), INTENT(OUT) 	:: x(:)
	REAL(our_dble), INTENT(OUT) 	:: f
	
	REAL(our_dble), INTENT(IN) 		:: xold(:)
	REAL(our_dble), INTENT(IN) 		:: stpmax
	REAL(our_dble), INTENT(IN) 		:: fold
	REAL(our_dble), INTENT(IN) 		:: g(:)

	REAL(our_dble), INTENT(INOUT) 	:: p(:)
	
	LOGICAL, INTENT(OUT) 			:: check

	INTERFACE

		FUNCTION func(x)

		USE shared_constants

		IMPLICIT NONE

		REAL(our_dble) 				:: func

		REAL(our_dble), INTENT(IN) 	:: x(:)

		END FUNCTION 

	END INTERFACE

    !/* internal objects        */

	REAL(our_dble), PARAMETER 		:: ALF=1.0e-4_our_dble
	REAL(our_dble), PARAMETER 		:: TOLX=epsilon(x)

	INTEGER(our_int) 				:: ndum

	REAL(our_dble) 					:: alamin
	REAL(our_dble) 					:: tmplam
	REAL(our_dble) 					:: slope
	REAL(our_dble) 					:: alam2
	REAL(our_dble) 					:: fold2
	REAL(our_dble) 					:: alam
	REAL(our_dble) 					:: disc
	REAL(our_dble) 					:: pabs
	REAL(our_dble) 					:: rhs1
	REAL(our_dble) 					:: rhs2
	REAL(our_dble) 					:: f2
	REAL(our_dble) 					:: a
	REAL(our_dble) 					:: b

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

	ndum = size(g)

	check = .FALSE.

	pabs = SQRT(DOT_PRODUCT(p, p))

	IF (pabs > stpmax) p = p * stpmax / pabs

	slope = DOT_PRODUCT(g, p)
	
	alamin = TOLX / MAXVAL(ABS(p)/MAX(ABS(xold), one_dble))

	alam = 1.0

	DO
		x(:) = xold(:) + alam * p(:)
		
		f = func(x)
		
		IF (alam < alamin) THEN

			x(:)=xold(:)
			check=.true.
			RETURN

		ELSEIF (f <= fold + ALF * alam * slope) THEN
		
			RETURN
		
		ELSE
		
			if (alam == one_dble) then
		
				tmplam = -slope / (two_dble * (f -fold - slope))
		
			else
		
				rhs1 = f - fold - alam * slope
				rhs2 = f2 - fold2 - alam2 * slope
				a = (rhs1 / alam ** 2 - rhs2 / alam2 ** 2) / (alam - alam2)
				b = (-alam2 * rhs1 / alam ** 2 + alam * rhs2 / alam2 ** 2) / (alam - alam2)
		
				IF (a == zero_dble) THEN

					tmplam = -slope / (two_dble * b)

				ELSE

					disc = b * b - three_dble * a * slope
					
					IF (disc < zero_dble) THEN
						disc = 0.001
					END IF

					tmplam = (-b + SQRT(disc)) / (three_dble * a)

				END IF
				
				IF (tmplam > half_dble * alam) tmplam = half_dble * alam
			
			END IF
		
		END IF
		
		alam2 = alam
		f2 = f
		fold2 = fold
		alam = MAX(tmplam, 0.1_our_dble * alam)
	
	END DO

END SUBROUTINE 
!******************************************************************************
!******************************************************************************
	SUBROUTINE nrerror(string)
	CHARACTER(LEN=*), INTENT(IN) :: string
	write (*,*) 'nrerror: ',string
	STOP 'program terminated by nrerror'
	END SUBROUTINE nrerror
!******************************************************************************
!******************************************************************************
	FUNCTION outerprod(a,b)
	REAL(our_dble), DIMENSION(:), INTENT(IN) :: a,b
	REAL(our_dble), DIMENSION(size(a),size(b)) :: outerprod
	outerprod = spread(a,dim=2,ncopies=size(b)) * &
		spread(b,dim=1,ncopies=size(a))
	END FUNCTION outerprod
!******************************************************************************
!******************************************************************************
END MODULE