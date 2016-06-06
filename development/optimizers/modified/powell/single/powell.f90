MODULE powell_function


CONTAINS
SUBROUTINE powell(p,xi,ftol,iter,fret)
USE nrtype;  
USE nr, ONLY :linmin
USE criterion_function
IMPLICIT NONE
REAL(SP), DIMENSION(:), INTENT(INOUT) :: p
REAL(SP), DIMENSION(:,:), INTENT(INOUT) :: xi
INTEGER(I4B), INTENT(OUT) :: iter
REAL(SP), INTENT(IN) :: ftol
REAL(SP), INTENT(OUT) :: fret

INTEGER(I4B), PARAMETER :: ITMAX=200
REAL(SP), PARAMETER :: TINY=1.0e-25_sp
!Minimization of a function func of N variables. (func is not an argument, it is a fixed
!function name.) Input consists of an initial starting point p, a vector of length N;an
!initial N × N matrix xi whose columns contain the initial set of directions (usually the N
!unit vectors);and ftol, the fractional tolerance in the function value such that failure to
!decrease by more than this amount on one iteration signals doneness. On output, p is set
!to the best point found, xi is the then-current direction set, fret is the returned function
!value at p, and iter is the number of iterations taken. The routine linmin is used.
!Parameters: Maximum allowed iterations, and a small number.
INTEGER(I4B) :: i,ibig,n
REAL(SP) :: del,fp,fptt,t
REAL(SP), DIMENSION(size(p)) :: pt,ptt,xit
n=size(p)
fret=func(p)
pt(:)=p(:) !Save the initial point.
iter=0
do
iter=iter+1
fp=fret
ibig=0
del=0.0 !Will be the biggest function decrease.
do i=1,n !Loop over all directions in the set.
xit(:)=xi(:,i) !Copy the direction,
fptt=fret
call linmin(p,xit,fret) !minimize along it,
if (fptt-fret > del) then !and record it if it is the largest decrease so
del=fptt-fret !far.
ibig=i
end if
end do
if (2.0_sp*(fp-fret) <= ftol*(abs(fp)+abs(fret))+TINY) RETURN
!Termination criterion.
if (iter == ITMAX) call &
nrerror('powell exceeding maximum iterations')
ptt(:)=2.0_sp*p(:)-pt(:) !Construct the extrapolated point and the average
!direction moved. Save the old starting
!point.
xit(:)=p(:)-pt(:)
pt(:)=p(:)
fptt=func(ptt) !Function value at extrapolated point.
if (fptt >= fp) cycle !One reason not to use new direction.
t=2.0_sp*(fp-2.0_sp*fret+fptt)*(fp-fret-del)**2-del*(fp-fptt)**2
if (t >= 0.0) cycle!Other reason not to use new direction.
call linmin(p,xit,fret) !Move to minimum of the new direction,
xi(:,ibig)=xi(:,n) !and save the new direction.
xi(:,n)=xit(:)
end do !Back for another iteration.
END SUBROUTINE powell
!*******************************************************************************
!*******************************************************************************
    SUBROUTINE nrerror(string)
    CHARACTER(LEN=*), INTENT(IN) :: string
    write (*,*) 'nrerror: ',string
    STOP 'program terminated by nrerror'
    END SUBROUTINE nrerror
!*******************************************************************************
!*******************************************************************************

    SUBROUTINE linmin(p,xi,fret)
    USE nrtype; USE nrutil, ONLY : assert_eq
    USE nr, ONLY : mnbrak,brent
    USE f1dim_mod
    IMPLICIT NONE
    REAL(SP), INTENT(OUT) :: fret
    REAL(SP), DIMENSION(:), TARGET, INTENT(INOUT) :: p,xi
    REAL(SP), PARAMETER :: TOL=1.0e-4_sp
    REAL(SP) :: ax,bx,fa,fb,fx,xmin,xx
    ncom=assert_eq(size(p),size(xi),'linmin')
    pcom=>p
    xicom=>xi
    ax=0.0
    xx=1.0
    call mnbrak(ax,xx,bx,fa,fx,fb,f1dim)
    fret=brent(ax,xx,bx,f1dim,TOL,xmin)
    xi=xmin*xi
    p=p+xi
    END SUBROUTINE linmin


END MODULE