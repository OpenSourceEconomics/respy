MODULE powell_function


CONTAINS
SUBROUTINE powell(p,xi,ftol,iter,fret)
USE nrtype; USE nrutil, ONLY :assert_eq,nrerror
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
n=assert_eq(size(p),size(xi,1),size(xi,2),'powell')
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

    SUBROUTINE linmin(p,xi,fret)
    USE nrtype; USE nrutil, ONLY : assert_eq
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










    SUBROUTINE mnbrak(ax,bx,cx,fa,fb,fc,func)
    USE nrtype; USE nrutil, ONLY : swap
    IMPLICIT NONE
    REAL(SP), INTENT(INOUT) :: ax,bx
    REAL(SP), INTENT(OUT) :: cx,fa,fb,fc
    INTERFACE
        FUNCTION func(x)
        USE nrtype
        IMPLICIT NONE
        REAL(SP), INTENT(IN) :: x
        REAL(SP) :: func
        END FUNCTION func
    END INTERFACE
    REAL(SP), PARAMETER :: GOLD=1.618034_sp,GLIMIT=100.0_sp,TINY=1.0e-20_sp
    REAL(SP) :: fu,q,r,u,ulim
    fa=func(ax)
    fb=func(bx)
    if (fb > fa) then
        call swap(ax,bx)
        call swap(fa,fb)
    end if
    cx=bx+GOLD*(bx-ax)
    fc=func(cx)
    do
        if (fb < fc) RETURN
        r=(bx-ax)*(fb-fc)
        q=(bx-cx)*(fb-fa)
        u=bx-((bx-cx)*q-(bx-ax)*r)/(2.0_sp*sign(max(abs(q-r),TINY),q-r))
        ulim=bx+GLIMIT*(cx-bx)
        if ((bx-u)*(u-cx) > 0.0) then
            fu=func(u)
            if (fu < fc) then
                ax=bx
                fa=fb
                bx=u
                fb=fu
                RETURN
            else if (fu > fb) then
                cx=u
                fc=fu
                RETURN
            end if
            u=cx+GOLD*(cx-bx)
            fu=func(u)
        else if ((cx-u)*(u-ulim) > 0.0) then
            fu=func(u)
            if (fu < fc) then
                bx=cx
                cx=u
                u=cx+GOLD*(cx-bx)
                call shft(fb,fc,fu,func(u))
            end if
        else if ((u-ulim)*(ulim-cx) >= 0.0) then
            u=ulim
            fu=func(u)
        else
            u=cx+GOLD*(cx-bx)
            fu=func(u)
        end if
        call shft(ax,bx,cx,u)
        call shft(fa,fb,fc,fu)
    end do
    CONTAINS
!BL
    SUBROUTINE shft(a,b,c,d)
    REAL(SP), INTENT(OUT) :: a
    REAL(SP), INTENT(INOUT) :: b,c
    REAL(SP), INTENT(IN) :: d
    a=b
    b=c
    c=d
    END SUBROUTINE shft
    END SUBROUTINE mnbrak


    FUNCTION brent(ax,bx,cx,func,tol,xmin)
    USE nrtype; USE nrutil, ONLY : nrerror

    IMPLICIT NONE
    REAL(SP), INTENT(IN) :: ax,bx,cx,tol
    REAL(SP), INTENT(OUT) :: xmin
    REAL(SP) :: brent
    INTERFACE
        FUNCTION func(x)
        USE nrtype
        IMPLICIT NONE
        REAL(SP), INTENT(IN) :: x
        REAL(SP) :: func
        END FUNCTION func
    END INTERFACE
    INTEGER(I4B), PARAMETER :: ITMAX=100
    REAL(SP), PARAMETER :: CGOLD=0.3819660_sp,ZEPS=1.0e-3_sp*epsilon(ax)
    INTEGER(I4B) :: iter
    REAL(SP) :: a,b,d,e,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm
    a=min(ax,cx)
    b=max(ax,cx)
    v=bx
    w=v
    x=v
    e=0.0
    fx=func(x)
    fv=fx
    fw=fx
    do iter=1,ITMAX
        xm=0.5_sp*(a+b)
        tol1=tol*abs(x)+ZEPS
        tol2=2.0_sp*tol1
        if (abs(x-xm) <= (tol2-0.5_sp*(b-a))) then
            xmin=x
            brent=fx
            RETURN
        end if
        if (abs(e) > tol1) then
            r=(x-w)*(fx-fv)
            q=(x-v)*(fx-fw)
            p=(x-v)*q-(x-w)*r
            q=2.0_sp*(q-r)
            if (q > 0.0) p=-p
            q=abs(q)
            etemp=e
            e=d
            if (abs(p) >= abs(0.5_sp*q*etemp) .or. &
                p <= q*(a-x) .or. p >= q*(b-x)) then
                e=merge(a-x,b-x, x >= xm )
                d=CGOLD*e
            else
                d=p/q
                u=x+d
                if (u-a < tol2 .or. b-u < tol2) d=sign(tol1,xm-x)
            end if
        else
            e=merge(a-x,b-x, x >= xm )
            d=CGOLD*e
        end if
        u=merge(x+d,x+sign(tol1,d), abs(d) >= tol1 )
        fu=func(u)
        if (fu <= fx) then
            if (u >= x) then
                a=x
            else
                b=x
            end if
            call shft(v,w,x,u)
            call shft(fv,fw,fx,fu)
        else
            if (u < x) then
                a=u
            else
                b=u
            end if
            if (fu <= fw .or. w == x) then
                v=w
                fv=fw
                w=u
                fw=fu
            else if (fu <= fv .or. v == x .or. v == w) then
                v=u
                fv=fu
            end if
        end if
    end do
    call nrerror('brent: exceed maximum iterations')
    CONTAINS
!BL
    SUBROUTINE shft(a,b,c,d)
    REAL(SP), INTENT(OUT) :: a
    REAL(SP), INTENT(INOUT) :: b,c
    REAL(SP), INTENT(IN) :: d
    a=b
    b=c
    c=d
    END SUBROUTINE shft
    END FUNCTION brent

END MODULE