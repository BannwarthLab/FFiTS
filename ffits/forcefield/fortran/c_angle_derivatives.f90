module c_angle_derivatives
    ! Original script by Nils van Staalduinen
    ! Modified by Daria Babushkina
use b_angle_derivatives, only: get_b_angle_derivatives
use fortran_helper, only: gidx
implicit none
public :: get_c_angle_derivatives
! integer, parameter :: wp = selected_real_kind(15)
contains

subroutine get_c_angle_derivatives(xkj,ykj,zkj,cjk, dcdx, d2cdxdy)
    real(8), intent(in) :: xkj,ykj,zkj,cjk
    real(8), intent(out) :: dcdx(9), d2cdxdy(45)
    real(8) :: dbdx(9), d2bdxdy(45)
    integer :: m, i, j, ii, jj, ij
    integer, parameter :: wp = selected_real_kind(15)
    dcdx = 0.0_wp
    d2cdxdy = 0.0_wp

    call get_b_angle_derivatives(xkj,ykj,zkj,cjk,dbdx,d2bdxdy)
    
    do m = 1, 3
        dcdx(m+6) = dbdx(m)
    enddo
    do m = 4, 6
        dcdx(m) = dbdx(m)
    enddo

    ! packed upper triangle of the 6x6 b block, m = (j-1)*j/2 + i for i <= j;
    ! coordinates 1-3 of b map to 7-9 of the angle, 4-6 stay
    m = 0
    do j = 1, 6
    do i = 1, j
        m = m + 1
        ii = i
        jj = j
        if (ii < 4) then
            ii = ii + 6
        endif
        if (jj < 4) then
            jj = jj + 6
        endif
        call gidx(ii, jj, ij)
        d2cdxdy(ij) = d2bdxdy(m)
    enddo
    enddo

end subroutine get_c_angle_derivatives

end module c_angle_derivatives
