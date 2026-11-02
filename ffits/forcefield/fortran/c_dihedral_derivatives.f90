module c_dihedral_derivatives
    ! Original script by Nils van Staalduinen
    ! Modified by Daria Babushkina
use b_dihedral_derivatives, only: get_b_dihedral_derivatives
use fortran_helper, only: gidx
implicit none
public :: get_c_dihedral_derivatives
! integer, parameter :: wp = selected_real_kind(15)
contains

subroutine get_c_dihedral_derivatives(xji,yji,zji,xkj,ykj,zkj,xlk,ylk,zlk, cjkl, & 
                                dcdx, d2cdxdy)
    real(8), intent(in) :: xji,yji,zji,xkj,ykj,zkj,xlk,ylk,zlk,cjkl
    real(8), intent(out) :: dcdx(12), d2cdxdy(78)
    real(8) :: ddx, ddy, ddz, dbdx(12), d2bdxdy(78)
    integer :: m, i, j, ij
    integer, parameter :: wp = selected_real_kind(15)
    dcdx = 0.0_wp
    d2cdxdy = 0.0_wp

    call get_b_dihedral_derivatives(xkj,ykj,zkj,xlk,ylk,zlk,xji,yji,zji, cjkl,& 
                                dbdx, d2bdxdy)
    
    do m = 1, 9
        dcdx(m+3) = dbdx(m)
    enddo

    ! packed upper triangle of the 9x9 b block, m = (j-1)*j/2 + i for i <= j;
    ! coordinates 1-9 of b map to 4-12 of the dihedral
    m = 0
    do j = 1, 9
    do i = 1, j
        m = m + 1
        call gidx(i + 3, j + 3, ij)
        d2cdxdy(ij) = d2bdxdy(m)
    enddo
    enddo

end subroutine get_c_dihedral_derivatives

end module c_dihedral_derivatives
