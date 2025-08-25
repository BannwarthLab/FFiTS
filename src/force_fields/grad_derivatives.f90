module grad_derivatives
    use ssff_datatype
    use calc_type
    use dihedral_derivatives
    use ff_utility
    use ff_interface
    implicit none 
    
contains

subroutine get_complete_first_derivative_bondangdih(hopot, grad_ref, grad_ff, xyz_ff, grad)
    type(ssFF_data) :: hopot
    real(wp), intent(in) :: grad_ref(:), grad_ff(:)
    real(wp), intent(inout) :: xyz_ff(:,:)
    real(wp), intent(out) :: grad(:)

    integer :: a, i, j, l, m, index

    grad = 0.0_wp
    ! print *, 'bondos'
    index = 1
    do a = 1, hopot%count_bond
        i = hopot%bond_list(1, a)
        j = hopot%bond_list(2, a)
        call derivative_bond_first_atomwise(hopot%nat, grad_ref, grad_ff, hopot%bondlengths(a), xyz_ff, i, j, hopot%c_bond(i,j), grad(index))
        print *, grad(index)
        index = index + 1
    end do 


    print *, 'angos'
    do a = 1, hopot%count_angle
        i = hopot%angle_list(1, a)
        j = hopot%angle_list(2, a)
        l = hopot%angle_list(3, a)
        call derivative_angle_first_atomwise(hopot%nat, grad_ref, grad_ff, hopot%angles(a), xyz_ff, i, j, l, hopot%c_angle(i,j,l), grad(index))
        print *, grad(index)
        index = index + 1
    end do 

    print *, 'dihedros'
    do a = 1, hopot%count_dihedral
        i = hopot%dihedral_list(1, a)
        j = hopot%dihedral_list(2, a)
        l = hopot%dihedral_list(3, a)
        m = hopot%dihedral_list(4, a)
        call derivative_dihedral_first_atomwise(hopot%nat, grad_ref, grad_ff, hopot%dihedrals(a), xyz_ff, i,j,l,m, hopot%c_dihedral(i,j,l,m), grad(index))
        print *, grad(index)
        index = index + 1
    end do 
    
end subroutine get_complete_first_derivative_bondangdih

subroutine gradfit_objectivefun(nat, grad_ff, grad_ref, delta)
    integer, intent(in) :: nat
    real(wp), intent(in) :: grad_ff(nat*3), grad_ref(nat*3)
    real(wp), intent(out) ::  delta
    
    integer :: i
    
    delta = 0.0_wp

    do i = 1, 3*nat
        delta = delta + (grad_ref(i)**2 - grad_ff(i)**2)**2
    end do

end subroutine gradfit_objectivefun

subroutine derivative_bond_first_atomwise(nat, grad_ref, grad_ff, val_ref, xyz, atom1, atom2, c, deriv)
    integer, intent(in) :: nat, atom1, atom2
    real(wp), intent(in) :: grad_ref(nat*3), grad_ff(nat*3), val_ref, xyz(3,nat)
    real(wp), intent(in) :: c
    real(wp), intent(out) :: deriv

    integer :: i, j, a, b
    real(wp)  :: bondlength_displ
    real(wp), allocatable :: gradient_single(:)

    if (.not. allocated(gradient_single)) allocate(gradient_single(nat*3))
    
    deriv = 0.0_wp
    gradient_single = 0.0_wp
    call get_bond_gradient_two_atoms(xyz, val_ref, atom1, atom2, c, gradient_single)
    call get_bondlength(xyz, atom1, atom2, bondlength_displ)
    
    call sum_bond_first_derivative(atom1, val_ref, bondlength_displ, grad_ff, grad_ref, gradient_single, deriv)
    call sum_bond_first_derivative(atom2, val_ref, bondlength_displ, grad_ff, grad_ref, gradient_single, deriv)

    if (allocated(gradient_single)) deallocate(gradient_single)
end subroutine derivative_bond_first_atomwise

subroutine derivative_bond_second_atomwise(nat, grad_ref, grad_ff, val_ref, xyz, atom1, atom2, c, deriv)
    integer, intent(in) :: nat, atom1, atom2
    real(wp), intent(in) :: grad_ref(nat*3), grad_ff(nat*3), val_ref, xyz(3,nat)
    real(wp), intent(in) :: c
    real(wp), intent(out) :: deriv

    integer :: i, j, a, b
    real(wp)  :: bondlength_displ
    real(wp), allocatable :: gradient_single(:)

    if (.not. allocated(gradient_single)) allocate(gradient_single(nat*3))
    
    deriv = 0.0_wp
    gradient_single = 0.0_wp
    call get_bond_gradient_two_atoms(xyz, val_ref, atom1, atom2, c, gradient_single)
    call get_bondlength(xyz, atom1, atom2, bondlength_displ)
    
    call sum_bond_second_derivative(atom1, val_ref, bondlength_displ, grad_ff, grad_ref, gradient_single, deriv)
    call sum_bond_second_derivative(atom2, val_ref, bondlength_displ, grad_ff, grad_ref, gradient_single, deriv)

    if (allocated(gradient_single)) deallocate(gradient_single)
end subroutine derivative_bond_second_atomwise

subroutine derivative_angle_first_atomwise(nat, grad_ref, grad_ff, val_ref, xyz, atom1, atom2, atom3, c, deriv)
    integer, intent(in) :: nat, atom1, atom2, atom3
    real(wp), intent(in) :: grad_ref(nat*3), grad_ff(nat*3), val_ref, xyz(3,nat)
    real(wp), intent(in) :: c
    real(wp), intent(out) :: deriv

    integer :: i, j, a, b
    real(wp)  :: angle_ref, angle_displ
    real(wp), allocatable :: gradient_single(:)

    if (.not. allocated(gradient_single)) allocate(gradient_single(nat*3))
    
    deriv = 0.0_wp
    gradient_single = 0.0_wp
    call get_angle_gradient_three_atoms(xyz, val_ref, atom1, atom2, atom3, c, gradient_single)
    call get_angle(xyz, atom1, atom2, atom3, angle_displ)
    
    call sum_angle_first_derivative(atom1, val_ref, angle_displ, grad_ff, grad_ref, gradient_single, deriv)
    call sum_angle_first_derivative(atom2, val_ref, angle_displ, grad_ff, grad_ref, gradient_single, deriv)
    call sum_angle_first_derivative(atom3, val_ref, angle_displ, grad_ff, grad_ref, gradient_single, deriv)

    if (allocated(gradient_single)) deallocate(gradient_single)
end subroutine derivative_angle_first_atomwise

subroutine derivative_angle_second_atomwise(nat, grad_ref, grad_ff, val_ref, xyz, atom1, atom2, atom3, c, deriv)
    integer, intent(in) :: nat, atom1, atom2, atom3
    real(wp), intent(in) :: grad_ref(nat*3), grad_ff(nat*3), val_ref, xyz(3,nat)
    real(wp), intent(in) :: c
    real(wp), intent(out) :: deriv

    integer :: i, j, a, b
    real(wp)  :: angle_displ
    real(wp), allocatable :: gradient_single(:)

    if (.not. allocated(gradient_single)) allocate(gradient_single(nat*3))
    
    deriv = 0.0_wp
    gradient_single = 0.0_wp
    call get_angle_gradient_three_atoms(xyz, val_ref, atom1, atom2, atom3, c, gradient_single)
    call get_angle(xyz, atom1, atom2, atom3, angle_displ)
    
    call sum_angle_second_derivative(atom1, val_ref, angle_displ, grad_ff, grad_ref, gradient_single, deriv)
    call sum_angle_second_derivative(atom2, val_ref, angle_displ, grad_ff, grad_ref, gradient_single, deriv)
    call sum_angle_second_derivative(atom3, val_ref, angle_displ, grad_ff, grad_ref, gradient_single, deriv)

    if (allocated(gradient_single)) deallocate(gradient_single)
end subroutine derivative_angle_second_atomwise

subroutine derivative_dihedral_first_atomwise(nat, grad_ref, grad_ff, val_ref, xyz, atom1, atom2, atom3, atom4, c, deriv)
integer, intent(in) :: nat, atom1, atom2, atom3, atom4
real(wp), intent(in) :: grad_ref(nat*3), grad_ff(nat*3), val_ref, xyz(3,nat)
real(wp), intent(in) :: c
real(wp), intent(out) :: deriv

integer :: i, j, a, b
real(wp)  :: dihedral_ref, dihedral_displ
real(wp), allocatable :: gradient_single(:)
integer, allocatable :: dih(:)

if (.not. allocated(gradient_single)) allocate(gradient_single(nat*3), dih(4))

deriv = 0.0_wp
gradient_single = 0.0_wp
call get_dihedral_gradient_four_atoms(xyz, val_ref, atom1, atom2, atom3, atom4, c, gradient_single)
call get_dihedral_angle(xyz, atom1, atom2, atom3, atom4, dihedral_displ)

dih = (/atom1, atom2, atom3, atom4/)

call sum_dihedral_first_derivative(atom1, dih, xyz, c, val_ref, dihedral_displ, grad_ff, grad_ref, gradient_single, deriv)
call sum_dihedral_first_derivative(atom2, dih, xyz, c, val_ref, dihedral_displ, grad_ff, grad_ref, gradient_single, deriv)
call sum_dihedral_first_derivative(atom3, dih, xyz, c, val_ref, dihedral_displ, grad_ff, grad_ref, gradient_single, deriv)
call sum_dihedral_first_derivative(atom4, dih, xyz, c, val_ref, dihedral_displ, grad_ff, grad_ref, gradient_single, deriv)

if (allocated(gradient_single)) deallocate(gradient_single, dih)
end subroutine derivative_dihedral_first_atomwise

subroutine derivative_dihedral_second_atomwise(nat, grad_ref, grad_ff, val_ref, xyz, atom1, atom2, atom3, atom4, c, deriv)
    integer, intent(in) :: nat, atom1, atom2, atom3, atom4
    real(wp), intent(in) :: grad_ref(nat*3), grad_ff(nat*3), val_ref, xyz(3,nat)
    real(wp), intent(in) :: c
    real(wp), intent(out) :: deriv
    
    integer :: i, j, a, b
    real(wp)  :: dihedral_ref, dihedral_displ
    real(wp), allocatable :: gradient_single(:)
    integer, allocatable :: dih(:)
    
    if (.not. allocated(gradient_single)) allocate(gradient_single(nat*3), dih(4))
    
    deriv = 0.0_wp
    gradient_single = 0.0_wp
    call get_dihedral_gradient_four_atoms(xyz, val_ref, atom1, atom2, atom3, atom4, c, gradient_single)
    call get_dihedral_angle(xyz, atom1, atom2, atom3, atom4, dihedral_displ)
    ! call get_dihedral_angle(xyz_ref, atom1, atom2, atom3, atom4, dihedral_ref)
    
    dih = (/atom1, atom2, atom3, atom4/)
    
    call sum_dihedral_second_derivative(atom1, dih, xyz, c, val_ref, dihedral_displ, grad_ff, grad_ref, gradient_single, deriv)
    call sum_dihedral_second_derivative(atom2, dih, xyz, c, val_ref, dihedral_displ, grad_ff, grad_ref, gradient_single, deriv)
    call sum_dihedral_second_derivative(atom3, dih, xyz, c, val_ref, dihedral_displ, grad_ff, grad_ref, gradient_single, deriv)
    call sum_dihedral_second_derivative(atom4, dih, xyz, c, val_ref, dihedral_displ, grad_ff, grad_ref, gradient_single, deriv)
    
    if (allocated(gradient_single)) deallocate(gradient_single, dih)
    end subroutine derivative_dihedral_second_atomwise
    

subroutine sum_bond_first_derivative(atom, bondlength_ref, bondlength_displ, grad_ff, grad_ref, grad_single, sum) 
    integer, intent(in) :: atom
    real(wp), intent(in) :: bondlength_ref, bondlength_displ
    real(wp), intent(in) :: grad_ref(:), grad_ff(:), grad_single(:)
    real(wp), intent(inout) :: sum
    integer :: i
    do i = 3*atom-2, 3*atom
        sum = sum + (1.0_wp / (bondlength_displ - bondlength_ref)) * (4.0_wp * grad_ff(i) * (grad_ref(i)**2.0_wp) * grad_single(i) &
            - 4.0_wp * (grad_ff(i)**3.0_wp) * grad_single(i))
        ! write (*,*) atom, i, grad_ref(i), grad_ff(i), grad_single(i), sum
    end do
    
end subroutine sum_bond_first_derivative

subroutine sum_angle_first_derivative(atom, angle_ref, angle_displ, grad_ff, grad_ref, grad_single, sum) 
    integer, intent(in) :: atom
    real(wp), intent(in) :: angle_ref, angle_displ
    real(wp), intent(in) :: grad_ref(:), grad_ff(:), grad_single(:)
    real(wp), intent(inout) :: sum
    integer :: i
    do i = 3*atom-2, 3*atom
        sum = sum + (1.0_wp / (angle_displ - angle_ref)) * (4 * grad_ff(i) * grad_ref(i)**2 * grad_single(i) &
            - 4 * grad_ff(i)**3 * grad_single(i))
    end do
    
end subroutine sum_angle_first_derivative

subroutine sum_dihedral_first_derivative(atom, dih, xyz, c, dihedral_ref, dihedral_displ, grad_ff, grad_ref, grad_single, sum) 
    integer, intent(in) :: atom, dih(4)
    real(wp), intent(in) :: dihedral_ref, dihedral_displ, c, xyz(:,:)
    real(wp), intent(in) :: grad_ref(:), grad_ff(:)
    real(wp), intent(inout) :: grad_single(:)
    real(wp), intent(inout) :: sum
    real(8) :: cosphi, sinphi, cosphi0, sinphi0
    integer :: i
    grad_single = 0.0_wp
    call get_1single_dihedral_gradient(xyz, dih, dihedral_ref, c**2, grad_single)

    do i = 3*atom-2, 3*atom
        sum = sum - 4.0_wp * grad_ff(i) * (grad_ref(i)**2 - grad_ff(i)**2) * grad_single(i) !(2.0_wp * c**2 * (cosphi * sinphi0 - sinphi * cosphi0) ) 
    end do
    
end subroutine sum_dihedral_first_derivative

subroutine sum_bond_second_derivative(atom, bondlength_ref, bondlength_displ, grad_ff, grad_ref, grad_single, sum) 
    integer, intent(in) :: atom
    real(wp), intent(in) :: bondlength_ref, bondlength_displ
    real(wp), intent(in) :: grad_ref(:), grad_ff(:), grad_single(:)
    real(wp), intent(inout) :: sum
    integer :: i
    do i = 3*atom-2, 3*atom
        sum = sum + (1.0_wp / ((bondlength_displ - bondlength_ref))**2) * (4 * grad_single(i)**2 * (grad_ref(i)**2 - 3 * grad_ff(i)**2))
    end do
    
end subroutine sum_bond_second_derivative

subroutine sum_angle_second_derivative(atom, angle_ref, angle_displ, grad_ff, grad_ref, grad_single, sum) 
    integer, intent(in) :: atom
    real(wp), intent(in) :: angle_ref, angle_displ
    real(wp), intent(in) :: grad_ref(:), grad_ff(:), grad_single(:)
    real(wp), intent(inout) :: sum
    integer :: i
    do i = 3*atom-2, 3*atom
        sum = sum + (1.0_wp / (angle_displ - angle_ref)) * (4 * grad_single(i)**2 * (grad_ref(i)**2 - 3 * grad_ff(i)**2))
    end do
    
end subroutine sum_angle_second_derivative

subroutine sum_dihedral_second_derivative(atom, dih, xyz, c, dihedral_ref, dihedral_displ, grad_ff, grad_ref, grad_single, sum) 
    integer, intent(in) :: atom, dih(4)
    real(wp), intent(in) :: dihedral_ref, dihedral_displ, c, xyz(:,:)
    real(wp), intent(in) :: grad_ref(:), grad_ff(:)
    real(wp), intent(inout) ::  grad_single(:)
    real(wp), intent(inout) :: sum
    real(8) :: cosphi, sinphi, cosphi0, sinphi0
    integer :: i
    
    
    call get_1single_dihedral_gradient(xyz, dih, dihedral_ref, c**2, grad_single)


    do i = 3*atom-2, 3*atom
        sum = sum - 4.0_wp * grad_ff(i) * (grad_ref(i)**2 - grad_ff(i)**2) * (2.0_wp * c**2 * (cosphi * cosphi0 + sinphi * sinphi0) ) 
    end do
    
end subroutine sum_dihedral_second_derivative


subroutine get_1single_dihedral_gradient(geometry, dihedral, phi0, k_dihedrals, gradient)
        
    real(8), intent(in) :: geometry(:, :), phi0
    integer, intent(in) ::  dihedral(4)
    real(8), intent(in) :: k_dihedrals
    real(8), intent(inout) :: gradient(:)
    real(8) :: cosphi0, sinphi0, cosphi, sinphi
    integer :: i,j,k,l, x_hat_i, y_hat_i, z_hat_i, x_hat_j, y_hat_j, z_hat_j
    integer :: xhat(12)
    real(8) :: gxi, gyi, gzi, gxj, gyj, gzj, gxk, gyk, gzk, gxl, gyl, gzl
    real(8) :: dcosdx(12), dsindx(12), d2cosdxdy(78), d2sindxdy(78), aijkl, bijk, cjkl, dijkl
    integer :: m

    i = dihedral(1)

    xhat(1) = 1+3*(i-1)
    xhat(2) = 2+3*(i-1)
    xhat(3) = 3+3*(i-1)

    j = dihedral(2)

    xhat(4) = 1+3*(j-1)
    xhat(5) = 2+3*(j-1)
    xhat(6) = 3+3*(j-1)

    k = dihedral(3)

    xhat(7) = 1+3*(k-1)
    xhat(8) = 2+3*(k-1)
    xhat(9)= 3+3*(k-1)

    l = dihedral(4)

    xhat(10) = 1+3*(l-1)
    xhat(11) = 2+3*(l-1)
    xhat(12) = 3+3*(l-1)

    call get_phi_derivatives(geometry, i, j, k, l, dcosdx, d2cosdxdy,dsindx, d2sindxdy, aijkl, bijk, cjkl, dijkl)

    cosphi0 = cos(phi0)
    sinphi0 = sin(phi0)
    cosphi = aijkl/(bijk*cjkl)
    sinphi = dijkl/(bijk*cjkl)

    call build_1dihedral_gradient(phi0, cosphi0,sinphi0,sinphi, cosphi, dcosdx, dsindx,k_dihedrals,xhat, gradient)

end subroutine get_1single_dihedral_gradient

subroutine build_1dihedral_gradient(phi0, cosphi0,sinphi0,sinphi, cosphi,dcosdx, dsindx, k_dihedrals,xhat, gradient)

    real(8), intent(in) :: phi0, cosphi0, sinphi0, sinphi, cosphi, dcosdx(12), dsindx(12), k_dihedrals
    integer, intent(in) :: xhat(12)
    real(8), intent(inout) :: gradient(:)
    real(8) :: g
    integer :: n

    do n=1, 12

        call get_1dVdihedraldx(dcosdx(n), dsindx(n), k_dihedrals, cosphi0, sinphi0, cosphi, sinphi, g)
        gradient(xhat(n)) = gradient(xhat(n)) + g

    enddo
end subroutine build_1dihedral_gradient

subroutine get_1dVdihedraldx(dcosdx, dsindx, k_dihedrals, cosphi0, sinphi0, cosphi, sinphi, dVdihedraldx)
    real(8), intent(in) :: dcosdx, dsindx, k_dihedrals, cosphi0, sinphi0, cosphi, sinphi
    real(8), intent(out) :: dVdihedraldx
    
    dVdihedraldx = 2*k_dihedrals*((sinphi0)*dcosdx-(cosphi0)*dsindx)
    
end subroutine get_1dVdihedraldx




end module grad_derivatives