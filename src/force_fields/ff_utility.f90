module ff_utility
    use iso_fortran_env, only:wp => real64
    use calc_type
    use bond_derivatives
    use lj_derivatives
    use angle_derivatives 
    use dihedral_derivatives
    use fortran_helper
    implicit none 
    
contains

!=========================================================================================!
!>  Gives a first guess for the forcefield constants for hessian mapping
!>  If no bond-order correction should be applied, use a nat*nat matrix filled with 1
!=========================================================================================!
subroutine get_c_tables(hopot, repulsive_start_ex)
    type(ssFF_data) :: hopot
    real(wp), optional :: repulsive_start_ex
    
    real(wp) :: bondlength, bl_angle1, bl_angle2, bl_dih1, bl_dih2, bl_dih3, repulsive_start
    integer :: i, atom1, atom2, atom3, atom4, j, l, m

    hopot%c_bond = 0.0_wp 
    do i = 1, hopot%count_bond
        atom1 = hopot%bond_list(1, i)
        atom2 = hopot%bond_list(2, i)
        call get_bondlength(hopot%xyz0, atom1, atom2, bondlength)
        hopot%c_bond(atom1, atom2) = (hopot%wbo(atom1,atom2)/bondlength) *1
        ! c_bond(atom2, atom1) = c_bond(atom1, atom2)
    end do 

    hopot%c_angle = 0.0_wp 
    do i = 1, hopot%count_angle
        atom1 = hopot%angle_list(1, i)
        atom2 = hopot%angle_list(2, i)
        atom3 = hopot%angle_list(3, i)
        call get_bondlength(hopot%xyz0, atom1, atom2, bl_angle1)
        call get_bondlength(hopot%xyz0, atom2, atom3, bl_angle2)
        hopot%c_angle(atom1, atom2, atom3) = (hopot%wbo(atom1,atom2)*hopot%wbo(atom2,atom3)/(bl_angle1*bl_angle2))**(1.0_wp/2.0_wp)* 1
        ! c_angle(atom3, atom2, atom1) = c_angle(atom1, atom2, atom3)
    end do 

    hopot%c_dihedral = 0.0_wp 
    do i = 1, hopot%count_dihedral
        atom1 = hopot%dihedral_list(1, i)
        atom2 = hopot%dihedral_list(2, i)
        atom3 = hopot%dihedral_list(3, i)
        atom4 = hopot%dihedral_list(4, i)
        call get_bondlength(hopot%xyz0, atom1, atom2, bl_dih1)
        call get_bondlength(hopot%xyz0, atom2, atom3, bl_dih2)
        call get_bondlength(hopot%xyz0, atom3, atom4, bl_dih3)
        hopot%c_dihedral(atom1,atom2,atom3,atom4) = (hopot%wbo(atom1,atom2)*hopot%wbo(atom2,atom3)*hopot%wbo(atom3,atom4)/(bl_dih1*bl_dih2*bl_dih3))**(1.0_wp/3.0_wp) * 1
        !hopot%c_dihedral(atom1,atom2,atom3,atom4) = (1/(bl_dih1*bl_dih2*bl_dih3))**(1.0_wp/3.0_wp) * 1
        write (*,*) hopot%c_dihedral(atom1,atom2,atom3,atom4), hopot%dihedral_list(:,i)
        write (*,*) hopot%wbo(atom1,atom2),hopot%wbo(atom2,atom3),hopot%wbo(atom3,atom4),bl_dih1,bl_dih2,bl_dih3
        ! c_dihedral(atom4, atom3, atom2, atom1) = c_dihedral(atom1, atom2, atom3, atom4)
    end do 
    
    if (.not. present(repulsive_start_ex)) then 
        repulsive_start = 0.01_wp
    else 
        repulsive_start = repulsive_start_ex
    end if

    hopot%c_lj = 0.0_wp
    do i = 1, hopot%count_lj
        atom1 = hopot%lj_list(1, i)
        atom2 = hopot%lj_list(2, i)
        hopot%c_lj(atom1,atom2) = repulsive_start
    end do 

end subroutine get_c_tables

!=========================================================================================!
!>  Defines the atoms for bonds, angles, dihedral angles and repulsive interaction and 
!>  and writes them in the corresponding lists in hopot. Definition happens by checking 
!>  whether atoms are connected by bonds, which are defined as a bond order > 0.
!=========================================================================================!
subroutine define_relevant_bonds(hopot)
    type(ssFF_data) :: hopot

    integer :: n_atom, i, j, l, m, b, counter, val1, val2
    real(wp) :: bo_threshold

    bo_threshold = 0.0_wp
    n_atom = hopot%nat

    print *, "For defined bonds, angles, dihedral angles and repulsive terms, please look in the generated force field file."
    counter = 0
    hopot%bond_list = 0
    ! print *, "DEFINED BONDS"
    do i = 1, n_atom 
        do j = 1, n_atom 
            if ((hopot%wbo(i,j) .gt. bo_threshold) .and. (i .ne. j)) then 
                do b = 1, counter
                    if ((hopot%bond_list(1, b) .eq. j) .and. (hopot%bond_list(2, b) .eq. i) ) go to 14
                end do
                counter = counter + 1 
                hopot%bond_list(1,counter) = i 
                hopot%bond_list(2,counter) = j 
                ! write (*,*) i, j
14          end if 
        end do 
    end do
    ! TODO add nicer printout

    counter = 0
    hopot%angle_list = 0
    ! print *, "DEFINED ANGLES"
    do i = 1, n_atom
        do j = 1, n_atom
            if ( i .eq. j ) then
                cycle 
            else
                do l = 1, n_atom
                    if ( (l .eq. i) .or. (l .eq. j)) then
                        cycle
                    else
                        if (hopot%wbo(i,j)*hopot%wbo(j,l) .gt. bo_threshold) then 
                            do b = 1, counter
                                if ((hopot%angle_list(1, b) .eq. l) .and. (hopot%angle_list(2, b) .eq. j) .and. (hopot%angle_list(3, b) .eq. i) ) go to 15
                            end do
                            counter = counter + 1 
                            hopot%angle_list(1,counter) = i 
                            hopot%angle_list(2,counter) = j 
                            hopot%angle_list(3,counter) = l 
15                      end if 
                    end if
                end do
            end if    
        end do
    end do

    counter = 0
    hopot%dihedral_list = 0
    ! print *, "DEFINED DIHEDRALS"
    do i = 1, n_atom
        do j = 1, n_atom
            if ( i .eq. j ) then
                cycle 
            else
                do l = 1, n_atom
                    if ( (l .eq. i) .or. (l .eq. j)) then
                        cycle
                    else
                        do m = 1, n_atom
                            if ( (m .eq. l) .or. (m .eq. i) .or. (m .eq. j) ) then
                                cycle
                            else
                                if (hopot%wbo(i,j)*hopot%wbo(j,l)*hopot%wbo(l,m) .gt. bo_threshold) then 
                                    do b = 1, counter
                                        if ((hopot%dihedral_list(1, b) .eq. m) .and. (hopot%dihedral_list(2, b) .eq. l) .and. (hopot%dihedral_list(3, b) .eq. j) .and. (hopot%dihedral_list(4, b) .eq. i) ) go to 16
                                    end do
                                    counter = counter + 1 
                                    hopot%dihedral_list(1,counter) = i 
                                    hopot%dihedral_list(2,counter) = j 
                                    hopot%dihedral_list(3,counter) = l 
                                    hopot%dihedral_list(4,counter) = m
                                    ! write (*,*) i, j, l, m
16                              end if 
                            end if
                        end do
                    end if
                end do
            end if    
        end do
    end do

    counter = 0
    hopot%lj_list = 0
    ! print *, "DEFINED LJ TERMS"
    do i = 1, n_atom 
        do j = 1, n_atom 
            if (i .ne. j) then
                do l = 1, hopot%count_bond
                    val1 = hopot%bond_list(1,l)
                    val2 = hopot%bond_list(2,l)
                    if ((val1 .eq. i) .and. (val2 .eq. j)) go to 17
                    if ((val1 .eq. j) .and. (val2 .eq. i)) go to 17
                end do 
                do l = 1, hopot%count_lj
                    val1 = hopot%lj_list(1,l)
                    val2 = hopot%lj_list(2,l)
                    if ((val1 .eq. j) .and. (val2 .eq. i)) go to 17
                end do 
                counter = counter + 1
                hopot%count_lj = counter 
                hopot%lj_list(1,counter) = i
                hopot%lj_list(2,counter) = j
                ! write (*,*), i, j
17          end if
        end do 
    end do

end subroutine define_relevant_bonds

!=========================================================================================!
!>  Counts how many defined bonds, angles, dihedral angles and repulsive interactions the 
!>  studied system has for correct allocation.
!=========================================================================================!
subroutine count_bonds_angles_dihedrals(hopot)
    type(ssFF_data) :: hopot

    integer :: i, j, l, m, n_atom
    real(wp) :: bo_threshold
    write (*,*) 'VGL', hopot%wbo(45,48), hopot%vander_matrix(45,48) 
    bo_threshold = 0.0_wp

    n_atom = hopot%nat

    hopot%count_bond = 0 
    hopot%count_angle = 0 
    hopot%count_dihedral = 0 
    hopot%count_lj = 0 
    do i = 1, n_atom
        do j = 1, n_atom
            if ( i .eq. j ) then
                cycle 
            else  
                if (hopot%wbo(i,j) .gt. bo_threshold) then 
                    hopot%count_bond = hopot%count_bond + 1
                end if
                do l = 1, n_atom
                    if ( (l .eq. i) .or. (l .eq. j)) then
                        cycle
                    else
                        if (hopot%wbo(i,j)*hopot%wbo(j,l) .gt. bo_threshold) then 
                            hopot%count_angle = hopot%count_angle + 1
                        end if
                        do m = 1, n_atom
                            if ( (m .eq. l) .or. (m .eq. i) .or. (m .eq. j) ) then
                                cycle
                            else
                                if (hopot%wbo(i,j)*hopot%wbo(j,l)*hopot%wbo(l,m) .gt. bo_threshold) then 
                                    hopot%count_dihedral = hopot%count_dihedral + 1
                                end if
                            end if
                        end do
                    end if
                end do
            end if    
        end do
    end do

    hopot%count_bond = hopot%count_bond /2
    hopot%count_angle = hopot%count_angle /2
    hopot%count_dihedral = hopot%count_dihedral /2
    hopot%count_lj = (n_atom*(n_atom-1)) - hopot%count_bond
    if (hopot%count_lj .lt. 0) hopot%count_lj = 0
    print *, hopot%count_lj

end subroutine count_bonds_angles_dihedrals

!=========================================================================================!
!>  Fills a nat*nat matrix with the VdW-distances for the corresponding atoms.  
!=========================================================================================!
subroutine get_vander_matrix(nat, at, vander_values, factor, vander_matrix)
    integer, intent(in) :: nat, at(nat)
    real(wp), intent(in) :: vander_values(86), factor
    real(wp), intent(inout) ::  vander_matrix(nat,nat)

    integer :: i, j
    do i = 1, nat
        do j = 1, nat
            vander_matrix(i,j) = vander_values(at(i)) * factor + vander_values(at(j)) * factor 
        end do 
    end do
end subroutine get_vander_matrix

!=========================================================================================!
!>  Fills a nat*nat matrix with the VdW-distances for the corresponding atoms.  
!=========================================================================================!
subroutine modify_bo_matrix(hopot)
    type(ssFF_data) :: hopot


    integer :: i, j
    real(wp) :: bondlength
    real(wp) :: factor

    factor = 1.5

    do i = 1, hopot%nat
        do j = 1, hopot%nat
            call get_bondlength(hopot%xyz0, i, j, bondlength)
            if (bondlength .gt. hopot%vander_matrix(i,j)/factor) then 
                hopot%wbo(i,j) = 0.0_wp
            end if
        end do 
    end do
end subroutine modify_bo_matrix

!=========================================================================================!
!>  Calculates the values for bonds, angles, dihedral angles, and sigmas (for repulsive 
!>  interactions), which are used for defining the FF.
!=========================================================================================!
subroutine calculate_structural_parameters(hopot)
    type(ssFF_data) :: hopot

    integer :: atom1, atom2, atom3, atom4, i
    real(wp) :: sigma
 
    do i = 1, hopot%count_bond
        atom1 = hopot%bond_list(1, i)
        atom2 = hopot%bond_list(2, i)
        call get_bondlength(hopot%xyz0, atom1, atom2, hopot%bondlengths(i))
    end do

    do i = 1, hopot%count_angle
        atom1 = hopot%angle_list(1, i)
        atom2 = hopot%angle_list(2, i)
        atom3 = hopot%angle_list(3, i)
        call get_angle(hopot%xyz0, atom1, atom2, atom3, hopot%angles(i))
    end do
    
    do i = 1, hopot%count_dihedral
        atom1 = hopot%dihedral_list(1, i)
        atom2 = hopot%dihedral_list(2, i)
        atom3 = hopot%dihedral_list(3, i)
        atom4 = hopot%dihedral_list(4, i)
        call get_dihedral_angle(hopot%xyz0, atom1, atom2, atom3, atom4, hopot%dihedrals(i))
    end do
    do i = 1, hopot%count_lj
        atom1 = hopot%lj_list(1, i)
        atom2 = hopot%lj_list(2, i)
        hopot%lj_lengths(i) = hopot%vander_matrix(atom1,atom2)/(2**(1/6))
    end do
end subroutine calculate_structural_parameters

!========================================================================================!
!> subroutine to get the FF hessian for a specified bond 
!> for given coordinates from the FF defined in hopot
subroutine get_bond_hessian_two_atoms(geometry_ff, val_ref, atom1, atom2, c, gradient, hessian)
    real(wp), intent(in) :: geometry_ff(:, :), val_ref
    integer, intent(in) :: atom1, atom2
    real(wp), intent(in) :: c
    real(wp), intent(inout) :: gradient(:)
    real(wp), intent(inout) :: hessian(:,:)
    integer :: bond_atoms(2)
    real(wp) :: bondlength_ref = 0

    bond_atoms = (/atom1, atom2/)
    hessian = hessian 
    gradient = gradient 
    call get_single_bond_derivative(geometry_ff, bond_atoms, val_ref, c**2, gradient, hessian)
    gradient = gradient
    hessian = hessian
end subroutine get_bond_hessian_two_atoms

!========================================================================================!
!> subroutine to get the FF hessian for a specified pair of repulsive interactions
!> for given coordinates from the FF defined in hopot
subroutine get_lj_hessian_two_atoms(geometry_ff, atom1, atom2, c, sigma, gradient, hessian)
    real(wp), intent(in) :: geometry_ff(:, :)
    integer, intent(in) :: atom1, atom2
    real(wp), intent(in) :: c
    real(wp), intent(in) :: sigma
    real(wp), intent(inout) :: gradient(:)
    real(wp), intent(inout) :: hessian(:,:)
    integer :: lj_atoms(2)
    real(wp) :: bondlength_ref = 0

    lj_atoms = (/atom1, atom2/)
    hessian = hessian 
    gradient = gradient 
    call get_single_lj_derivative(geometry_ff, lj_atoms, sigma, c**2, gradient, hessian)
    gradient = gradient
    hessian = hessian 
end subroutine get_lj_hessian_two_atoms

!========================================================================================!
!> subroutine to get the FF hessian for a specified angle 
!> for given coordinates from the FF defined in hopot
subroutine get_angle_hessian_three_atoms(geometry_ff ,val_ref, atom1, atom2, atom3, c, gradient, hessian)
    real(wp), intent(in) :: geometry_ff(:, :),val_ref
    integer, intent(in) :: atom1, atom2, atom3
    real(wp), intent(in) :: c
    real(wp), intent(inout) :: gradient(:)
    real(wp), intent(inout) :: hessian(:,:)
    integer :: angle_atoms(3)
    real(wp) :: angle_ref = 0

    angle_atoms = (/atom1, atom2, atom3/)
    hessian = hessian 
    gradient = gradient
    call get_single_angle_derivative(geometry_ff, angle_atoms, val_ref, c**2.0_wp, gradient, hessian)
    gradient = gradient
    hessian = hessian 
end subroutine get_angle_hessian_three_atoms

!========================================================================================!
!> subroutine to get the FF hessian for a specified dihedral angle 
!> for given coordinates from the FF defined in hopot
subroutine get_dihedral_hessian_four_atoms(geometry_ff, val_ref, atom1, atom2, atom3, atom4, c, gradient, hessian)
    real(wp), intent(in) :: geometry_ff(:, :), val_ref
    integer, intent(in) :: atom1, atom2, atom3, atom4
    real(wp), intent(in) :: c
    real(wp), intent(inout) :: gradient(:)
    real(wp), intent(inout) :: hessian(:,:)
    integer :: dihedral_atoms(4)
    real(wp) :: dihedral_ref, sin_dihedral_ref, cos_dihedral_ref
    
    dihedral_atoms = (/atom1, atom2, atom3, atom4/)
    hessian = hessian 
    gradient = gradient 
    call get_single_dihedral_derivative(geometry_ff, dihedral_atoms,val_ref, c**2.0_wp, gradient, hessian)
    gradient = gradient 
    hessian = hessian
end subroutine get_dihedral_hessian_four_atoms

!========================================================================================!
!> subroutine to get the FF gradient for a specified bond 
!> for given coordinates from the FF defined in hopot
subroutine get_bond_gradient_two_atoms(geometry_ff, val_ref, atom1, atom2, c, gradient)
    real(wp), intent(in) :: geometry_ff(:, :), val_ref
    integer, intent(in) :: atom1, atom2
    real(wp), intent(in) :: c
    real(wp), intent(inout) :: gradient(:)
    real(8) :: rij
    integer :: xhat(6)
    real(8) :: drdx(6), d2rdxdy(21)

    gradient = gradient 

    xhat(1) = 1+3*(atom1-1)
    xhat(2) = 2+3*(atom1-1)
    xhat(3) = 3+3*(atom1-1)

    xhat(4) = 1+3*(atom2-1)
    xhat(5) = 2+3*(atom2-1)
    xhat(6) = 3+3*(atom2-1)

    call get_r_derivatives(geometry_ff, atom1, atom2, drdx, d2rdxdy, rij)

    call build_bond_gradient(val_ref, rij, drdx, c**2.0_wp, xhat, gradient)
    gradient = gradient 
    
end subroutine get_bond_gradient_two_atoms

!========================================================================================!
!> subroutine to get the FF gradient for a specified pair of repulsive interactions
!> for given coordinates from the FF defined in hopot
subroutine get_lj_gradient_two_atoms(geometry_ff, atom1, atom2, c, sigma, gradient)
    real(wp), intent(in) :: geometry_ff(:, :)
    integer, intent(in) :: atom1, atom2
    real(wp), intent(in) :: c, sigma
    real(wp), intent(inout) :: gradient(:)
    real(8) :: rij
    integer :: xhat(6)
    real(8) :: drdx(6), d2rdxdy(21), bondlength_ref

    gradient = gradient 

    xhat(1) = 1+3*(atom1-1)
    xhat(2) = 2+3*(atom1-1)
    xhat(3) = 3+3*(atom1-1)

    xhat(4) = 1+3*(atom2-1)
    xhat(5) = 2+3*(atom2-1)
    xhat(6) = 3+3*(atom2-1)

    call get_rlj_derivatives(geometry_ff, atom1, atom2, drdx, d2rdxdy, rij)

    call build_lj_gradient(sigma, rij, drdx, c**2.0_wp, xhat, gradient)
    gradient = gradient 
    
end subroutine get_lj_gradient_two_atoms

!========================================================================================!
!> subroutine to get the FF gradient for a specified angle
!> for given coordinates from the FF defined in hopot
subroutine get_angle_gradient_three_atoms(geometry_ff, val_ref, atom1, atom2, atom3, c, gradient)
    real(wp), intent(in) :: geometry_ff(:, :), val_ref
    integer, intent(in) :: atom1, atom2, atom3
    real(wp), intent(in) :: c
    real(wp), intent(inout) :: gradient(:)
    real(8) :: costheta0, costheta
    integer :: xhat(9)
    real(8) :: dcosdx(9), d2cosdxdy(45), aijk , bij, cjk

    gradient = gradient 

    xhat(1) = 1+3*(atom1-1)
    xhat(2) = 2+3*(atom1-1)
    xhat(3) = 3+3*(atom1-1)

    xhat(4) = 1+3*(atom2-1)
    xhat(5) = 2+3*(atom2-1)
    xhat(6) = 3+3*(atom2-1)

    xhat(7) = 1+3*(atom3-1)
    xhat(8) = 2+3*(atom3-1)
    xhat(9)=  3+3*(atom3-1)

    call get_theta_derivatives(geometry_ff, atom1, atom2, atom3, dcosdx, d2cosdxdy, aijk, bij, cjk)

    costheta0 = cos(val_ref)
    costheta = aijk/(bij*cjk)
    call build_angle_gradient(costheta0, costheta, dcosdx, c**2.0_wp, xhat, gradient)
    gradient = gradient 
end subroutine get_angle_gradient_three_atoms

!========================================================================================!
!> subroutine to get the FF hessian for a specified dihedral angle
!> for given coordinates from the FF defined in hopot
subroutine get_dihedral_gradient_four_atoms(geometry_ff, val_ref, atom1, atom2, atom3, atom4, c, gradient)
    real(wp), intent(in) :: geometry_ff(:, :), val_ref
    integer, intent(in) :: atom1, atom2, atom3, atom4
    real(wp), intent(in) :: c
    real(wp), intent(inout) :: gradient(:)
    integer :: dihedral_atoms(4)
    real(8) :: cosphi0, sinphi0, cosphi, sinphi
    integer :: xhat(12)
    real(8) :: dcosdx(12), dsindx(12), d2cosdxdy(78), d2sindxdy(78), aijkl, bijk, cjkl, dijkl

    dihedral_atoms = (/atom1, atom2, atom3, atom4/)
    gradient = gradient 

    cosphi = 0
    sinphi = 0

    xhat(1) = 1+3*(atom1-1)
    xhat(2) = 2+3*(atom1-1)
    xhat(3) = 3+3*(atom1-1)

    xhat(4) = 1+3*(atom2-1)
    xhat(5) = 2+3*(atom2-1)
    xhat(6) = 3+3*(atom2-1)

    xhat(7) = 1+3*(atom3-1)
    xhat(8) = 2+3*(atom3-1)
    xhat(9) = 3+3*(atom3-1)

    xhat(10) = 1+3*(atom4-1)
    xhat(11) = 2+3*(atom4-1)
    xhat(12) = 3+3*(atom4-1)
    call get_phi_derivatives(geometry_ff, atom1, atom2, atom3, atom4, dcosdx, d2cosdxdy,dsindx, d2sindxdy, aijkl, bijk, cjkl, dijkl)

    cosphi0 = cos(val_ref)
    sinphi0 = sin(val_ref)
    cosphi = aijkl/(bijk*cjkl)
    sinphi = dijkl/(bijk*cjkl)

    call build_dihedral_gradient(val_ref, cosphi0,sinphi0,sinphi, cosphi, dcosdx, dsindx,c**2,xhat, gradient)

    gradient = gradient 
end subroutine get_dihedral_gradient_four_atoms

!========================================================================================!
!> function which returns the larger value of the two input values
function larger_of(val1, val2) result(retval)
    real(wp), intent(in) :: val1, val2
    real(wp) :: retval

    retval = val1
    if (val1 .gt. val2) then 
        retval = val1 
    else if (val2 .gt. val1) then 
        retval = val2 
    end if
    
end function larger_of

!========================================================================================!
!> subroutine for calculation of the RMSD between two Hessians
subroutine calculate_hessian_rmsd(hessian1, hessian2, n, rmsd)
    implicit none

    integer, intent(in) :: n  ! Number of atoms or degrees of freedom
    real(kind=8), intent(in) :: hessian1(n,n), hessian2(n,n)  ! Hessian matrices
    real(kind=8), intent(out) :: rmsd
    integer :: i, j

    ! Calculate the RMSD
    rmsd = 0.0
    do i = 1, n
        do j = 1, n
            rmsd = rmsd + 0.5_wp*(hessian1(i,j) - hessian2(i,j))**2
        end do
    end do
    rmsd = sqrt(rmsd / (n * n))

end subroutine calculate_hessian_rmsd
end module ff_utility