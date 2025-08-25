module numerical_derivatives
    
    use fortran_helper
    use ff_interface
    use ff_utility
    use calc_type
    use grad_derivatives
    implicit none 
contains 

subroutine numerical_deriv1_bondangledih(hopot, xyz_ff, grad_ref, grad_bondangdih)
    integer, parameter :: wp = selected_real_kind(15)
    type(ssFF_data) :: hopot
    real(wp), intent(inout) :: xyz_ff(:,:), grad_ref(:)
    real(wp), intent(out) :: grad_bondangdih(:) ! needs to be changed according to which type of constant is looked at
    real(wp), allocatable :: delta, hess_ff1(:, :), hess_ff2(:, :)
    real(wp), allocatable :: bond(:), bond_temp1(:), bond_temp2(:), ang(:), ang_temp1(:), ang_temp2(:), dih(:), dih_temp1(:), dih_temp2(:)
    real(wp) :: delta_1, delta_2
    real(wp), allocatable :: grad(:), grad1(:), grad2(:)
    integer :: i, j, l, m, index, atom1, atom2, atom3, atom4, n_atom 

    n_atom = hopot%nat
    allocate(bond(hopot%count_bond), bond_temp1(hopot%count_bond), bond_temp2(hopot%count_bond))
    allocate(ang(hopot%count_angle), ang_temp1(hopot%count_angle), ang_temp2(hopot%count_angle))
    allocate(dih(hopot%count_dihedral), dih_temp1(hopot%count_dihedral), dih_temp2(hopot%count_dihedral))


    allocate(grad1(3*hopot%nat), grad2(3*hopot%nat))

    ! To calculate the gradient of delta2 in regards to the bond/angle/dihedral constants, decomment the corresponding parts
    allocate(grad(3*hopot%nat))
    grad = 0.0_wp
    grad1 = 0.0_wp
    grad2 =  0.0_wp
    delta = 0.001_wp
    index = 1

    grad_bondangdih = 0.0_wp

    index = 1
    print *, "Numerical"

    bond = hopot%bondlengths
    bond_temp1 = hopot%bondlengths
    bond_temp2 = hopot%bondlengths
    print *, 'bondis'
    do i = 1, hopot%count_bond
        bond_temp1(i) = bond_temp1(i) - delta
        bond_temp2(i) = bond_temp2(i) + delta

        hopot%bondlengths = bond_temp1
        call get_complete_gradient(xyz_ff, hopot, grad1)
        hopot%bondlengths = bond_temp2
        call get_complete_gradient(xyz_ff, hopot, grad2)

        call gradfit_objectivefun(n_atom, grad1, grad_ref, delta_1)
        call gradfit_objectivefun(n_atom, grad2, grad_ref, delta_2)

        grad_bondangdih(index) = (delta_2-delta_1)/(2.0_wp*delta)

        print *, grad_bondangdih(index)
        index = index + 1
        bond_temp1 = bond
        bond_temp2 = bond 
        hopot%bondlengths = bond
    end do 
    ang = hopot%angles
    ang_temp1 = hopot%angles
    ang_temp2 = hopot%angles

    print *, 'winklis'
    do i = 1, hopot%count_angle
        ang_temp1(i) = ang_temp1(i) - delta
        ang_temp2(i) = ang_temp2(i) + delta

        hopot%angles = ang_temp1
        call get_complete_gradient(xyz_ff, hopot, grad1)
        hopot%angles = ang_temp2
        call get_complete_gradient(xyz_ff, hopot, grad2)

        call gradfit_objectivefun(n_atom, grad1, grad_ref, delta_1)
        call gradfit_objectivefun(n_atom, grad2, grad_ref, delta_2)

        grad_bondangdih(index) = (delta_2-delta_1)/(2.0_wp*delta)
        print *, grad_bondangdih(index)
        index = index + 1
        ang_temp1 = ang
        ang_temp2 = ang
        hopot%angles = ang
    end do 

    dih = hopot%dihedrals
    dih_temp1 = hopot%dihedrals
    dih_temp2 = hopot%dihedrals
    print *, 'torsionis'
    do i = 1, hopot%count_dihedral
        dih_temp1(i) = dih_temp1(i) - delta
        dih_temp2(i) = dih_temp2(i) + delta

        hopot%dihedrals = dih_temp1
        call get_complete_gradient(xyz_ff, hopot, grad1)
        hopot%dihedrals = dih_temp2
        call get_complete_gradient(xyz_ff, hopot, grad2)
        call gradfit_objectivefun(n_atom, grad1, grad_ref, delta_1)
        call gradfit_objectivefun(n_atom, grad2, grad_ref, delta_2)

        grad_bondangdih(index) = (delta_2-delta_1)/(2.0_wp*delta)
        print *, grad_bondangdih(index)
        index = index + 1
        dih_temp1 = dih
        dih_temp2 = dih
        hopot%dihedrals = dih
    end do 

    print *, grad_bondangdih
    print *, "End Numerical"

    if (allocated(grad) ) deallocate(grad)

    deallocate(bond, ang, dih, bond_temp1, bond_temp2, ang_temp1, ang_temp2, dih_temp1, dih_temp2)
    deallocate(grad1, grad2)
end subroutine numerical_deriv1_bondangledih

subroutine numerical_deriv1_c(hopot, xyz_ff, gradient_c)
    integer, parameter :: wp = selected_real_kind(15)
    type(ssFF_data) :: hopot
    real(wp), intent(inout) :: xyz_ff(:,:)
    real(wp), intent(out) :: gradient_c(:) ! needs to be changed according to which type of constant is looked at
    real(wp), allocatable :: delta, hess_ff1(:, :), hess_ff2(:, :)
    real(wp), allocatable :: c_btemp1(:,:),c_b0(:,:), c_btemp2(:,:),c_ljtemp1(:,:),c_lj0(:,:), c_ljtemp2(:,:), c_atemp1(:, :,:), c_a0(:, :, :), c_atemp2(:, :, :)
    real(wp), allocatable :: c_dtemp2(:,:,:,:), c_d0(:,:,:,:), c_dtemp1(:,:,:,:)
    real(wp) :: delta_1, delta_2
    real(wp), allocatable :: grad(:)
    integer :: i, j, l, m, index, atom1, atom2, atom3, atom4, n_atom 

    n_atom = hopot%nat

    allocate(c_btemp1(n_atom,n_atom),c_b0(n_atom,n_atom), c_btemp2(n_atom,n_atom),c_ljtemp1(n_atom,n_atom),c_lj0(n_atom,n_atom), c_ljtemp2(n_atom,n_atom), c_atemp1(n_atom, n_atom,n_atom), c_a0(n_atom, n_atom, n_atom), c_atemp2(n_atom, n_atom, n_atom))
    allocate(hess_ff1(3*n_atom, 3*n_atom), hess_ff2(3*n_atom, 3*n_atom))
    allocate(c_dtemp2(n_atom,n_atom,n_atom,n_atom), c_d0(n_atom,n_atom,n_atom,n_atom), c_dtemp1(n_atom,n_atom,n_atom,n_atom))

    ! To calculate the gradient of delta2 in regards to the bond/angle/dihedral constants, decomment the corresponding parts
    allocate(grad(3*n_atom))
    grad = 0.0_wp
    delta = 0.0001_wp
    index = 1
    c_btemp1 = hopot%c_bond
    c_b0 = hopot%c_bond
    c_btemp2 = hopot%c_bond
    c_ljtemp1 = hopot%c_lj
    c_ljtemp2 = hopot%c_lj
    c_lj0 = hopot%c_lj
    c_atemp1 = hopot%c_angle
    c_a0 = hopot%c_angle
    c_atemp2 = hopot%c_angle
    c_dtemp1 = hopot%c_dihedral
    c_dtemp2 = hopot%c_dihedral
    c_d0 = hopot%c_dihedral
    print *, "Numerical"
    do i = 1, hopot%count_bond
        atom1 = hopot%bond_list(1, i)
        atom2 = hopot%bond_list(2, i)
        c_btemp1(atom1,atom2) = c_btemp1(atom1,atom2) - delta 
        c_btemp2(atom1,atom2) = c_btemp2(atom1,atom2) + delta 

        hopot%c_bond = c_btemp1
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff1)
        hopot%c_bond = c_btemp2
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff2)
        call delta2_func(n_atom, hess_ff1, hopot%hessian, delta_1) 
        call delta2_func(n_atom, hess_ff2, hopot%hessian, delta_2)
        gradient_c(index)  = (delta_2-delta_1)/(2.0_wp*delta)
        print *, gradient_c(index)
        index  = index + 1
        hopot%c_bond = c_b0
        c_btemp1 = hopot%c_bond
        c_btemp2 = hopot%c_bond
    end do 
    print *, 'angle'
    do i = 1, hopot%count_angle
        atom1 = hopot%angle_list(1, i)
        atom2 = hopot%angle_list(2, i)
        atom3 = hopot%angle_list(3, i)
        c_atemp1(atom1, atom2, atom3) = c_atemp1(atom1, atom2, atom3) - delta 
        c_atemp2(atom1, atom2, atom3) = c_atemp2(atom1, atom2, atom3) + delta 

        hopot%c_angle = c_atemp1
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff1)
        hopot%c_angle = c_atemp2
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff2)
        call delta2_func(n_atom, hess_ff1, hopot%hessian, delta_1) 
        call delta2_func(n_atom, hess_ff2, hopot%hessian, delta_2)
        ! write (*,*) delta_1, delta_2
        gradient_c(index) = (delta_2-delta_1)/(2.0_wp*delta)
        print *, gradient_c(index)
        index  = index + 1

        hopot%c_angle = c_a0
        c_atemp1 = hopot%c_angle
        c_atemp2 = hopot%c_angle
    end do 
    print *, 'dihedral'
    do i = 1, hopot%count_dihedral
        atom1 = hopot%dihedral_list(1, i)
        atom2 = hopot%dihedral_list(2, i)
        atom3 = hopot%dihedral_list(3, i)
        atom4 = hopot%dihedral_list(4, i)
        c_dtemp1(atom1, atom2, atom3, atom4) = c_dtemp1(atom1, atom2, atom3, atom4) - delta 
        c_dtemp2(atom1, atom2, atom3, atom4) = c_dtemp2(atom1, atom2, atom3, atom4) + delta 

        hopot%c_dihedral = c_dtemp1
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff1)
        hopot%c_dihedral = c_dtemp2
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff2)
        call delta2_func(n_atom, hess_ff1, hopot%hessian, delta_1) 
        call delta2_func(n_atom, hess_ff2, hopot%hessian, delta_2)
        gradient_c(index) = (delta_2-delta_1)/(2.0_wp*delta)
        print *, gradient_c(index)
        index  = index + 1

        hopot%c_dihedral = c_d0
        c_dtemp1 = hopot%c_dihedral
        c_dtemp2 = hopot%c_dihedral
    end do 


    ! do i = 1, hopot%count_lj
    !     atom1 = hopot%lj_list(1, i)
    !     atom2 = hopot%lj_list(2, i)
    !     c_ljtemp1(atom1,atom2) = c_ljtemp1(atom1,atom2) - delta 
    !     c_ljtemp2(atom1,atom2) = c_ljtemp2(atom1,atom2) + delta 
        
    !     hopot%c_lj = c_ljtemp1
    !     call get_complete_hessian(hopot, xyz_ff, grad, hess_ff1)
    !     hopot%c_lj = c_ljtemp2
    !     call get_complete_hessian(hopot, xyz_ff, grad, hess_ff2)
    !     call delta2_func(n_atom, hess_ff1, hopot%hessian, delta_1) 
    !     call delta2_func(n_atom, hess_ff2, hopot%hessian, delta_2)
    !     gradient_c(index)  = (delta_2-delta_1)/(2.0_wp*delta)
    !     ! print *, gradient_c(index)
    !     index  = index + 1
    !     hopot%c_lj = c_lj0
    !     c_ljtemp1 = hopot%c_lj
    !     c_ljtemp2 = hopot%c_lj
    ! end do
    ! print *, grad
    ! print *, gradient_c
    print *, "End Numerical"

    if (allocated(grad) ) deallocate(grad)
end subroutine numerical_deriv1_c

subroutine numerical_deriv2_c(hopot, xyz_ff, gradient_c)
    integer, parameter :: wp = selected_real_kind(15)
    type(ssFF_data) :: hopot
    real(wp), intent(inout) :: xyz_ff(:,:)
    real(wp), intent(out) :: gradient_c(:)
    
    real(wp), allocatable :: delta, val1, val2, hess_ff(:, :), hess_ff1(:, :), hess_ff2(:, :)
    real(wp), allocatable :: c_btemp1(:,:),c_b0(:,:), c_btemp2(:,:),c_ljtemp1(:,:),c_lj0(:,:), c_ljtemp2(:,:), c_atemp1(:, :,:), c_a0(:, :, :), c_atemp2(:, :, :)
    real(wp), allocatable :: c_dtemp2(:,:,:,:), c_d0(:,:,:,:), c_dtemp1(:,:,:,:)
    real(wp) :: delta_1, delta_2, delta_3
    real(wp), allocatable :: grad(:)
    integer :: i, j, l, m, index, atom1, atom2, atom3, atom4, n_atom 

    n_atom = hopot%nat

    allocate(c_btemp1(n_atom,n_atom),c_b0(n_atom,n_atom), c_btemp2(n_atom,n_atom),c_ljtemp1(n_atom,n_atom),c_lj0(n_atom,n_atom), c_ljtemp2(n_atom,n_atom), c_atemp1(n_atom, n_atom,n_atom), c_a0(n_atom, n_atom, n_atom), c_atemp2(n_atom, n_atom, n_atom))
    allocate(hess_ff1(3*n_atom, 3*n_atom),hess_ff(3*n_atom, 3*n_atom), hess_ff2(3*n_atom, 3*n_atom))
    allocate(c_dtemp2(n_atom,n_atom,n_atom,n_atom), c_d0(n_atom,n_atom,n_atom,n_atom), c_dtemp1(n_atom,n_atom,n_atom,n_atom))

    if (.not. allocated(grad)) allocate(grad(n_atom*3))

    delta = 0.0001_wp
    index = 1
    c_btemp1 = hopot%c_bond
    c_b0 = hopot%c_bond
    c_btemp2 = hopot%c_bond
    c_ljtemp1 = hopot%c_lj
    c_ljtemp2 = hopot%c_lj
    c_lj0 = hopot%c_lj
    c_atemp1 = hopot%c_angle
    c_a0 = hopot%c_angle
    c_atemp2 = hopot%c_angle
    c_dtemp1 = hopot%c_dihedral
    c_dtemp2 = hopot%c_dihedral
    c_d0 = hopot%c_dihedral

    gradient_c = 0.0_wp

    do i = 1, hopot%count_bond      
        atom1 = hopot%bond_list(1, i)
        atom2 = hopot%bond_list(2, i)
        c_btemp1(atom1,atom2) = c_btemp1(atom1,atom2) - 2.0_wp*delta 
        c_btemp2(atom1,atom2) = c_btemp2(atom1,atom2) + 2.0_wp*delta 


        hopot%c_bond = c_btemp1
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff1)
        hopot%c_bond = c_btemp2
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff2)
        hopot%c_bond = c_b0
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff)

        
        call delta2_func(n_atom, hess_ff1, hopot%hessian, delta_1) 
        call delta2_func(n_atom, hess_ff2, hopot%hessian, delta_2)
        call delta2_func(n_atom, hess_ff,  hopot%hessian, delta_3) 
        
        gradient_c(index) = (delta_1 + delta_2 - 2.0_wp*delta_3)/(4.0_wp*delta**2)
        index  = index + 1
        c_btemp1 = hopot%c_bond
        c_btemp2 = hopot%c_bond
    end do 

    do i = 1, hopot%count_angle
        atom1 = hopot%angle_list(1, i)
        atom2 = hopot%angle_list(2, i)
        atom3 = hopot%angle_list(3, i)
        c_atemp1(atom1,atom2,atom3) = c_atemp1(atom1,atom2,atom3) - 2.0_wp*delta 
        c_atemp2(atom1,atom2,atom3) = c_atemp2(atom1,atom2,atom3) + 2.0_wp*delta 

        hopot%c_angle = c_atemp1
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff1)
        hopot%c_angle = c_atemp2
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff2)
        hopot%c_angle = c_a0
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff)

        
        call delta2_func(n_atom, hess_ff1, hopot%hessian, delta_1) 
        call delta2_func(n_atom, hess_ff2, hopot%hessian, delta_2)
        call delta2_func(n_atom, hess_ff,  hopot%hessian, delta_3) 
        gradient_c(index) = (delta_1 + delta_2 - 2.0_wp*delta_3)/(4.0_wp*delta**2)
        index  = index + 1
        c_atemp1 = hopot%c_angle
        c_atemp2 = hopot%c_angle
    end do 

    do i = 1, hopot%count_dihedral
        atom1 = hopot%dihedral_list(1, i)
        atom2 = hopot%dihedral_list(2, i)
        atom3 = hopot%dihedral_list(3, i)
        atom4 = hopot%dihedral_list(4, i)
        c_dtemp1(atom1,atom2,atom3,atom4) = c_dtemp1(atom1,atom2,atom3,atom4) - 2.0_wp*delta 
        c_dtemp2(atom1,atom2,atom3,atom4) = c_dtemp2(atom1,atom2,atom3,atom4) + 2.0_wp*delta 
       
        hopot%c_dihedral = c_dtemp1
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff1)
        hopot%c_dihedral = c_dtemp2
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff2)
        hopot%c_dihedral = c_d0
        call get_complete_hessian(hopot, xyz_ff, grad, hess_ff)

        
        call delta2_func(n_atom, hess_ff1, hopot%hessian, delta_1) 
        call delta2_func(n_atom, hess_ff2, hopot%hessian, delta_2)
        call delta2_func(n_atom, hess_ff,  hopot%hessian, delta_3) 

        gradient_c(index) = (delta_1 + delta_2 - 2.0_wp*delta_3)/(4.0_wp*delta**2)
        index  = index + 1  
        c_dtemp1 = c_d0
        c_dtemp2 = c_d0
    end do 

    ! do i = 1, hopot%count_lj    
    !     atom1 = hopot%lj_list(1, i)
    !     atom2 = hopot%lj_list(2, i)
    !     c_ljtemp1(atom1,atom2) = c_ljtemp1(atom1,atom2) - 2.0_wp*delta 
    !     c_ljtemp2(atom1,atom2) = c_ljtemp2(atom1,atom2) + 2.0_wp*delta 
        
    !     hopot%c_lj = c_ljtemp1
    !     call get_complete_hessian(hopot, xyz_ff, grad, hess_ff1)
    !     hopot%c_lj = c_ljtemp2
    !     call get_complete_hessian(hopot, xyz_ff, grad, hess_ff2)
    !     hopot%c_lj = c_lj0
    !     call get_complete_hessian(hopot, xyz_ff, grad, hess_ff)

        
    !     call delta2_func(n_atom, hess_ff1, hopot%hessian, delta_1) 
    !     call delta2_func(n_atom, hess_ff2, hopot%hessian, delta_2)
    !     call delta2_func(n_atom, hess_ff,  hopot%hessian, delta_3) 

    !     gradient_c(index) = (delta_1 + delta_2 - 2.0_wp*delta_3)/(4.0_wp*delta**2)
    !     index  = index + 1
    !     c_ljtemp1 = hopot%c_lj
    !     c_ljtemp2 = hopot%c_lj
    !     end do

    print *, "Numerical"
    print *, gradient_c
    if (allocated(grad)) deallocate(grad)
end subroutine numerical_deriv2_c

subroutine numerical_deriv1_ff(hopot, geometry_ff, gradient)
    ! secant method for checking gradient of force-field energy with respect to coordinates
    integer, parameter :: wp = selected_real_kind(15)
    type(ssFF_data) :: hopot
    real(wp), intent(inout) :: gradient(:), geometry_ff(:,:)

    integer :: i,j,l,m,k,p, n_atom
    real(wp), allocatable :: xyz0(:,:), xyz1(:,:)
    real(wp) :: delta, energy0, energy1, angle_displ, angle
    
    n_atom = hopot%nat
    allocate(xyz0(3, n_atom), xyz1(3, n_atom))
    gradient = 0.0_wp

    delta = 0.00001_wp
    xyz0 = geometry_ff 
    xyz1 = xyz0

    do k=1, n_atom
        do p = 1, 3
            xyz1(p,k) = xyz1(p,k) + delta
            xyz0(p,k) = xyz0(p,k) - delta
            call energy_ff(xyz0, hopot, energy0)
            call energy_ff(xyz1, hopot, energy1)

            gradient(3*(k-1)+p) = (energy1-energy0)/(2.0_wp*delta)
            xyz1 = geometry_ff
            xyz0 = xyz1
        end do
    end do

    ! do i = 1, n_atom
    !     do j = 1, n_atom
    !         if ( i .eq. j ) then
    !             cycle 
    !         else
    !             ! write(*,*) "Analytical", i, j
    !             ! call get_bond_hessian_two_atoms(geometry_ref, geometry_ref, i, j, c_bond(i,j), gradient, hessian)
    !             ! print *, gradient
    !             ! gradient = 0.0_wp
    !             ! write(*,*) "Numerical", i, j
    !             do k=1, n_atom
    !                 do m = 1, 3
    !                     xyz1(m,k) = xyz1(m,k) + delta
    !                     xyz0(m,k) = xyz0(m,k) - delta
    !                     call get_bondlength(geometry_ref, i, j, angle)
    !                     call get_bondlength(xyz1, i, j,  angle_displ)
    !                     energy1 = c_angle(i,j,l)**2*((angle_displ - angle)**2)
    !                     call get_bondlength(geometry_ref, i, j,  angle)
    !                     call get_bondlength(xyz0, i, j,  angle_displ)
    !                     energy0 = c_bond(i,j)**2*((angle_displ - angle)**2)
                
    !                     gradient(3*(k-1)+m) = (energy1-energy0)/(2.0_wp*delta)
    !                     xyz1 = geometry_ref
    !                     xyz0 = xyz1
    !                 end do
    !             end do
    !             ! print *, gradient
    !             ! gradient = 0.0_wp


    !             do l = 1, n_atom
    !                 if ( (l .eq. i) .or. (l .eq. j)) then
    !                     cycle
    !                 else
    !             ! !         write(*,*) "Analytical", i, j, l
    !             !         ! call get_angle_hessian_three_atoms(geometry_ref, geometry_ref, i, j, l, c_angle(i,j,l), gradient, hessian)
    !             ! !         print *, gradient
    !             ! !         gradient = 0.0_wp

    !             !         ! write(*,*) "Numerical", i, j, l
    !                         do k=1, n_atom
    !                             do m = 1, 3
    !                                 xyz1(m,k) = xyz1(m,k) + delta
    !                                 xyz0(m,k) = xyz0(m,k) - delta
    !                                 call get_angle(geometry_ref, i, j, l, angle)
    !                                 call get_angle(xyz1, i, j, l, angle_displ)
    !                                 energy1 = c_angle(i,j,l)**2*((angle_displ - angle)**2)
    !                                 call get_angle(geometry_ref, i, j, l, angle)
    !                                 call get_angle(xyz0, i, j, l, angle_displ)
    !                                 energy0 = c_angle(i,j,l)**2*((angle_displ - angle)**2)
                            
    !                                 gradient(3*(k-1)+m) = (energy1-energy0)/(2.0_wp*delta)
    !                                 xyz1 = geometry_ref
    !                                 xyz0 = xyz1
    !                             end do
    !                         end do
    !             ! !         print *, gradient
    !             ! !         gradient = 0.0_wp

    !                     do m = 1, n_atom
    !                         if ( (m .eq. l) .or. (m .eq. i) .or. (m .eq. j) ) then
    !                             cycle
    !                         else
    !                             ! write(*,*) "Analytical", i, j, l,m
    !                             ! call get_dihedral_hessian_four_atoms(geometry_ref, geometry_ref, i, j, l, m, c_dihedral(i,j,l,m), gradient, hessian)
    !                             ! print *, gradient
    !                             ! gradient = 0.0_wp

    !                             ! write(*,*) "Numerical", i, j, l,m
    !                             do k=1, n_atom
    !                                 do p = 1, 3
    !                                     xyz1(p,k) = xyz1(p,k) + delta
    !                                     xyz0(p,k) = xyz0(p,k) - delta
    !                                     call get_dihedral_angle(geometry_ref, i, j, l,m, angle)
    !                                     call get_dihedral_angle(xyz1, i, j, l,m, angle_displ)
    !                                     energy1 = c_dihedral(i,j,l,m)**2*((angle_displ - angle)**2)
    !                                     call get_dihedral_angle(geometry_ref, i, j, l,m, angle)
    !                                     call get_dihedral_angle(xyz0, i, j, l,m, angle_displ)
    !                                     energy0 = c_dihedral(i,j,l,m)**2*((angle_displ - angle)**2)
                                
    !                                     gradient(3*(k-1)+p) = (energy1-energy0)/(2.0_wp*delta)
    !                                     xyz1 = geometry_ref
    !                                     xyz0 = xyz1
    !                                 end do
    !                             end do
    !                         ! print *, gradient
    !                         ! gradient = 0.0_wp
    !                         end if
    !                     end do
    !                 end if
    !             end do
    !         end if    
    !     end do
    ! end do
    print *, "Numerical"
    print *, gradient
        
end subroutine numerical_deriv1_ff

subroutine numerical_deriv2_ff(geometry_ff, hopot, hessian)
    integer, parameter :: wp = selected_real_kind(15)
    type(ssFF_data) :: hopot
    real(wp), intent(inout) :: geometry_ff(:,:), hessian(:,:)

    integer :: i,j,l,m,k,p, coordinate1, atom1, coordinate2, atom2, n_atom
    real(wp), allocatable :: xyz11(:,:), xyz1m1(:,:), xyzm11(:,:), xyzm1m1(:,:)
    real(wp) :: delta, energy11, energy1m1, energym11, energym1m1 

    n_atom = hopot%nat
    allocate(xyz11(3, n_atom), xyz1m1(3, n_atom), xyzm11(3, n_atom), xyzm1m1(3, n_atom))

    hessian = 0.0_wp
    delta = 0.00001_wp
    xyz11 = geometry_ff
    xyz1m1 = geometry_ff
    xyzm11 = geometry_ff
    xyzm1m1 = geometry_ff



    do j = 1, n_atom*3
        do i = 1, n_atom*3
            write (*,*) i, j
            coordinate1 = MODULO(j, 3)
            coordinate2 = MODULO(i, 3)
            if (coordinate1 .eq. 0) THEN
                atom1 = j / 3 
                coordinate1 = 3
            else    
                atom1 = j / 3 + 1
            end if 
            if (coordinate2 .eq. 0) THEN
                atom2 = i / 3 
                coordinate2 = 3
            else    
                atom2 = i / 3 + 1
            end if 

            xyz11(coordinate1, atom1) = xyz11(coordinate1, atom1) + delta
            xyz11(coordinate2, atom2) = xyz11(coordinate2, atom2) + delta
            xyz1m1(coordinate1, atom1) = xyz1m1(coordinate1, atom1) + delta
            xyz1m1(coordinate2, atom2) = xyz1m1(coordinate2, atom2) - delta
            xyzm11(coordinate1, atom1) = xyzm11(coordinate1, atom1) - delta
            xyzm11(coordinate2, atom2) = xyzm11(coordinate2, atom2) + delta
            xyzm1m1(coordinate1, atom1) = xyzm1m1(coordinate1, atom1) - delta
            xyzm1m1(coordinate2, atom2) = xyzm1m1(coordinate2, atom2) - delta

            call energy_ff(xyz11, hopot, energy11)
            call energy_ff(xyz1m1, hopot, energy1m1)
            call energy_ff(xyzm11, hopot, energym11)
            call energy_ff(xyzm1m1, hopot, energym1m1)


            hessian(j, i) = (energy11 - energy1m1 - energym11 + energym1m1)/(4*delta**2)

            xyz11 = geometry_ff
            xyz1m1 = geometry_ff
            xyzm11 = geometry_ff
            xyzm1m1 = geometry_ff
        end do
    end do
    deallocate(xyz11, xyz1m1, xyzm11, xyzm1m1)

end subroutine numerical_deriv2_ff
end module numerical_derivatives 