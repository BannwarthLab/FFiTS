module ff_interface
    use iso_fortran_env, only:wp => real64
    use calc_type
    use ff_utility
    use bond_derivatives
    use lj_derivatives
    use angle_derivatives 
    use dihedral_derivatives
    use fortran_helper
    use zdata 
    use strucrd, only: rdcoord
    implicit none 
    
    !> THIS IS A COPY FROM optimize/modelhessian.f90 as otherwise there is a circular dependancy
    real(wp),parameter :: vander(86) = (/ &
    & 0.91_wp,0.92_wp, & ! H, He
    & 0.75_wp,1.28_wp,1.35_wp,1.32_wp,1.27_wp,1.22_wp,1.17_wp,1.13_wp, & ! Li-Ne
    & 1.04_wp,1.24_wp,1.49_wp,1.56_wp,1.55_wp,1.53_wp,1.49_wp,1.45_wp, & ! Na-Ar
    & 1.35_wp,1.34_wp, & ! K, Ca
    & 1.42_wp,1.42_wp,1.42_wp,1.42_wp,1.42_wp, & ! Sc-Zn
    & 1.42_wp,1.42_wp,1.42_wp,1.42_wp,1.42_wp, &
    & 1.50_wp,1.57_wp,1.60_wp,1.61_wp,1.59_wp,1.57_wp, & ! Ga-Kr
    & 1.48_wp,1.46_wp, & ! Rb, Sr
    & 1.49_wp,1.49_wp,1.49_wp,1.49_wp,1.49_wp, & ! Y-Cd
    & 1.49_wp,1.49_wp,1.49_wp,1.49_wp,1.49_wp, &
    & 1.52_wp,1.64_wp,1.71_wp,1.72_wp,1.72_wp,1.71_wp, & ! In-Xe
    & 2.00_wp,2.00_wp, &
    & 2.00_wp,2.00_wp,2.00_wp,2.00_wp,2.00_wp,2.00_wp,2.00_wp, & ! La-Yb
    & 2.00_wp,2.00_wp,2.00_wp,2.00_wp,2.00_wp,2.00_wp,2.00_wp, &
    & 2.00_wp,2.00_wp,2.00_wp,2.00_wp,2.00_wp, & ! Lu-Hg
    & 2.00_wp,2.00_wp,2.00_wp,2.00_wp,2.00_wp, &
    & 2.00_wp,2.00_wp,2.00_wp,2.00_wp,2.00_wp,2.00_wp/) ! Tl-Rn
    !>  C6 coefficients used in the D2 model    
contains



!=========================================================================================!
!>  Uses the given information to define all values of the ff besides the ff parameters
!>  namely, used bonds, angles, dihedral angles, VdW distances, atom pairs for rep interactions and so on
!=========================================================================================!
subroutine define_ff_from_structure(hopot, rep)
    type(ssFF_data) :: hopot
    real(wp), optional :: rep
    real(wp),parameter :: bohr = 0.52917726_wp

    call get_vander_matrix(hopot%nat, hopot%at, vander, 1.0_wp/bohr, hopot%vander_matrix) 
   call modify_bo_matrix(hopot)
    call count_bonds_angles_dihedrals(hopot)
    call define_relevant_bonds(hopot)
   call clean_dihedral(hopot)
   call add_improper_dihedral(hopot)
    call calculate_structural_parameters(hopot)
    if (present(rep)) call get_c_tables(hopot=hopot, repulsive_start_ex=rep)
    if (.not. present(rep)) call get_c_tables(hopot=hopot)
    

end subroutine define_ff_from_structure



!=========================================================================================!
!>  removes dihedrals where central bond was already used in other dihedral angles
!=========================================================================================!
subroutine clean_dihedral(hopot)
    type(ssFF_data) :: hopot
    integer :: i, j, at1, at2, at3, at4, counter, counter2, counter3
    integer :: cent1, cent2
    logical :: is_doubled, has_improper_dihedral
    integer, allocatable :: temp_dihedral_list(:,:)
    integer, allocatable :: new_dihedral_temp(:), new_dihedral(:)
    real(wp) :: bo_threshold

    bo_threshold = 0.8_wp

    allocate(temp_dihedral_list(4, hopot%count_dihedral), new_dihedral_temp(8), new_dihedral(4))
    new_dihedral_temp = 0
    new_dihedral = 0
    counter = 0

    do i = 1, hopot%count_dihedral
        write (*,*) "CHECKED DIH IS", hopot%dihedral_list(:, i)
        ! Extract atoms for the current dihedral
        at1 = hopot%dihedral_list(1, i)
        at2 = hopot%dihedral_list(2, i)
        at3 = hopot%dihedral_list(3, i)
        at4 = hopot%dihedral_list(4, i)

        ! Normalize central bond (at2, at3)
        if (at2 < at3) then
            cent1 = at2
            cent2 = at3
        else
            cent1 = at3
            cent2 = at2
        end if

        ! Check if this central bond has already been used
        is_doubled = .false.
        do j = 1, counter
            if ((min(temp_dihedral_list(2,j), temp_dihedral_list(3,j)) == cent1) .and. &
                (max(temp_dihedral_list(2,j), temp_dihedral_list(3,j)) == cent2)) then
                is_doubled = .true.
                exit
            end if
        end do


        if (.not. is_doubled) then
            counter = counter + 1
            temp_dihedral_list(:, counter) = hopot%dihedral_list(:, i)
        end if 

    end do

    ! print *, hopot%wbo

    ! Replace the original list with the filtered list
    hopot%count_dihedral = counter
    hopot%dihedral_list(:,1:counter) = temp_dihedral_list(:,1:counter)
    write (*,*) "DIH COUNT", hopot%count_dihedral

    deallocate(temp_dihedral_list, new_dihedral, new_dihedral_temp)
end subroutine clean_dihedral

subroutine add_improper_dihedral(hopot)
    type(ssFF_data) :: hopot
    integer :: i, j, at1, at2, at3, at4
    integer :: counter, counter2, counter3
    integer :: central_atom
    integer, allocatable :: partner_list(:), new_dihedral(:)
    real(wp) :: bo_threshold
    logical :: has_improper_dihedral

    bo_threshold = 0.8_wp
    counter = 0

    allocate(partner_list(4))
    allocate(new_dihedral(4))

    do i = 1, hopot%count_dihedral
        at1 = hopot%dihedral_list(1, i)
        at2 = hopot%dihedral_list(2, i)
        at3 = hopot%dihedral_list(3, i)
        at4 = hopot%dihedral_list(4, i)

        ! --- Check both central atoms ---
        do central_atom = at2, at3, at3 - at2
            partner_list = 0
            counter2 = 0
            has_improper_dihedral = .false.

            ! Count bonded partners above threshold
            do j = 1, hopot%nat
                if (hopot%wbo(j, central_atom) > bo_threshold) then
                    counter2 = counter2 + 1
                    if (counter2 <= 4) partner_list(counter2) = j
                end if
            end do

            ! Only proceed if exactly four bonded partners
            if (counter2 == 4) then
                counter3 = 0
                new_dihedral = 0

                do j = 1, 4
                    if (partner_list(j) /= merge(at3, at2, central_atom == at2)) then
                        counter3 = counter3 + 1
                        ! Place central atom in second position
                        if (counter3 == 2) then
                            new_dihedral(counter3) = central_atom
                            counter3 = counter3 + 1
                        end if
                        new_dihedral(counter3) = partner_list(j)
                    end if
                end do

                has_improper_dihedral = .true.
            end if

            ! Append if new and valid
            if (has_improper_dihedral) then
                if (.not. is_in_list(new_dihedral, hopot%dihedral_list, hopot%count_dihedral + counter)) then
                    counter = counter + 1
                    hopot%dihedral_list(:, hopot%count_dihedral + counter) = new_dihedral
                end if
            end if
        end do
    end do

    hopot%count_dihedral = hopot%count_dihedral + counter
    write(*,*) "Updated dihedral count:", hopot%count_dihedral

    deallocate(partner_list, new_dihedral)
end subroutine add_improper_dihedral

!=========================================================================================!
!>  Checks wether a value is in the given list
!=========================================================================================!
function is_in_list(val, list, len_list) result(retval)
    integer, intent(in) :: val(:), list(:,:), len_list
    logical :: retval

    integer :: i, j 

    retval = .false.
    do i = 1, len_list
        if (all(list(:,i) .eq. val)) then 
            retval = .true.
        end if 
    end do

end function is_in_list


!========================================================================================!
!> subroutine for the calculation of Delta2, the elementwise, squared difference between two hessians 
subroutine delta2_func(n_atom, hess1, hess2, delta2)
    integer, intent(in) :: n_atom
    real(wp), intent(in) :: hess1(n_atom*3, n_atom*3), hess2(n_atom*3, n_atom*3)
    real(wp), intent(out) :: delta2

    integer :: i, j, a, b

    delta2 = 0.0_wp
    
    do i = 1, n_atom
        do j = 1, n_atom
            if (i .ne. j) then 
                do a = 3*i-2, 3*i
                    do b = 3*j-2, 3*j
                       delta2 = delta2 + 0.5_wp * (hess1(a,b) - hess2(a,b))**2  
                    end do
                end do
            end if 
        end do 
    end do


end subroutine delta2_func

!========================================================================================!
!> subroutine for calculating the FF energy from the values defined in hopot
subroutine energy_ff(geometry_displ, hopot, energy)
    type(ssFF_data) :: hopot
    real(wp), intent(in) :: geometry_displ(3, hopot%nat)
    real(wp), intent(out) :: energy

    real(wp) :: bondlength, angle, dihedral, sigma
    real(wp) :: bondlength_displ, angle_displ, dihedral_displ, fact_bond, fact_ang, fact_dih,  temp
    integer :: a, i, j, m, l

    fact_bond = 1.0_wp
    fact_ang = 1.0_wp
    fact_dih = 1.0_wp
    energy = 0.0_wp
    
    do a = 1, hopot%count_bond
        i = hopot%bond_list(1, a)
        j = hopot%bond_list(2, a)
        call get_bondlength(geometry_displ, i, j, bondlength_displ)
        energy = energy + fact_bond * hopot%c_bond(i,j)**2 * ((bondlength_displ - hopot%bondlengths(a))**2)
    end do 
    ! write (*,*) 'after bond', energy

    do a = 1, hopot%count_angle
        i = hopot%angle_list(1, a)
        j = hopot%angle_list(2, a)
        l = hopot%angle_list(3, a)
        call get_angle(geometry_displ, i, j, l, angle_displ)
        energy = energy + fact_ang*hopot%c_angle(i,j,l)**2 * ((angle_displ - hopot%angles(a))**2)
    end do 
    ! write (*,*) 'after ang', energy

    do a = 1, hopot%count_dihedral
        i = hopot%dihedral_list(1, a)
        j = hopot%dihedral_list(2, a)
        l = hopot%dihedral_list(3, a)
        m = hopot%dihedral_list(4, a)
        call get_dihedral_angle(geometry_displ, i, j, l, m, dihedral_displ)
        temp = energy
        
        energy = energy + fact_dih*hopot%c_dihedral(i,j,l,m)**2*((cos(hopot%dihedrals(a))-cos(dihedral_displ))**2+(sin(hopot%dihedrals(a))-sin(dihedral_displ))**2)
        ! energy = energy + fact_dih*c_dihedral(i,j,l,m)**2*((dihedral_displ - dihedral)**2)
    end do 
    ! write (*,*) 'after dih', energy

    do a = 1, hopot%count_lj
        i = hopot%lj_list(1, a)
        j = hopot%lj_list(2, a)
        ! call get_bondlength(hopot%xyz0, i, j, bondlength)
        call get_bondlength(geometry_displ, i, j, bondlength_displ)
        ! sigma = hopot%vander_matrix(i,j) /(2**(1/6))
        energy = energy + 4 * hopot%c_lj(i,j)**2 * (hopot%lj_lengths(a)/bondlength_displ)**12 !- (hopot%lj_lengths(a)/bondlength_displ)**6) !+ c_lj(i,j)**2
    end do 
    ! write (*,*) 'after lj', energy
end subroutine energy_ff

!========================================================================================!
!> subroutine to get the FF gradient for given coordinates from the FF defined in hopot
subroutine get_complete_gradient(geometry_displ, hopot, gradient)
    type(ssFF_data) :: hopot
    real(wp), intent(in) :: geometry_displ(:, :)
    real(wp), intent(inout) :: gradient(:)
    integer :: a, i, j, l, m

    gradient = 0.0_wp

    do a = 1, hopot%count_bond
        i = hopot%bond_list(1, a)
        j = hopot%bond_list(2, a)
        call get_bond_gradient_two_atoms(geometry_displ, hopot%bondlengths(a), i, j, hopot%c_bond(i,j), gradient)
    end do 

    do a = 1, hopot%count_angle
        i = hopot%angle_list(1, a)
        j = hopot%angle_list(2, a)
        l = hopot%angle_list(3, a)
        call get_angle_gradient_three_atoms(geometry_displ, hopot%angles(a), i, j, l, hopot%c_angle(i,j,l), gradient)
    end do 

    do a = 1, hopot%count_dihedral
        i = hopot%dihedral_list(1, a)
        j = hopot%dihedral_list(2, a)
        l = hopot%dihedral_list(3, a)
        m = hopot%dihedral_list(4, a)
        call get_dihedral_gradient_four_atoms(geometry_displ, hopot%dihedrals(a), i, j, l, m, hopot%c_dihedral(i,j,l,m), gradient)
    end do 

    do a = 1, hopot%count_lj
        i = hopot%lj_list(1, a)
        j = hopot%lj_list(2, a)
        call get_lj_gradient_two_atoms(geometry_displ, i, j, hopot%c_lj(i,j), hopot%lj_lengths(a), gradient)
    end do 

end subroutine get_complete_gradient

!========================================================================================!
!> subroutine to get the FF hessian for given coordinates from the FF defined in hopot
subroutine get_complete_hessian(hopot, geometry_displ, gradient, hessian)
    type(ssFF_data) :: hopot
    real(wp), intent(in) :: geometry_displ(:, :)
    real(wp), intent(inout) :: gradient(:), hessian(:,:)

    integer :: a, i, j, l, m
    

    gradient = 0.0_wp
    hessian = 0.0_wp
    do a = 1, hopot%count_bond
        i = hopot%bond_list(1, a)
        j = hopot%bond_list(2, a)
        call get_bond_hessian_two_atoms(geometry_displ, hopot%bondlengths(a), i, j, hopot%c_bond(i,j), gradient, hessian)
    end do 

    do a = 1, hopot%count_angle
        i = hopot%angle_list(1, a)
        j = hopot%angle_list(2, a)
        l = hopot%angle_list(3, a)
        call get_angle_hessian_three_atoms(geometry_displ, hopot%angles(a), i, j, l, hopot%c_angle(i,j,l), gradient, hessian)
    end do 

    do a = 1, hopot%count_dihedral
        i = hopot%dihedral_list(1, a)
        j = hopot%dihedral_list(2, a)
        l = hopot%dihedral_list(3, a)
        m = hopot%dihedral_list(4, a)
        call get_dihedral_hessian_four_atoms(geometry_displ, hopot%dihedrals(a), i, j, l, m, hopot%c_dihedral(i,j,l,m), gradient, hessian)
    end do 

    do a = 1, hopot%count_lj
        i = hopot%lj_list(1, a)
        j = hopot%lj_list(2, a)
        call get_lj_hessian_two_atoms(geometry_displ, i, j, hopot%c_lj(i,j), hopot%lj_lengths(a), gradient, hessian)
    end do 

end subroutine get_complete_hessian

!========================================================================================!
! COPY from optimize_utils to avoid dependency cycle
subroutine readhess(nat3,h,fname)
    integer,intent(in)  :: nat3
    real(wp),intent(out) :: h(nat3,nat3)
    character(len=*),intent(in) :: fname
    integer  :: iunit,i,j,mincol,maxcol
    character(len=5)  :: adum
    character(len=80) :: a80

    open (newunit=iunit,file=fname)
50  read (iunit,'(a)') a80
    if (index(a80,'$hessian') .ne. 0) then
      do i = 1,nat3
        maxcol = 0
200     mincol = maxcol+1
        maxcol = min(maxcol+5,nat3)
        read (iunit,*) (h(j,i),j=mincol,maxcol)
        if (maxcol .lt. nat3) goto 200
      end do
      close (iunit)
      goto 300
    end if
    goto 50

300 return
  end subroutine readhess

!========================================================================================!
!> subroutine to write a FF, defined in hopot, to a file with name "filename"
subroutine write_force_field(hopot, filename) !UNTESTED
    type(ssFF_data) :: hopot
    character(len=*), intent(inout) :: filename

    integer :: a, i, j, l, m
    real(wp) :: val, sigma

    open(unit=10, file=filename, status='replace')

    
    if (hopot%count_bond .gt. 0.0_wp) then
        write(10, *) "$bonds",',', hopot%count_bond
        do a = 1, hopot%count_bond
            i = hopot%bond_list(1,a)
            j = hopot%bond_list(2,a)
            write (10,"(i3,a,i3,a,F16.8,a,F16.8)") i,',', j,',', hopot%c_bond(i,j),',', hopot%bondlengths(a)
        end do
    end if

    if (hopot%count_angle .gt. 0.0_wp) then
        write(10, *) "$angles",',', hopot%count_angle
        do a = 1, hopot%count_angle
            i = hopot%angle_list(1,a)
            j = hopot%angle_list(2,a)
            l = hopot%angle_list(3,a)
            call get_angle(hopot%xyz0, i, j, l, val)
            write (10,"(i3,a,i3,a,i3,a,F16.8,a,F16.8)") i,',', j,',',l,',', hopot%c_angle(i,j,l),',', hopot%angles(a)
        end do
    end if
    
    if (hopot%count_dihedral.gt. 0.0_wp) then
        write(10, *) "$dihedrals",',', hopot%count_dihedral
        do a = 1, hopot%count_dihedral
            i = hopot%dihedral_list(1,a)
            j = hopot%dihedral_list(2,a)
            l = hopot%dihedral_list(3,a)
            m = hopot%dihedral_list(4,a)
            call get_dihedral_angle(hopot%xyz0, i, j, l, m, val)
            write (10,"(i3,a,i3,a,i3,a,i3,a,F16.8,a,F16.8)") i,',', j,',',l,',',m,',', hopot%c_dihedral(i,j,l,m),',', hopot%dihedrals(a)
        end do
    end if

    if (hopot%count_lj .gt. 0.0_wp) then
        write(10, *) "$lj-terms",',', hopot%count_lj
        do a = 1, hopot%count_lj
            i = hopot%lj_list(1,a)
            j = hopot%lj_list(2,a)
            write (10,"(i3,a,i3,a,F16.8,a,F16.8)") i,',',j,',', hopot%c_lj(i,j),',', hopot%lj_lengths(a)
        end do
    end if

    ! Close file
    close(10)

end subroutine write_force_field

!========================================================================================!
!> subroutine to read a FF, specified by filename and saves the given values in hopot
subroutine read_force_field(hopot, filename)
    type(ssFF_data) :: hopot
    character(len=*), intent(in) :: filename

    integer :: unit_no, iunit, atom1, atom2, atom3, atom4, idx, io, counter
    integer :: count_bond, count_angle, count_dihedral, count_lj, count
    real(wp) :: param, value
    character(len=256) :: line, header, text
    logical :: read_bonds, read_angles, read_dihedrals, read_lj_terms, ex
    ex = .false.
    inquire (file=filename,exist=ex)
    if (ex) then 
        open(unit=10, file=filename)
    else 
        print *, "DATA not found"
    end if
    
    read_bonds = .false.
    read_angles = .false.
    read_dihedrals = .false.
    read_lj_terms = .false.

    hopot%count_lj = 0

    write (*,*) "Force Field definition is read from", filename
    do while (ex)
        read(10, '(a)', iostat=iunit) line
        if (iunit /= 0) exit
        header = trim(adjustl(line))
        if (header(1:6) == "$bonds") then
            read_bonds = .true.
            read_angles = .false.
            read_dihedrals = .false.
            read_lj_terms = .false.
            read (header,*,iostat=io) text,count
            hopot%count_bond = count
            counter = 0
        elseif (header(1:7) == "$angles") then
            read_bonds = .false.
            read_angles = .true.
            read_dihedrals = .false.
            read_lj_terms = .false.
            read (header,*,iostat=io) text,count
            hopot%count_angle = count
            counter = 0
        elseif (header(1:10) == "$dihedrals") then
            read_bonds = .false.
            read_angles = .false.
            read_dihedrals = .true.
            read_lj_terms = .false.
            read (header,*,iostat=io) text,count
            hopot%count_dihedral = count
            counter = 0
        elseif (header(1:9) == "$lj-terms") then
            read_bonds = .false.
            read_angles = .false.
            read_dihedrals = .false.
            read_lj_terms = .true.
            read (header,*,iostat=io) text,count
            hopot%count_lj = count
            counter = 0
            if (hopot%count_lj .eq. 0) read_lj_terms = .False.
        elseif (read_bonds .and. hopot%count_bond .gt. 0) then
            read (header,*,iostat=io) atom1, atom2, param, value
            counter = counter + 1
            hopot%bond_list(1, counter) = atom1
            hopot%bond_list(2, counter) = atom2
            hopot%c_bond(atom1, atom2) = param
            hopot%bondlengths(counter) = value
        elseif (read_angles .and. hopot%count_angle .gt. 0) then
            read (header,*,iostat=io) atom1, atom2, atom3, param, value
            counter = counter + 1
            hopot%angle_list(1, counter) = atom1
            hopot%angle_list(2, counter) = atom2
            hopot%angle_list(3, counter) = atom3
            hopot%c_angle(atom1, atom2, atom3) = param
            hopot%angles(counter) = value
        elseif (read_dihedrals .and. hopot%count_dihedral .gt. 0) then
            read (header,*,iostat=io) atom1, atom2, atom3, atom4, param, value
            counter = counter + 1
            hopot%dihedral_list(1, counter) = atom1
            hopot%dihedral_list(2, counter) = atom2
            hopot%dihedral_list(3, counter) = atom3
            hopot%dihedral_list(4, counter) = atom4
            hopot%c_dihedral(atom1, atom2, atom3, atom4) = param
            hopot%dihedrals(counter) = value
        elseif (read_lj_terms .and. hopot%count_lj .gt. 0) then
            read (header,*,iostat=io) atom1, atom2, param, value
            counter = counter + 1
            hopot%lj_list(1, counter) = atom1
            hopot%lj_list(2, counter) = atom2
            hopot%c_lj(atom1, atom2) = param
            hopot%lj_lengths(counter) = value
        endif
        if (io .ne. 0) cycle
        
    end do
    write (*,*) "Finished with COUNTS:", hopot%count_bond, hopot%count_angle, hopot%count_dihedral, hopot%count_lj

    close(10)

end subroutine read_force_field

subroutine read_fitting_values(filename, maxit, stepsize, threshold, constant_repulsion, rep_start)
    character(len=*), intent(in) :: filename
    integer, intent(out) :: maxit
    logical, intent(out) :: constant_repulsion 
    real(wp), intent(out) :: stepsize, threshold, rep_start
    integer :: a
    logical :: d 
    real(wp) :: b, c, e
    character(len=256) :: line, header
    integer :: iunit, io
    ! open(unit=10, file=filename)


    open(unit=10, file=filename)
    open(unit=10, file=filename)
    read(10, '(a)', iostat=iunit) line
    header = trim(adjustl(line))
    read (header,*,iostat=io) a, b, c, d, e
    maxit = a 
    stepsize = b 
    threshold = c 
    constant_repulsion = d 
    rep_start = e

    close(10)
end subroutine read_fitting_values

subroutine read_weights(filename, weight_1, weight_2)
    implicit none
    character(len=*), intent(in) :: filename
    real(wp), intent(out) :: weight_1, weight_2
    character(len=100) :: line, key, value
    integer :: ios, pos

    ! Open the file
    open(unit=10, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
       print *, 'Error: Unable to open file.'
       stop
    end if

    ! Read the file line by line
    do
       read(10, '(A)', iostat=ios) line
       if (ios /= 0) exit  ! Exit loop at end of file or on error

       ! Remove comments
       if (index(line, '#') > 0) then
          line = line(1:index(line, '#') - 1)
       end if

       ! Trim whitespace from the line
       line = trim(adjustl(line))

       ! Find the colon (:) that separates key and value
       pos = index(line, ':')
       if (pos > 0) then
          key = trim(adjustl(line(1:pos-1)))   ! Extract key before the colon
          value = trim(adjustl(line(pos+1:))) ! Extract value after the colon

          ! Assign the value to the corresponding variable
          if (key == "weight_1") then
             read(value, *) weight_1
          else if (key == "weight_2") then
             read(value, *) weight_2
          end if
       end if
    end do
    if (weight_1 + weight_2 .ne. 1.0_wp) then 
        write (*,*) 'Error: Sum of values from ', filename, ' is not equal to 1.'
        stop
    end if 
    ! Close the file
    close(10)
end subroutine read_weights

    
end module ff_interface