module fortran_helper
    ! Original script by Nils van Staalduinen
    ! Modified by Daria Babushkina
implicit none
public :: gidx, get_R_ij, get_theta_ijk, get_cosphi_ijkl, get_sinphi_ijkl, get_aijkl, get_bijk, get_cjkl, get_dijkl
!integer, parameter :: wp = selected_real_kind(15)

    ! TODO: some subroutines are repeating each other. Needs to be cleaned

contains

function cross(a, b)
    real(8), dimension(3) :: cross
    real(8), dimension(3), intent(in) :: a, b

    cross(1) = a(2) * b(3) - a(3) * b(2)
    cross(2) = a(3) * b(1) - a(1) * b(3)
    cross(3) = a(1) * b(2) - a(2) * b(1)
end function cross


subroutine get_bondlength(geometry, atom1, atom2, bondlength)
    real(8), intent(in) :: geometry(:, :)  ! xyz file as array
    integer, intent(in) :: atom1, atom2
    real(8), intent(out) ::  bondlength

    real(8), dimension(3) :: vector_12

    vector_12 = (/geometry(1, atom1), geometry(2, atom1), geometry(3, atom1)/)-&
     (/geometry(1, atom2), geometry(2, atom2), geometry(3, atom2)/) 

    bondlength = SQRT(vector_12(1)**2+vector_12(2)**2+vector_12(3)**2)

    
    
end subroutine get_bondlength

subroutine get_angle(geometry, atom1, atom2, atom3, angle)
    ! returns the angle between 3 atoms, with atom2 being the bridging atom
    real(8), intent(in) :: geometry(:, :)  ! xyz file as array
    integer, intent(in) :: atom1, atom2, atom3  ! indices of atoms, relevant for the angle
    real(8), intent(out) ::  angle  ! angle between atom 1,2 and 3 in degrees

    real(8), dimension(3) :: xyz_1, xyz_2, xyz_3  ! xyz coordinates of atoms
    real(8), dimension(3) :: vector_12, vector_23
    real(8) :: length_12 = 0, length_23 = 0  ! describes the length of the vector \sqrt(x^2+y^2+z^2)
    real(8), parameter :: PI = 1

    ! get points 
    xyz_1 = (/geometry(1, atom1), geometry(2, atom1), geometry(3, atom1)/) 
    xyz_2 = (/geometry(1, atom2), geometry(2, atom2), geometry(3, atom2)/) 
    xyz_3 = (/geometry(1, atom3), geometry(2, atom3), geometry(3, atom3)/) 

    ! get vectors 1-2 and 2-3 and dot product
    vector_12 = xyz_1 - xyz_2
    vector_23 = xyz_3 - xyz_2

    ! get length of vectors
    length_12 = SQRT(vector_12(1)**2+vector_12(2)**2+vector_12(3)**2)
    length_23 = SQRT(vector_23(1)**2+vector_23(2)**2+vector_23(3)**2)

    ! calculate angle [degrees]
    angle = ACOS(DOT_PRODUCT(vector_12, vector_23)/(length_12*length_23))  
end subroutine get_angle

subroutine get_dihedral_angle(geometry, atom1, atom2, atom3, atom4, dihedral_angle)
    real(8), intent(in) :: geometry(:, :)  ! xyz file as array
    integer, intent(in) :: atom1, atom2, atom3, atom4  ! indices of atoms, relevant for the angle
    real(8), intent(out) ::  dihedral_angle  ! dihedral angle between atom 1, 2, 3 and 4 in degrees
    
    real(8), dimension(3) :: xyz_1, xyz_2, xyz_3, xyz_4, vector_12, vector_23, vector_34, normal1, normal2  ! normal are normal vectors
    real(8) :: length1, length2 , length3! lengths of normal-vector 1 and 2
    real(8), parameter :: PI = 3.14159265358979323

    ! get points 
    xyz_1 = (/geometry(1, atom1), geometry(2, atom1), geometry(3, atom1)/) 
    xyz_2 = (/geometry(1, atom2), geometry(2, atom2), geometry(3, atom2)/) 
    xyz_3 = (/geometry(1, atom3), geometry(2, atom3), geometry(3, atom3)/) 
    xyz_4 = (/geometry(1, atom4), geometry(2, atom4), geometry(3, atom4)/) 

    ! get vectors 1-2, 2-3 and 3-4 
    vector_12 = xyz_2 - xyz_1
    vector_23 = xyz_3 - xyz_2
    vector_34 = xyz_4 - xyz_3

    ! get normal vectors 
    normal1 = cross(vector_12, vector_23)
    normal2 = cross(vector_23, vector_34)
    
    
    length1 = SQRT(normal1(1)**2+normal1(2)**2+normal1(3)**2)
    length2 = SQRT(normal2(1)**2+normal2(2)**2+normal2(3)**2)

    length3 = SQRT(vector_23(1)**2+vector_23(2)**2+vector_23(3)**2)
    
    ! dihedral_angle = ACOS(DOT_PRODUCT(normal1, normal2)/(length1*length2)) 
    ! write (*,*) "ACOS", dihedral_angle
    
    dihedral_angle = ATAN2(DOT_PRODUCT(length3*vector_12,normal2),DOT_PRODUCT(normal1, normal2)) 
    ! write (*,*) "ATAN", dihedral_angle
    
end subroutine get_dihedral_angle

! Nils' Code
subroutine gidx(i, j, ij)
    integer, intent(in) :: i, j
    integer, intent(out) :: ij
    if (i < j) then
        ij = (j - 1) * j  / 2 + i 
    else
        ij = (i - 1)* i / 2 + j 
    endif
end subroutine gidx

subroutine get_R_ij(geometry, i, j, R_ij)

    intrinsic :: sqrt   

    real(8), intent(in) :: geometry(:, :)

    integer, intent(in) :: i, j

    real(8), intent(out) :: R_ij

    real(8) :: x_ij, y_ij, z_ij

    x_ij = geometry(1,i)-geometry(1,j)
    y_ij = geometry(2,i)-geometry(2,j)
    z_ij = geometry(3,i)-geometry(3,j)

    R_ij = sqrt(x_ij**2 + y_ij**2 + z_ij**2)

end subroutine get_R_ij

subroutine get_theta_ijk(geometry, i, j, k, theta)

    intrinsic :: dot_product, acos

    real(8), intent(in) :: geometry(:, :)
    integer, intent(in) :: i, j, k
    real(8), intent(out) :: theta
    real(8) :: r_ij(3), r_kj(3)
    real(8) :: numerator,  denominator, norm_r_ij, norm_r_kj

    r_ij = (/ geometry(1,i)-geometry(1,j), geometry(2,i)-geometry(2,j), geometry(3,i)-geometry(3,j) /)

    r_kj = (/ geometry(1,k)-geometry(1,j), geometry(2,k)-geometry(2,j), geometry(3,k)-geometry(3,j) /)

    call get_l2norm(r_ij, norm_r_ij)

    call get_l2norm(r_kj, norm_r_kj)

    numerator = dot_product(r_ij, r_kj)

    denominator = norm_r_ij*norm_r_kj

    theta = acos(numerator/denominator)

end subroutine get_theta_ijk

subroutine get_aijk(geometry, i, j, k, aijk)

    intrinsic :: dot_product
    real(8), intent(in) :: geometry(:, :)
    integer, intent(in) :: i, j, k
    real(8), intent(out) :: aijk
    real(8) :: r_ij(3), r_kj(3)

    r_ij = (/ geometry(1,i)-geometry(1,j), geometry(2,i)-geometry(2,j), geometry(3,i)-geometry(3,j)/)

    r_kj = (/ geometry(1,k)-geometry(1,j), geometry(2,k)-geometry(2,j), geometry(3,k)-geometry(3,j)/)

    aijk = dot_product(r_ij, r_kj)

end subroutine get_aijk

subroutine get_bij(geometry, i, j, bij)

    intrinsic :: dot_product
    real(8), intent(in) :: geometry(:, :)
    integer, intent(in) :: i, j
    real(8), intent(out) :: bij
    real(8) :: r_ij(3)

    r_ij = (/ geometry(1,i)-geometry(1,j), geometry(2,i)-geometry(2,j), geometry(3,i)-geometry(3,j)/)

    call get_l2norm(r_ij, bij)

end subroutine get_bij

subroutine get_aijkl(geometry, i, j, k, l, aijkl)
    intrinsic :: dot_product
    real(8), intent(in) :: geometry(:, :)
    integer, intent(in) :: i, j, k, l
    real(8), intent(out) :: aijkl
    real(8) :: r_ji(3), r_kj(3), r_lk(3), cross_jikj(3), cross_kjlk(3)

    r_ji = (/ geometry(1,j)-geometry(1,i), geometry(2,j)-geometry(2,i), geometry(3,j)-geometry(3,i)/)

    r_kj = (/ geometry(1,k)-geometry(1,j), geometry(2,k)-geometry(2,j), geometry(3,k)-geometry(3,j)/)
    
    r_lk = (/ geometry(1,l)-geometry(1,k), geometry(2,l)-geometry(2,k), geometry(3,l)-geometry(3,k)/)

    call cross_product(r_ji, r_kj, cross_jikj)

    call cross_product(r_kj, r_lk, cross_kjlk)

    aijkl = dot_product(cross_jikj, cross_kjlk)

end subroutine get_aijkl

subroutine get_bijk(geometry, i, j, k, bijk)
    intrinsic :: dot_product
    real(8), intent(in) :: geometry(:, :)
    integer, intent(in) :: i, j, k
    real(8), intent(out) :: bijk
    real(8) :: r_ji(3), r_kj(3), cross_jikj(3)

    r_ji = (/ geometry(1,j)-geometry(1,i), geometry(2,j)-geometry(2,i), geometry(3,j)-geometry(3,i)/)

    r_kj = (/ geometry(1,k)-geometry(1,j), geometry(2,k)-geometry(2,j), geometry(3,k)-geometry(3,j)/)
    
    call cross_product(r_ji, r_kj, cross_jikj)

    call get_l2norm(cross_jikj, bijk)

end subroutine get_bijk

subroutine get_cjkl(geometry, j, k,l, cjkl)
    real(8), intent(in) :: geometry(:, :)
    integer, intent(in) :: j,k,l
    real(8), intent(out) :: cjkl
    real(8) ::r_kj(3), r_lk(3), cross_kjlk(3)

    r_kj = (/ geometry(1,k)-geometry(1,j), geometry(2,k)-geometry(2,j), geometry(3,k)-geometry(3,j)/)

    r_lk = (/ geometry(1,l)-geometry(1,k), geometry(2,l)-geometry(2,k), geometry(3,l)-geometry(3,k)/)
    
    call cross_product(r_kj, r_lk, cross_kjlk)

    call get_l2norm(cross_kjlk, cjkl)

end subroutine get_cjkl

subroutine get_dijkl(geometry, i, j, k, l, dijkl)
    intrinsic :: dot_product
    real(8), intent(in) :: geometry(:, :)
    integer, intent(in) :: i, j, k, l
    real(8), intent(out) :: dijkl
    real(8) :: r_ji(3), r_kj(3), r_lk(3), cross_jikj(3), cross_kjlk(3), cross_ijkl(3)
    real(8) :: norm_r_kj,dot_jikjlk

    r_ji = (/ geometry(1,j)-geometry(1,i), geometry(2,j)-geometry(2,i), geometry(3,j)-geometry(3,i)/)

    r_kj = (/ geometry(1,k)-geometry(1,j), geometry(2,k)-geometry(2,j), geometry(3,k)-geometry(3,j)/)
    
    r_lk = (/ geometry(1,l)-geometry(1,k), geometry(2,l)-geometry(2,k), geometry(3,l)-geometry(3,k)/)

    call cross_product(r_ji, r_kj, cross_jikj)

    call cross_product(r_kj, r_lk, cross_kjlk)

    call get_l2norm(r_kj, norm_r_kj)

    call cross_product(cross_jikj, cross_kjlk, cross_ijkl)

    dot_jikjlk = dot_product(r_kj, cross_ijkl)

    dijkl = dot_jikjlk/norm_r_kj

end subroutine get_dijkl



subroutine get_cosphi_ijkl(geometry, i, j, k, l, cosphi)
    intrinsic :: dot_product
    real(8), intent(in) :: geometry(:, :)
    integer, intent(in) :: i, j, k, l
    real(8), intent(out) :: cosphi
    real(8) :: r_ji(3), r_kj(3), r_lk(3), cross_jikj(3), cross_kjlk(3)
    real(8) :: numerator, denominator, norm_cross_jikj, norm_cross_kjlk

    r_ji = (/ geometry(1,j)-geometry(1,i), geometry(2,j)-geometry(2,i), geometry(3,j)-geometry(3,i)/)

    r_kj = (/ geometry(1,k)-geometry(1,j), geometry(2,k)-geometry(2,j), geometry(3,k)-geometry(3,j)/)
    
    r_lk = (/ geometry(1,l)-geometry(1,k), geometry(2,l)-geometry(2,k), geometry(3,l)-geometry(3,k)/)

    call cross_product(r_ji, r_kj, cross_jikj)

    call cross_product(r_kj, r_lk, cross_kjlk)

    numerator = dot_product(cross_jikj, cross_kjlk)

    call get_l2norm(cross_jikj, norm_cross_jikj)

    call get_l2norm(cross_kjlk, norm_cross_kjlk)

    denominator = norm_cross_jikj * norm_cross_kjlk

    cosphi = numerator/denominator

end subroutine get_cosphi_ijkl

subroutine get_sinphi_ijkl(geometry, i, j, k, l, sinphi)
    intrinsic :: dot_product
    real(8), intent(in) :: geometry(:, :)
    integer, intent(in) :: i, j, k, l
    real(8), intent(out) :: sinphi
    real(8) :: r_ji(3), r_kj(3), r_lk(3), cross_jikj(3), cross_kjlk(3), cross_ijkl(3)
    real(8) :: numerator, denominator, norm_cross_jikj, norm_cross_kjlk, norm_r_kj,dot_jikjlk

    r_ji = (/ geometry(1,j)-geometry(1,i), geometry(2,j)-geometry(2,i), geometry(3,j)-geometry(3,i)/)

    r_kj = (/ geometry(1,k)-geometry(1,j), geometry(2,k)-geometry(2,j), geometry(3,k)-geometry(3,j)/)
    
    r_lk = (/ geometry(1,l)-geometry(1,k), geometry(2,l)-geometry(2,k), geometry(3,l)-geometry(3,k)/)

    call cross_product(r_ji, r_kj, cross_jikj)

    call cross_product(r_kj, r_lk, cross_kjlk)

    call get_l2norm(r_kj, norm_r_kj)
    call cross_product(cross_kjlk, cross_jikj, cross_ijkl)

    dot_jikjlk = dot_product(r_kj, cross_ijkl)

    numerator = dot_jikjlk

    call get_l2norm(cross_jikj, norm_cross_jikj)

    call get_l2norm(cross_kjlk, norm_cross_kjlk)

    denominator = norm_r_kj*norm_cross_jikj * norm_cross_kjlk

    sinphi = numerator/denominator

end subroutine get_sinphi_ijkl

subroutine cross_product(vec1, vec2, vec3)

    real(8), intent(in) :: vec1(3), vec2(3)
    real(8), intent(out) :: vec3(3)

    vec3(1) = vec1(2) * vec2(3) - vec1(3) * vec2(2)
    vec3(2) = vec1(3) * vec2(1) - vec1(1) * vec2(3)
    vec3(3) = vec1(1) * vec2(2) - vec1(2) * vec2(1)
    
end subroutine cross_product

subroutine get_l2norm(vec1, norm)

    intrinsic :: sqrt, selected_real_kind
    integer, parameter :: wp = selected_real_kind(15)
    real(8), intent(in) :: vec1(:)
    real(8), intent(out) :: norm
    integer :: i

    norm = 0.0_wp

    do i=1, size(vec1)
        norm = norm + vec1(i)**2
    enddo

    norm = sqrt(norm)

end subroutine

end module fortran_helper
