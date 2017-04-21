!******************************************************************************
!******************************************************************************
MODULE simulate_auxiliary

    !/* external modules    */

    USE shared_interfaces

    USE shared_constants

    USE shared_utilities

    USE shared_types

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!******************************************************************************
FUNCTION get_random_type(num_types, type_spec) RESULT(type_)

    !/* external objects    */

    TYPE(TYPE_DICT), INTENT(IN)     :: type_spec

    INTEGER(our_int), INTENT(IN)    :: num_types

    INTEGER(our_int)                :: type_

    !/* internal objects    */

    INTEGER(our_int)                :: candidates(num_types)
    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: u

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    candidates = (/ (i, i = 0, num_types - 1) /)

    CALL RANDOM_NUMBER(u)

    DO type_ = 0, num_types - 1
        IF (u < type_spec%shares(type_ + 1)) EXIT
        u = u - type_spec%shares(type_ + 1)
    END DO

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION get_random_types(num_types, type_spec, num_agents_sim, is_debug) RESULT(types)

    !/* external objects    */

    TYPE(TYPE_DICT)                 :: type_spec

    INTEGER(our_int)                :: types(num_agents_sim)
    INTEGER(our_int)                :: num_agents_sim
    INTEGER(our_int)                :: num_types

    LOGICAL, INTENT(IN)             :: is_debug

    !/* internal objects    */

    INTEGER                         :: i

    LOGICAL                         :: READ_IN

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    INQUIRE(FILE='.types.respy.test', EXIST=READ_IN)

    IF ((READ_IN .EQV. .True.)  .AND. (is_debug .EQV. .True.)) THEN

        OPEN(UNIT=99, FILE='.types.respy.test', ACTION='READ')
        DO i = 1, num_agents_sim
            87 FORMAT(i10)
            READ(99, 87) types(i)
        END DO
        CLOSE(99)
    ELSE
        DO i = 1, num_agents_sim
            types(i) = get_random_type(num_types, type_spec)
        END DO

    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
END MODULE
