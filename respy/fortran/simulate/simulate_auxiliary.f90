!******************************************************************************
!******************************************************************************
MODULE simulate_auxiliary

    !/* external modules    */

    USE shared_interface

    !/* setup   */

    IMPLICIT NONE

    PUBLIC

CONTAINS
!******************************************************************************
!*****************************************************************************
FUNCTION get_random_edu_start(edu_spec, is_debug) RESULT(edu_start)

    !/* external objects    */

    TYPE(EDU_DICT), INTENT(IN)      :: edu_spec

    LOGICAL, INTENT(IN)             :: is_debug

    INTEGER(our_int)                :: edu_start(num_agents_sim)

    !/* internal objects    */

    TYPE(EDU_DICT)                  :: edu_spec_sorted

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: u

    LOGICAL                         :: READ_IN

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------


    ! We want to ensure that the order of initial schooling levels in the initialization files does not matter for the simulated sample. That is why we create an ordered version for this function.
    edu_spec_sorted = sort_edu_spec(edu_spec)

    INQUIRE(FILE='.initial_schooling.respy.test', EXIST=READ_IN)

    IF ((READ_IN .EQV. .True.)  .AND. (is_debug .EQV. .True.)) THEN

        OPEN(NEWUNIT=u, FILE='.initial_schooling.respy.test', ACTION='READ')
        DO i = 1, num_agents_sim
            88 FORMAT(i10)
            READ(u, 88) edu_start(i)
        END DO

        CLOSE(u)

    ELSE
        DO i = 1, num_agents_sim
            edu_start(i) = get_random_draw(edu_spec_sorted%start, edu_spec_sorted%share)
        END DO

    END IF

END FUNCTION
!******************************************************************************
!*****************************************************************************
FUNCTION get_random_lagged_start(edu_spec, edu_start, is_debug) RESULT(lagged_start)

    !/* external objects    */

    TYPE(EDU_DICT), INTENT(IN)      :: edu_spec

    INTEGER(our_int), INTENT(IN)    :: edu_start(num_agents_sim)

    LOGICAL, INTENT(IN)             :: is_debug

    INTEGER(our_int)                :: lagged_start(num_agents_sim)

    !/* internal objects    */

    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j
    INTEGER(our_int)                :: u
    LOGICAL                         :: READ_IN

    REAL(our_dble)                  :: probs(2)

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    INQUIRE(FILE='.initial_lagged.respy.test', EXIST=READ_IN)

    IF ((READ_IN .EQV. .True.)  .AND. (is_debug .EQV. .True.)) THEN
        OPEN(NEWUNIT=u, FILE='.initial_lagged.respy.test', ACTION='READ')
        DO i = 1, num_agents_sim
            88 FORMAT(i10)
            READ(u, 88) lagged_start(i)
        END DO

        CLOSE(u)

    ELSE
        DO i = 1, num_agents_sim

            ! We need to determine the corresponding position of the lagged probability entry.
            DO j = 1, SIZE(edu_spec%start)
                IF(edu_start(i) .EQ. edu_spec%start(j)) EXIT
            END DO

            probs = (/ edu_spec%lagged(j), one_dble - edu_spec%lagged(j) /)
            lagged_start(i) = get_random_draw((/ three_int, four_int /), probs)

        END DO

    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION get_random_types(num_types, optim_paras, num_agents_sim, edu_start, is_debug) RESULT(types)

    !/* external objects    */

    INTEGER(our_int), INTENT(IN)        :: num_agents_sim
    INTEGER(our_int), INTENT(IN)        :: num_types
    INTEGER(our_int), INTENT(IN)        :: edu_start(num_agents_sim)
    INTEGER(our_int)                    :: types(num_agents_sim)

    TYPE(OPTIMPARAS_DICT), INTENT(IN)   :: optim_paras

    LOGICAL, INTENT(IN)                 :: is_debug

    !/* internal objects    */

    INTEGER(our_int)                    :: type_info_order(num_types)
    INTEGER(our_int)                    :: i
    INTEGER(our_int)                    :: u

    REAL(our_dble)                      :: type_info_shares(num_types * 2)
    REAL(our_dble)                      :: probs(num_types)

    LOGICAL                             :: READ_IN

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL sort_type_info(type_info_order, type_info_shares, optim_paras%type_shares)

    INQUIRE(FILE='.types.respy.test', EXIST=READ_IN)

    IF ((READ_IN .EQV. .True.)  .AND. (is_debug .EQV. .True.)) THEN

        OPEN(NEWUNIT=u, FILE='.types.respy.test', ACTION='READ')
        DO i = 1, num_agents_sim
            87 FORMAT(i10)
            READ(u, 87) types(i)
        END DO

        CLOSE(u)

    ELSE
        DO i = 1, num_agents_sim
            probs = get_conditional_probabilities(type_info_shares, edu_start(i))
            types(i) = get_random_draw(type_info_order, probs)
        END DO

    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION get_random_draw(candidates, probs) RESULT(rslt)

    !/* external objects    */

    REAL(our_dble), INTENT(IN)      :: probs(:)

    INTEGER(our_int), INTENT(IN)    :: candidates(:)

    INTEGER(our_int)                :: rslt

    !/* internal objects    */

    INTEGER(our_int)                :: i

    REAL(our_dble)                  :: u

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    CALL RANDOM_NUMBER(u)

    DO i = 1, SIZE(candidates)
        IF (u < probs(i)) EXIT
        u = u - probs(i)
    END DO

    rslt = candidates(i)

END FUNCTION
!******************************************************************************
!******************************************************************************
SUBROUTINE sort_type_info(type_info_order, type_info_shares, type_shares)

    !/* external objects    */

    REAL(our_dble), INTENT(OUT)     :: type_info_shares(num_types * 2)

    REAL(our_dble), INTENT(IN)      :: type_shares(num_types * 2)

    INTEGER(our_int), INTENT(OUT)   :: type_info_order(num_types)

    !/* internal objects        */

    REAL(our_dble)                  :: type_intercepts_sorted(num_types)
    REAL(our_dble)                  :: type_intercepts(num_types)
    REAL(our_dble)                  :: type_shares_array(num_types, 2)

    INTEGER(our_int)                :: lower
    INTEGER(our_int)                :: upper
    INTEGER(our_int)                :: i
    INTEGER(our_int)                :: j

    LOGICAL                         :: is_duplicated

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    ! We first start by determining a unique ordering of the types based on the value of the intercept.
    j = 1
    DO i = 1, num_types * 2, 2
        type_intercepts(j) = type_shares(i)
        j = j + 1
    END DO

    type_intercepts_sorted = sorted(type_intercepts, num_types)

     DO i = 1, num_types
         DO j = 1, num_types
             IF(type_intercepts_sorted(i) .NE. type_intercepts(j)) CYCLE
             type_info_order(i) = j - 1
         END DO
    END DO

    ! We cannot enforce a unique oder if types are identical with respect to their intercepts. In that case we defaul to their order of specification.
    is_duplicated = .False.
    DO i = 1, num_types - 1
        DO j = i + 1, num_types
            IF(type_info_order(i) .EQ. type_info_order(j)) is_duplicated = .True.
        END DO
    END DO

    IF(is_duplicated) THEN
        DO i = 1, num_types
            type_info_order(i) = i - 1
        END DO
    END IF

    ! We then align the coefficients with the new ordering.
    type_shares_array = TRANSPOSE(RESHAPE(type_shares, (/2, num_types/)))

    DO i = 1, num_types
        j = type_info_order(i) + 1
        lower = (i - 1) * 2 + 1
        upper = i  * 2
        type_info_shares(lower:upper) = type_shares_array(j, :)
     END DO

END SUBROUTINE
!******************************************************************************
!******************************************************************************
END MODULE
