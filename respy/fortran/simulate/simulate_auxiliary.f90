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

    INTEGER                         :: i

    LOGICAL                         :: READ_IN

!------------------------------------------------------------------------------
! Algorithm
!------------------------------------------------------------------------------

    INQUIRE(FILE='.initial.respy.test', EXIST=READ_IN)

    IF ((READ_IN .EQV. .True.)  .AND. (is_debug .EQV. .True.)) THEN

        OPEN(UNIT=99, FILE='.initial.respy.test', ACTION='READ')
        DO i = 1, num_agents_sim
            88 FORMAT(i10)
            READ(99, 88) edu_start(i)
        END DO

        CLOSE(99)

    ELSE
        DO i = 1, num_agents_sim
            edu_start(i) = get_random_draw(edu_spec%start, edu_spec%share)
        END DO

    END IF

END FUNCTION
!******************************************************************************
!******************************************************************************
FUNCTION get_random_types(num_types, optim_paras, num_agents_sim, edu_start, is_debug) RESULT(types)

    !/* external objects    */

    TYPE(OPTIMPARAS_DICT), INTENT(IN) :: optim_paras

    INTEGER(our_int)                  :: edu_start(num_agents_sim)
    INTEGER(our_int)                  :: types(num_agents_sim)
    INTEGER(our_int)                  :: num_agents_sim
    INTEGER(our_int)                  :: num_types

    LOGICAL, INTENT(IN)               :: is_debug

    !/* internal objects    */

    INTEGER(our_int)                  :: candidates(num_types)
    INTEGER(our_int)                  :: i

    REAL(our_dble)                    :: probs(num_types)

    LOGICAL                           :: READ_IN

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
        candidates = (/ (i, i = 0, num_types - 1) /)
        DO i = 1, num_agents_sim
            probs = get_conditional_probabilities(optim_paras%type_shares, edu_start(i))

            types(i) = get_random_draw(candidates, probs)
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
END MODULE
