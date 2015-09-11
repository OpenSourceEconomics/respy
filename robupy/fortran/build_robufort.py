""" This module contains all functions required to build the ROBUFORT
executable. Tedious modifications of the original source code is required to
get the maximum performance. The main component is an iterative inlining of
all functionality from the ROBUFORT library. This allows to maintain a
well-organized code without any loss of performance.
"""

# standard library
import shlex
import shutil
import copy
import os

# module-wide variables
DEBUG_OPTIONS = ' -O2 -fimplicit-none  -Wall  -Wline-truncation ' \
                ' -Wcharacter-truncation  -Wsurprising  -Waliasing' \
                ' -Wimplicit-interface  -Wunused-parameter  -fwhole-file ' \
                ' -fcheck=all  -std=f2008  -pedantic  -fbacktrace'

PRODUCTION_OPTIONS = '-O3'


def robufort_build(self, is_debug=False):
    """ Building the ROBUFORT executable for high speed execution.
    """
    # Compilation of executable for fastest performance
    current_directory = os.getcwd()

    path = self.env.project_paths['ROBUPY']

    os.chdir(path + '/fortran')

    # Performance considerations require an automatic inlining of the core
    # subroutines in a single file. Prepare starting version of extended
    # ROBUFORT code. At first the module containing the program constants is
    # inserted as well as the auxiliary functions. Then the code tailored for
    # the inlinings is added.
    with open('robufort_extended.f90', 'w') as outfile:
        for fname in ['robufort_program_constants.f90', 'robufort_auxiliary.f90',
                      'robufort.f90']:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
            outfile.write('\n')

    # This loop iteratively marks subroutines for inlinings and then replaces
    # them with the relevant code lines. The loop stops once no subroutines
    # are marked for further inlining. The original code of the subroutines
    # is read in directly from the robufort_core.f90 file.
    subroutines = _read_subroutines()

    while True:

        _mark_inlinings(subroutines)

        count = _replace_inlinings(subroutines)

        # Check for further applicability and cleaning.
        if count == 0:
            os.remove('.robufort_inlining.f90')
            break

    # The compiler's vectorization efforts fail due to the case distinction
    # in the get_future_payoffs function. Reading in the fully inlined code for
    # the backward induction routine. Then modify it by splitting the loop
    # over all periods into the last period and all other periods. Remove the
    # IF-CLAUSE in the get_future_payoffs that distinguished between the last
    # and all other periods. Then we write the code back into the extended
    # ROBUFORT program, and again inline the get_future_payoff subroutine.
    code_lines = _read_backward_induction_code()

    code_lines = _split_backward_induction_loops(code_lines)

    # The replacement of the code lines for the last period is
    # straightforward. For all other periods, an inlining marker is placed
    # in the file, which is later replaced with a new insertion of the
    # get_future_payoffs subroutine when calling the replace_inlinings
    # function a couple of lines down.
    code_lines = _replace_vectorization_obstacle(code_lines)

    _write_to_main(code_lines)

    _replace_inlinings(subroutines)

    os.remove('.robufort_inlining.f90')

    # Compile the executable
    if is_debug:
        compiler_options = DEBUG_OPTIONS
    else:
        compiler_options = PRODUCTION_OPTIONS

    os.system('gfortran ' + compiler_options + ' -o robufort ' \
                'robufort_extended.f90')

    shutil.move('robufort', 'bin/robufort')

    os.unlink('robufort_extended.f90')

    os.unlink('robufort_program_constants.mod')

    os.chdir(current_directory)


def _read_backward_induction_code():
    """ This function reads in the fully inlined backward induction procedure.
    """

    # Determine number of lines
    with open('robufort_extended.f90', 'r') as old_file:
        num_lines = len(old_file.readlines())

    # Extract information
    code_lines = dict()
    code_lines['original'] = []

    # Read in information from fully extended ROBUFORT codes.
    with open('robufort_extended.f90') as file_:

        for _ in range(num_lines):

            list_ = shlex.split(file_.readline())

            # Skip all lines that cannot be relevant as the code marker
            # results in a list with at least four elements.
            if len(list_) < 4:
                continue

            # The WHILE loop iterates over all lines of the replacement for
            # the backward induction routine.
            is_start = (list_[3] == 'backward_induction_replacement_start')

            if is_start:
                while True:
                    code_line = file_.readline()
                    list_ = shlex.split(code_line)

                    # Check for end of inlined code
                    try:
                        is_end = (list_[3] ==
                                      'backward_induction_replacement_end')
                    except IndexError:
                        is_end = False

                    if is_end:
                        break

                    # Collect information
                    code_lines['original'] += [code_line]

        # Finishing
        return code_lines


def _split_backward_induction_loops(code_lines):
    """ Split the backward induction procedure into the last and all
    other periods. This is very sensitive to the formatting of of the
    extracted for the backward induction routine
    """
    # Initialize containers
    for name in ['final_period', 'other_periods']:
        code_lines[name] = []

    # Iterate over original code lines
    for element in code_lines['original']:

        # Put code into the containers for the two parts of the loop
        for name in ['final_period', 'other_periods']:
            code_lines[name] += [element]

        # Replace loop counts
        if ('DO' in element) and ('num_periods' in element):
            code_lines['final_period'][-1] = \
                    'DO period = (num_periods - 1), (num_periods - 1), -1'
            code_lines['other_periods'][-1] = \
                    'DO period = (num_periods - 2), 0, -1'

        # Make sure that the initialization of the outcome variables is not
        # redone when moving from the last to previous periods.
        if ('periods_emax' in element) and ('missing_dble' in element):
            code_lines['other_periods'][-1] = ''
        if ('periods_future_payoffs' in element) and ('missing_dble' in element):
            code_lines['other_periods'][-1] = ''
        if ('periods_payoffs_ex_post' in element) and ('missing_dble' in element):
            code_lines['other_periods'][-1] = ''

    # Finishing
    return code_lines


def _replace_vectorization_obstacle(code_lines):
    """ The vectorization efforts by the compiler break down due to the case
    distinction in the get_future_payoffs subroutine.
    """
    # Modify the two sets of relevant codes
    for which in ['final_period', 'other_periods']:

        # Initialize indicator for vectorization block
        is_vectorization = False

        # Independent copy of the baseline code for the backward induction
        # procedure
        periods_copy = copy.deepcopy(code_lines[which])

        for i, element in enumerate(periods_copy):

            # As the default, the same line will just be inserted
            code_lines[which][i] = element

            # Check for vectorization block
            start_vectorization = ('! BEGIN VECTORIZATION A' in element)
            end_vectorization = ('! END VECTORIZATION A' in element)

            if start_vectorization:
                is_vectorization = True

            if end_vectorization:
                is_vectorization = False

            # Special case of vectorization block
            if is_vectorization:

                # The default is to just write out empty lines inside the
                # vectorization block
                code_lines[which][i] = ''

                # Right at the beginning, the required changes are undertaken
                # or at least prepared.
                if start_vectorization and which == 'final_period':
                    code_lines[which][i] = 'future_payoffs = zero_dble'
                if start_vectorization and which == 'other_periods':
                    # This write a mark to the file which will then again be
                    # replaced by all the relevant code when calling the
                    # replace_inlinings() function.
                    code_lines[which][i] = 'INLINING: get_future_payoffs \n'

    # Finishing
    return code_lines


def _write_to_main(code_lines):
    """ Replace the initial inlining of the backward induction procedure with
    the two independent code blocks.
    """
    # Insert second backward induction
    with open('robufort_extended.f90', 'r') as file_:
        old_file = file_.readlines()

    # Indicator for baseline inlining block
    is_previous = False

    # Note that I write to inlining again
    with open('.robufort_inlining.f90', 'w') as new_file:

        for i in range(len(old_file)):

            old_line = old_file[i]

            # Check if replicated backward induction procedure needs to be
            # inserted.
            old_list = shlex.split(old_line)
            try:
                is_start = (old_list[3] =='backward_induction_replacement_start')
                is_end = (old_list[3] =='backward_induction_replacement_end')

            except IndexError:
                is_start = False
                is_end = False

            if is_start:
                is_previous = True

            if is_end:
                is_previous = False

            # Write out all codes that is not related to the backward
            # induction procedure right out to the file.
            if not is_previous:
                new_file.write(old_line)

            # At the start of the original backward induction block,
            # write out the separate set of codes
            if not is_start:
                continue

            for which in ['final_period', 'other_periods']:
                for line_ in code_lines[which]:
                    new_file.write(line_)


def _mark_inlinings(subroutines):
    """ This function marks the subroutines for which inlining
    information is available.
    """
    # Auxiliary objects
    inlining_routines = subroutines.keys()

    # Read file with pre-inlining code
    with open('robufort_extended.f90', 'r') as old_file:
        num_lines = len(old_file.readlines())

    # Initialize logical variables
    is_program = False

    with open('robufort_extended.f90', 'r') as old_file:
        with open('.robufort_inlining.f90', 'w') as new_file:

            for _ in range(num_lines):

                # Extract old information
                old_line = old_file.readline()
                old_list = shlex.split(old_line)

                # Skip all empty lines
                if not old_list:
                    new_file.write(old_line)
                    continue

                # Skip modifying all lines before actual program begins.
                # This skips over the module where the program constants
                # are defined.
                if not is_program:
                    is_program = (old_list[0] == 'PROGRAM')
                    new_file.write(old_line)
                    continue

                # Skip modifying of all lines without a CALL statement
                is_call = (old_list[0] == 'CALL')
                if not is_call:
                    new_file.write(old_line)
                    continue

                # Determine name is call that will be replaced. Note that
                # not all functions or routines will be replaced.
                name = old_list[1].split('(')[0]
                if name not in inlining_routines:
                    new_file.write(old_line)
                    continue

                # Write out keyword for future replacement. Ensure that
                # interfaces that run across multiple lines are removed
                # completely.
                new_file.write('INLINING: ' + name + ' \n')
                while True:
                    is_end = ')' in old_list[-1]
                    if is_end:
                        break
                    old_list = shlex.split(old_file.readline())


def _replace_inlinings(subroutines):
    """ This function replaces subroutines marked for inlining with the
    relevant code.
    """
    # Auxiliary objects
    count = 0

    # Read file with inlining instructions
    with open('.robufort_inlining.f90', 'r') as old_file:
        old_lines = old_file.readlines()

    # Construct new FORTRAN file.
    with open('robufort_extended.f90', 'w') as new_file:
        for old_line in old_lines:

            # Check for subroutines marked for replacement
            is_inlining = 'INLINING' in old_line

            if not is_inlining:
                new_file.write(old_line)
            else:
                # Write out code of relevant subroutine and mark the
                # beginning and end of the replacement.
                name = shlex.split(old_line)[1]
                new_file.write('\n\t! BEGIN REPLACEMENT: ' + name +
                               '_replacement_start\n')
                for code_line in subroutines[name]:
                    new_file.write(code_line)
                new_file.write('\n\t! END REPLACEMENT: ' + name +
                               '_replacement_end\n')
                # Store workload
                count += 1

    # Finishing
    return count


def _read_subroutines():
    """ Read information on all subroutines which are candidates for
    inlining.
    """
    # Initialize container
    subroutines = dict()

    # Auxiliary containers
    name = None

    # Determine number of lines
    with open('robufort_core.f90', 'r') as old_file:
        num_lines = len(old_file.readlines())

    # Extract information
    with open('robufort_core.f90') as file_:

        for _ in range(num_lines):

            list_ = shlex.split(file_.readline())

            # Skip all empty lines
            if not list_:
                continue

            # Initialize container for new subroutine
            new_subroutine = (list_[0] == 'SUBROUTINE')
            if new_subroutine:
                name = list_[1].split('(')[0]
                subroutines[name] = []

            # Collect algorithm information.
            is_algorithm = ('Algorithm' in list_)

            # The WHILE loop iterates over all lines of the file until
            # the subroutine ends.
            if is_algorithm:
                while True:
                    code_line = file_.readline()
                    list_ = shlex.split(code_line)

                    is_end = False
                    try:
                        is_end = list_[:2] == ['END', 'SUBROUTINE']
                    except IndexError:
                        pass

                    if is_end:
                        break

                    # Collect information
                    subroutines[name] += [code_line]

    # Finishing
    return subroutines

