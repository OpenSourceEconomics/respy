#!/usr/bin/env python
""" Script to cleanup after a testing runs.
"""

from modules.auxiliary import cleanup_testing_infrastructure

cleanup_testing_infrastructure(False)
