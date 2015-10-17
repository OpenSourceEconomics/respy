""" This script is used to generate some messages on QUANTIFIEDCODE 
to include an interesting dashboard in the presentation.
"""
#-------------------------------------------------------------------------------
# Minor Issue: Avoid untyped exception handlers
#-------------------------------------------------------------------------------
def divide(a, b):
    try:
      result = a / b
    except:
      result = None

    return result

#-------------------------------------------------------------------------------
# Critical: Import naming collision
#-------------------------------------------------------------------------------
from numpy import floor
from numpy import array
from math import floor # Overwrites already imported floor function

values = array([2.3, 8.7, 9.1])
  
#-------------------------------------------------------------------------------
# Critical: Avoid concatenating different built-in types
#-------------------------------------------------------------------------------
i = 12
err_msg= "Error Index: "

print(err_msg + i)
#-------------------------------------------------------------------------------
# Recommendation: Comma-separated imports
#-------------------------------------------------------------------------------
from multiprocessing import Array, Pool

#-------------------------------------------------------------------------------
# Critical: Upgrade from `md5` to `hashlib`
#-------------------------------------------------------------------------------
import md5
md5_hash = md5.new()
md5_hash.update("This module is deprecated")
md5_hash.digest()
