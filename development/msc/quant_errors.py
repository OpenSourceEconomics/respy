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
floor(values)

#-------------------------------------------------------------------------------
# Potential Bug: Avoid nested loop joins
#-------------------------------------------------------------------------------
# search the corresponding organization for each person and print it
for person in persons:
  for organization in organizations:
    if person.organization_id == organization.id:
       print "{person} belongs to {organization}".format({
           "person": person.name,
           "organization": organization.name
         })

# search if the user_input dict somewhere contains
# a magic value and replace all those magic values.
for key in search_space.keys():
  for value in magic_values:
    if user_input[key] == value:
      user_input[key] = interpret_magic_value(user_input[key])

#-------------------------------------------------------------------------------
# Recommendation: Comma-separated imports
#-------------------------------------------------------------------------------
from zope.component import getMultiAdapter, getSiteManager

