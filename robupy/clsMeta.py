""" Module for the meta class.
"""

# standard library
import pickle as pkl


class MetaCls(object):
    def __init__(self):
        pass

    ''' Meta methods.
    '''

    def get_status(self):
        """ Get status of class instance.
        """

        return self.is_locked

    def lock(self):
        """ Lock class instance.
        """
        # Antibugging.
        assert (self.get_status() is False)

        # Update class attributes.
        self.is_locked = True

        # Finalize.
        self._derived_attributes()

        self._check_integrity()

    def unlock(self):
        """ Unlock class instance.
        """
        # Antibugging.
        assert (self.get_status() is True)

        # Update class attributes.
        self.is_locked = False

    def get_attr(self, key):
        """ Get attributes.
        """
        # Antibugging.
        assert (self.get_status() is True)
        assert (self._check_key(key) is True)

        # Finishing.
        return self.attr[key]

    def set_attr(self, key, value):
        """ Get attributes.
        """
        # Antibugging.
        assert (self.get_status() is False)
        assert (self._check_key(key) is True)

        # Finishing.
        self.attr[key] = value

    def store(self, file_name):
        """ Store class instance.
        """
        # Antibugging.
        assert (self.get_status() is True)
        assert (isinstance(file_name, str))

        # Store.
        pkl.dump(self, open(file_name, 'wb'))

    def _check_key(self, key):
        """ Check that key is present.
        """
        # Check presence.
        assert (key in self.attr.keys())

        # Finishing.
        return True

    def _derived_attributes(self):
        """ Calculate derived attributes.
        """

        pass

    def _check_integrity(self):
        """ Check integrity of class instance.
        """

        pass
