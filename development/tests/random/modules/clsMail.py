''' This module holds the class for mailing capabilities. A file with the
    log in credentials is required in its directory.


    Example Usage:


        mailObj = mailCls()

        mailObj.setAttr('subject', ' Example ')

        mailObj.setAttr('message', ' This is an example message. ')

        mailObj.setAttr('attachment', 'logging.test.txt')

        mailObj.lock()

        mailObj.send()

'''

# standard library
from email.mime.multipart   import MIMEMultipart
from email.mime.text        import MIMEText

import pickle as pkl

import smtplib
import socket
import copy
import json
import os


''' Private class.
'''
class _meta(object):

    def __init__(self):

        pass

    ''' Meta methods.
    '''
    def getStatus(self):
        ''' Get status of class instance.
        '''

        return self.isLocked

    def lock(self):
        ''' Lock class instance.
        '''
        # Antibugging.
        assert (self.getStatus() == False)

        # Update class attributes.
        self.isLocked = True

        # Finalize.
        self._derivedAttributes()

        self._checkIntegrity()

    def unlock(self):
        ''' Unlock class instance.
        '''
        # Antibugging.
        assert (self.getStatus() == True)

        # Update class attributes.
        self.isLocked = False

    def getAttr(self, key, deep = False):
        ''' Get attributes.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (deep in [True, False])

        # Copy requested object.
        if(deep):

            attr = copy.deepcopy(self.attr[key])

        else:

            attr = self.attr[key]

        # Finishing.
        return attr

    def setAttr(self, key, value, deep = False):
        ''' Get attributes.
        '''
        # Antibugging.
        assert (self.getStatus() == False)
        assert (key in self.attr.keys())

        # Copy requested object.
        if(deep):

            attr = copy.deepcopy(value)

        else:

            attr = value

        # Finishing.
        self.attr[key] = attr

    def _derivedAttributes(self):
        ''' Calculate derived attributes.
        '''

        pass

    def _checkIntegrity(self):
        ''' Check integrity of class instance.
        '''

        pass

    def store(self, fileName):
        ''' Store class instance.
        '''
        # Antibugging.
        assert (self.getStatus() == True)
        assert (isinstance(fileName, str))

        # Store.
        pkl.dump(self, open(fileName, 'wb'))

''' Main Class.
'''
class mailCls(_meta):

    def __init__(self):

        self.attr = {}

        # Constitutive attributes.
        self.attr['subject'] = None

        self.attr['message'] = None

        self.attr['attachment'] = None

        # Setup.
        self.attr['sender']    = socket.gethostname()

        self.attr['recipient'] = 'eisenhauer@policy-lab.org'

        # Derived attributes.
        self.attr['username'] = None

        self.attr['password'] = None

        # Status indicator.
        self.isLocked = False

    ''' Public methods.
    '''
    def send(self):
        ''' Send message.
        '''
        # Antibugging.
        assert (self.getStatus() == True)

        # Distribute class attributes.
        subject    = self.attr['subject']

        message    = self.attr['message']

        sender     = self.attr['sender']

        recipient  = self.attr['recipient']

        username   = self.attr['username']

        password   = self.attr['password']

        attachment = self.attr['attachment']

        # Connect to Gmail.
        server = smtplib.SMTP('smtp.gmail.com:587')

        server.starttls()

        server.login(username, password)

        # Formatting.
        msg = MIMEMultipart('alternative')

        msg['Subject'], msg['From'] = subject, sender

        # Attachment.
        f = open(attachment, 'r')

        attached = MIMEText(f.read())

        attached.add_header('Content-Disposition', 'attachment', filename = attachment)

        msg.attach(attached)

        # Message.
        message = MIMEText(message, 'plain')

        msg.attach(message)

        # Send.
        server.sendmail(sender, recipient, msg.as_string())

        # Disconnect.
        server.quit()

    ''' Private methods.
    '''
    def _derivedAttributes(self):
        ''' Construct derived attributes.
        '''
        # Antibugging
        assert (self.getStatus() == True)

        # Check availability.
        assert (self.attr['message'] is not None)

        # Process credentials.
        dict_ = json.load(open(os.environ['HOME'] + '/.credentials'))

        self.attr['username'] = dict_['username']

        self.attr['password'] = dict_['password']
