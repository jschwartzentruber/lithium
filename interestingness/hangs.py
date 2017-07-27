# coding=utf-8
# pylint: disable=invalid-name,missing-docstring
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import print_function

from . import timedRun


class Hangs(object):

    def __init__(self, interesting_script=False):
        if interesting_script:
            global interesting  # pylint: disable=global-variable-undefined, invalid-name
            interesting = self.interesting

    def interesting(self, args, tempPrefix):
        timeout = int(args[0])

        runinfo = timedRun.timed_run(args[1:], timeout, tempPrefix)

        if runinfo.sta == timedRun.TIMED_OUT:
            return True

        print("Exited in %.3f seconds" % runinfo.elapsedtime)
        return False


Hangs(True)
