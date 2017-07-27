# coding=utf-8
# pylint: disable=invalid-name
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Repeats an interestingness test a given number of times.
If "RANGENUM" is present, it is replaced in turn with each number in the range.

Use for:

1. Intermittent testcases.

   Repeating the test can make the bug occur often enough for Lithium to make progress.

    lithium.py range 1 20 crashes --timeout=3 ./js-dbg-32-mozilla-central-linux -m -n intermittent.js

2. Unstable testcases.

   Varying a number in the test (using RANGENUM) may allow other parts of the testcase to be
   removed (Lithium), or may allow different versions of the shell to crash (autoBisect).

   In the testcase:
     schedulegc(n);

   On the command line:
     lithium.py range 1 50 crashes --timeout=3 ./js-dbg-32-mozilla-central-linux -e "n=RANGENUM;" 740654.js
"""

from __future__ import print_function

from optparse import OptionParser  # pylint: disable=deprecated-module

from ..utils import ximport  # noqa  pylint: disable=relative-import,wrong-import-position


def parseOptions(arguments):  # pylint: disable=missing-docstring
    parser = OptionParser()
    parser.disable_interspersed_args()
    _options, args = parser.parse_args(arguments)

    return int(args[0]), int(args[1]), args[2:]  # args[0] is minLoopNum, args[1] maxLoopNum


class Range(object):

    def __init__(self, interesting_script=False):
        if interesting_script:
            global interesting  # pylint: disable=global-variable-undefined, invalid-name
            interesting = self.interesting

    def interesting(self, cliArgs, tempPrefix):  # pylint: disable=missing-docstring
        (rangeMin, rangeMax, arguments) = parseOptions(cliArgs)
        conditionScript = ximport(arguments[0])
        conditionArgs = arguments[1:]

        if hasattr(conditionScript, "init"):
            conditionScript.init(conditionArgs)

        assert (rangeMax - rangeMin) >= 0
        for i in range(rangeMin, rangeMax + 1):
            # This doesn't do anything if RANGENUM is not found.
            replacedConditionArgs = [s.replace("RANGENUM", str(i)) for s in conditionArgs]
            print("Range number %d:" % i, end=" ")
            if conditionScript.interesting(replacedConditionArgs, tempPrefix):
                return True

        return False


Range(True)
