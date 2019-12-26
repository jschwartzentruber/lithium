#!/usr/bin/env python
# coding=utf-8
# pylint: disable=missing-docstring
# pylint: disable=too-many-lines
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import argparse
import logging
import os
import multiprocessing
import platform
import re
import sys
import tempfile
import threading
import time

from .interestingness.utils import rel_or_abs_import

if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue

log = logging.getLogger("lithium")  # pylint: disable=invalid-name


class LithiumError(Exception):  # pylint: disable=missing-docstring
    pass


class Testcase(object):
    """Abstract testcase class.

    Implementers should define readTestcaseLine() and writeTestcase() methods.
    """

    def __init__(self):
        self.before = b""
        self.after = b""
        self.parts = []

        self.filename = None
        self.extension = None

    def __len__(self):  # pylint: disable=missing-docstring,missing-return-doc,missing-return-type-doc
        return len(self.parts)

    def copy(self):  # pylint: disable=missing-docstring,missing-return-doc,missing-return-type-doc
        new = type(self)()

        new.before = self.before
        new.after = self.after
        new.parts = self.parts[:]

        new.filename = self.filename
        new.extension = self.extension

        return new

    def readTestcase(self, filename):  # pylint: disable=invalid-name,missing-docstring
        hasDDSection = False  # pylint: disable=invalid-name

        self.__init__()
        self.filename = filename
        self.extension = os.path.splitext(self.filename)[1]

        with open(self.filename, "rb") as f:
            # Determine whether the f has a DDBEGIN..DDEND section.
            for line in f:
                if line.find(b"DDEND") != -1:
                    raise LithiumError("The testcase (%s) has a line containing 'DDEND' "
                                       "without a line containing 'DDBEGIN' before it." % self.filename)
                if line.find(b"DDBEGIN") != -1:
                    hasDDSection = True  # pylint: disable=invalid-name
                    break

            f.seek(0)

            if hasDDSection:
                # Reduce only the part of the file between 'DDBEGIN' and 'DDEND',
                # leaving the rest unchanged.
                # log.info("Testcase has a DD section")
                self.readTestcaseWithDDSection(f)
            else:
                # Reduce the entire file.
                # log.info("Testcase does not have a DD section")
                for line in f:
                    self.readTestcaseLine(line)

    def readTestcaseWithDDSection(self, f):  # pylint: disable=invalid-name,missing-docstring
        for line in f:
            self.before += line
            if line.find(b"DDBEGIN") != -1:
                break

        for line in f:
            if line.find(b"DDEND") != -1:
                self.after += line
                break
            self.readTestcaseLine(line)
        else:
            raise LithiumError("The testcase (%s) has a line containing 'DDBEGIN' but no line "
                               "containing 'DDEND'." % self.filename)

        for line in f:
            self.after += line

    def readTestcaseLine(self, line):  # pylint: disable=invalid-name,missing-docstring
        raise NotImplementedError()

    def writeTestcase(self, filename=None):  # pylint: disable=invalid-name,missing-docstring
        raise NotImplementedError()


class TestcaseLine(Testcase):  # pylint: disable=missing-docstring
    atom = "line"

    def readTestcaseLine(self, line):
        self.parts.append(line)

    def writeTestcase(self, filename=None):
        if filename is None:
            filename = self.filename
        with open(filename, "wb") as f:
            f.write(self.before)
            f.writelines(self.parts)
            f.write(self.after)


class TestcaseChar(TestcaseLine):  # pylint: disable=missing-docstring
    atom = "char"

    def readTestcaseWithDDSection(self, f):
        Testcase.readTestcaseWithDDSection(self, f)

        if self.parts:
            # Move the line break at the end of the last line out of the reducible
            # part so the "DDEND" line doesn't get combined with another line.
            self.parts.pop()
            self.after = b"\n" + self.after

    def readTestcaseLine(self, line):
        for i in range(len(line)):
            self.parts.append(line[i:i + 1])


class TestcaseJsStr(TestcaseChar):
    """Testcase type for splitting JS strings byte-wise.

    Data between JS string contents (including the string quotes themselves!) will be a single token for reduction.

    Escapes are also kept together and treated as a single token for reduction.
    ref: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String#Escape_notation
    """
    atom = "jsstr char"

    def readTestcaseWithDDSection(self, f):
        Testcase.readTestcaseWithDDSection(self, f)

    def readTestcase(self, filename):
        # these are temporary attributes used to track state in readTestcaseLine (called by super().readTestcase)
        # they are both deleted after the call below and not available in the instance normally
        self._instr = None  # pylint: disable=attribute-defined-outside-init
        self._chars = []  # pylint: disable=attribute-defined-outside-init

        super(TestcaseJsStr, self).readTestcase(filename)

        # if we hit EOF while looking for end of string, we need to rewind to the state before we matched on
        # that quote character and try again.
        while self._instr is not None:
            idx = None
            for idx in reversed(range(len(self.parts))):  # pylint: disable=range-builtin-not-iterating
                if self.parts[idx].endswith(self._instr) and idx not in self._chars:
                    break
            else:
                raise RuntimeError("error while backtracking from unmatched %s" % (self._instr,))
            self.parts, rest = self.parts[:idx+1], b"".join(self.parts[idx+1:])
            self._chars = [c for c in self._chars if c < idx]  # pylint: disable=attribute-defined-outside-init
            self._instr = None  # pylint: disable=attribute-defined-outside-init
            self.readTestcaseLine(rest)
        del self._instr

        # self._chars is a list of all the indices in self.parts which are chars
        # merge all the non-chars since this was parsed line-wise

        chars = self._chars
        del self._chars

        # beginning and end are special because we can put them in self.before/self.after
        if chars:
            # merge everything before first char (pre chars[0]) into self.before
            offset = chars[0]
            if offset:
                header, self.parts = b"".join(self.parts[:offset]), self.parts[offset:]
                self.before = self.before + header
                # update chars which is a list of offsets into self.parts
                chars = [c - offset for c in chars]

            # merge everything after last char (post chars[-1]) into self.after
            offset = chars[-1] + 1
            if offset < len(self.parts):
                self.parts, footer = self.parts[:offset], b"".join(self.parts[offset:])
                self.after = footer + self.after

        # now scan for chars with a gap > 2 between, which means we can merge
        # the goal is to take a string like this:
        #   parts = [a x x x b c]
        #   chars = [0       4 5]
        # and merge it into this:
        #   parts = [a xxx b c]
        #   chars = [0     2 3]
        for i in range(len(chars) - 1):
            char1, char2 = chars[i], chars[i + 1]
            if (char2 - char1) > 2:
                self.parts[char1 + 1:char2] = [b"".join(self.parts[char1 + 1:char2])]
                offset = char2 - char1 - 2  # num of parts we eliminated
                chars[i + 1:] = [c - offset for c in chars[i + 1:]]

    def readTestcaseLine(self, line):
        last = 0
        while True:
            if self._instr:
                match = re.match(br"(\\u[0-9A-Fa-f]{4}|\\x[0-9A-Fa-f]{2}|\\u\{[0-9A-Fa-f]+\}|\\.|.)", line[last:],
                                 re.DOTALL)
                if not match:
                    break
                self._chars.append(len(self.parts))
                if match.group(0) == self._instr:
                    self._instr = None  # pylint: disable=attribute-defined-outside-init
                    self._chars.pop()
            else:
                match = re.search(br"""['"]""", line[last:])
                if not match:
                    break
                self._instr = match.group(0)  # pylint: disable=attribute-defined-outside-init
            self.parts.append(line[last:last + match.end(0)])
            last += match.end(0)
        if last != len(line):
            self.parts.append(line[last:])


class TestcaseSymbol(TestcaseLine):  # pylint: disable=missing-docstring
    atom = "symbol-delimiter"
    DEFAULT_CUT_AFTER = b"?=;{["
    DEFAULT_CUT_BEFORE = b"]}:"

    def __init__(self):
        TestcaseLine.__init__(self)

        self.cutAfter = self.DEFAULT_CUT_AFTER  # pylint: disable=invalid-name
        self.cutBefore = self.DEFAULT_CUT_BEFORE  # pylint: disable=invalid-name

    def readTestcaseLine(self, line):
        cutter = (b"[" + self.cutBefore + b"]?" +
                  b"[^" + self.cutBefore + self.cutAfter + b"]*" +
                  b"(?:[" + self.cutAfter + b"]|$|(?=[" + self.cutBefore + b"]))")
        for statement in re.finditer(cutter, line):
            if statement.group(0):
                self.parts.append(statement.group(0))


class Strategy(object):
    """Abstract minimization strategy class

    Implementers should define a main() method which takes a testcase and calls the interesting callback repeatedly
    to minimize the testcase.
    """

    def addArgs(self, parser):  # pylint: disable=invalid-name,missing-docstring
        pass

    def processArgs(self, parser, args):  # pylint: disable=invalid-name,missing-docstring
        pass

    def main(self, testcase, interesting, tempFilename):  # pylint: disable=invalid-name,missing-docstring
        raise NotImplementedError()


class CheckOnly(Strategy):  # pylint: disable=missing-docstring
    name = "check-only"

    def main(self, testcase, interesting, tempFilename):  # pylint: disable=missing-return-doc,missing-return-type-doc
        r = interesting(testcase, writeIt=False)  # pylint: disable=invalid-name
        log.info("Lithium result: %s", ("interesting." if r else "not interesting."))
        return int(not r)


class HalfEmpty(Strategy):
    """Parallel minimization strategy based on https://github.com/googleprojectzero/halfempty

    Interestingness scripts using this must be callable in parallel.

    This strategy works using bisection, but it sees the bisection as a decision tree
    where each branch indicates removal/non-removal of a chunk. Since reduction is
    mostly characterized by sequential non-removals, it will execute those
    speculatively in parallel.
    """
    name = "half-empty"
    load_check_period = 120
    thread_check_period = 5

    def __init__(self):
        self.max_workers = None
        self.target_load = None

    def addArgs(self, parser):
        grp_add = parser.add_argument_group(description="Additional options for the %s strategy" % self.name)
        # TODO: most of the Minimize options could apply here too.
        grp_add.add_argument(
            "--target-load",
            type=int,
            help="Target load average to scale workers (0 to disable scaling)",
            default=0 if platform.system == "Windows" else multiprocessing.cpu_count(),
        )
        grp_add.add_argument(
            "--max-workers",
            type=int,
            help="Maximum number of parallel workers to use. (0 to disable limit)",
            default=0,
        )

    def processArgs(self, parser, args):
        if args.max_workers < 0:
            parser.error("--max-workers must be positive")
        if args.target_load < 0:
            parser.error("--target-load must be positive")
        if args.max_workers:
            self.max_workers = args.max_workers
        else:
            self.max_workers = None
        if args.target_load:
            if platform.system() == "Windows":
                log.warning("--target-load only works on unix")
                self.target_load = None
            else:
                self.target_load = args.target_load
        else:
            self.target_load = None
        if self.target_load is None and self.max_workers is None:
            log.warning(
                "Neither --target-load nor --max-workers is specified, "
                "only one worker will be used"
            )
            self.max_workers = 1

    def _monitor(self, exit_evt, clear_evt, work_queue, result_queue, interesting):
        """Launch workers if needed.
        """
        log.info(
            "Starting load monitor. Target load is %r, max workers is %r",
            self.target_load,
            self.max_workers,
        )
        workers = []
        try:
            next_load_check = time.time() + self.load_check_period
            last_worker_add = 0
            while not exit_evt.is_set():
                now = time.time()
                can_add_workers = (
                    (self.max_workers is None or len(workers) < self.max_workers)
                    # use 60 seconds since that's the minimum period for load averages
                    # to register a change
                    and (now - last_worker_add) >= 60
                )
                if self.target_load is None or not workers:
                    need_workers = True
                elif now >= next_load_check:
                    loadavg = os.getloadavg()
                    # check if adding one more worker would exceed our target load
                    if (now - last_worker_add) < 5 * 60:
                        current_load = loadavg[0]  # use 1-minute average
                    elif (now - last_worker_add) < 15 * 60:
                        current_load = loadavg[1]  # use 5-minute average
                    else:
                        current_load = loadavg[2]  # use 15-minute average
                    load_per_worker = current_load / len(workers)
                    next_worker_load = (len(workers) + 1) * load_per_worker
                    need_workers = next_worker_load <= self.target_load
                    next_load_check = time.time() + self.load_check_period
                    log.info("Current load: %r, need workers? %r", loadavg, need_workers)
                else:
                    need_workers = False
                if can_add_workers and need_workers:
                    log.info("Adding a worker...")
                    # add worker
                    workers.append(
                        threading.Thread(
                            target=self._worker,
                            args=(
                                exit_evt,
                                clear_evt,
                                work_queue,
                                result_queue,
                                interesting,
                            ),
                        )
                    )
                    workers[-1].start()
                    last_worker_add = time.time()
                time.sleep(self.thread_check_period)
        except:  # noqa pylint: disable=bare-except
            exit_evt.set()
            raise
        finally:
            log.info("Waiting for workers...")
            for worker in workers:
                worker.join()

    def _worker(self, exit_evt, clear_evt, work_queue, result_queue, interesting):
        while True:
            if exit_evt.is_set() and not clear_evt.is_set():
                # don't break if clear_evt is set, or we could deadlock the main thread
                # the work queue must be cleared first
                break
            try:
                work = work_queue.get(timeout=self.thread_check_period)
            except queue.Empty:
                continue
            if clear_evt.is_set():
                # clear_evt should just drain the work queue.
                work_queue.task_done()
                continue
            if work is None:
                # reduction is finished. pass the sentinal to the result queue
                result_queue.put(None)
                work_queue.task_done()
                continue
            try:
                testcase, chunk_to_try = work
                test_to_try, next_chunk = self._remove_chunk(testcase, chunk_to_try)
                if interesting(test_to_try):
                    log.info("Chunk %r removal was interesting", chunk_to_try)
                    result_queue.put((test_to_try, next_chunk))
                else:
                    result_queue.put((testcase, next_chunk))
            except:  # noqa pylint: disable=bare-except
                exit_evt.set()
                raise
            finally:
                work_queue.task_done()

    @staticmethod
    def _remove_chunk(testcase, chunk_iter):
        """Remove the given chunk from a testcase.

        Args:
            testcase (TestCase): Input testcase
            chunk_iter (object): Current chunk iterator

        Returns:
            (testcase, chunk_iter): Modified testcase (copy) and next value of
                                    chunk_iter for `_iter_chunks()`
        """
        chunk_size, chunk_end = chunk_iter
        chunk_start = max(0, chunk_end - chunk_size)
        result = testcase.copy()
        result.parts = result.parts[:chunk_start] + result.parts[chunk_end:]
        return result, (chunk_size, chunk_start)

    @staticmethod
    def _iter_chunks(testcase, chunk_iter):
        """Return an iteration of chunk_size/chunk_number which are given as args to
        `remove_chunk()` to perform reduction.

        Any removal will invalidate this sequence.

        Args:
            testcase (TestCase): Input testcase
            chunk_iter (object): Current chunk iterator (or None to start)

        Returns:
            generator of chunk_iter objects
        """
        if chunk_iter is None:
            chunk_size = largestPowerOfTwoSmallerThan(len(testcase))
            chunk_end = len(testcase)
        else:
            chunk_size, chunk_end = chunk_iter

        while True:
            if chunk_end - chunk_size < 0:
                # If the testcase is empty, end minimization
                if not testcase.parts:
                    break

                # If the chunk_size is less than or equal to the min_chunk_size and...
                if chunk_size <= 1:
                    break

                # If none of the conditions apply, reduce the chunk_size and continue
                chunk_end = len(testcase)
                while chunk_size > 1:  # smallest valid chunk size is 1
                    chunk_size >>= 1
                    # To avoid testing with an empty testcase (wasting cycles) only break when
                    # chunkSize is less than the number of testcase parts available.
                    if chunk_size < len(testcase):
                        break

            yield (chunk_size, chunk_end)

            # Decrement chunk_size
            # To ensure the file is fully reduced, decrement chunk_end by 1 when chunk_size <= 2
            if chunk_size <= 2:
                chunk_end -= 1
            else:
                chunk_end -= chunk_size

    def _fill_queue(self, work_queue, testcase, current_chunk):
        # fill the queue with work, assuming every reduction will fail
        for chunk in self._iter_chunks(testcase, current_chunk):
            # testcase is a single object, it isn't copied in the queue
            work_queue.put((testcase, chunk))
        # add a sentinal item to signal that reduction is complete
        work_queue.put(None)

    def main(self, testcase, interesting, temp_filename):
        work_queue = queue.Queue()
        result_queue = queue.Queue()
        exit_evt = threading.Event()
        clear_evt = threading.Event()

        monitor = threading.Thread(
            target=self._monitor,
            args=(
                exit_evt,
                clear_evt,
                work_queue,
                result_queue,
                interesting,
            ),
        )
        monitor.start()
        try:
            self._fill_queue(work_queue, testcase, None)
            while True:
                if exit_evt.is_set():
                    break
                try:
                    result = result_queue.get(timeout=self.thread_check_period)
                    if result is None:
                        log.info("reduction finished")
                        # reduction is finished!
                        return 0
                    test_tried, next_chunk = result
                    if test_tried is not testcase:
                        log.info("Removing the chunk and resetting work queue")
                        # wow, it worked!
                        clear_evt.set()
                        work_queue.join()
                        try:
                            result_queue.get_nowait()
                        except queue.Empty:
                            pass
                        # remove the chunk for real
                        testcase = test_tried
                        # re-fill the queue
                        self._fill_queue(work_queue, testcase, next_chunk)
                except queue.Empty:
                    continue
        finally:
            exit_evt.set()
            monitor.join()
        return 1


class Minimize(Strategy):
    """    Main reduction algorithm

    This strategy attempts to remove chunks which might not be interesting
    code, but which can be removed independently of any other.  This happens
    frequently with values which are computed, but either after the execution,
    or never used to influenced the interesting part.

      a = compute();
      b = compute();   <-- !!!
      interesting(a);
      c = compute();   <-- !!!"""

    name = "minimize"

    def __init__(self):
        self.minimizeRepeat = "last"  # pylint: disable=invalid-name
        self.minimizeMin = 1  # pylint: disable=invalid-name
        self.minimizeMax = pow(2, 30)  # pylint: disable=invalid-name
        self.minimizeChunkStart = 0  # pylint: disable=invalid-name
        self.minimizeChunkSize = None  # pylint: disable=invalid-name
        self.minimizeRepeatFirstRound = False  # pylint: disable=invalid-name
        self.stopAfterTime = None  # pylint: disable=invalid-name

    def addArgs(self, parser):
        grp_add = parser.add_argument_group(description="Additional options for the %s strategy" % self.name)
        grp_add.add_argument(
            "--min", type=int,
            default=1,
            help="must be a power of two. default: 1")
        grp_add.add_argument(
            "--max", type=int,
            default=pow(2, 30),
            help="must be a power of two. default: about half of the file")
        grp_add.add_argument(
            "--repeat",
            default="last",
            choices=["always", "last", "never"],
            help="Whether to repeat a chunk size if chunks are removed. default: last")
        grp_add.add_argument(
            "--chunksize", type=int,
            default=None,
            help="Shortcut for repeat=never, min=n, max=n. chunk size must be a power of two.")
        grp_add.add_argument(
            "--chunkstart", type=int,
            default=0,
            help="For the first round only, start n chars/lines into the file. Best for max to divide n. "
                 "[Mostly intended for internal use]")
        grp_add.add_argument(
            "--repeatfirstround", action="store_true",
            help="Treat the first round as if it removed chunks; possibly repeat it. "
                 "[Mostly intended for internal use]")
        grp_add.add_argument(
            "--maxruntime", type=int,
            default=None,
            help="If reduction takes more than n seconds, stop (and print instructions for continuing).")

    def processArgs(self, parser, args):
        if args.chunksize:
            self.minimizeMin = args.chunksize
            self.minimizeMax = args.chunksize
            self.minimizeRepeat = "never"
        else:
            self.minimizeMin = args.min
            self.minimizeMax = args.max
            self.minimizeRepeat = args.repeat
        self.minimizeChunkStart = args.chunkstart
        self.minimizeRepeatFirstRound = args.repeatfirstround
        if args.maxruntime:
            self.stopAfterTime = time.time() + args.maxruntime
        if not isPowerOfTwo(self.minimizeMin) or not isPowerOfTwo(self.minimizeMax):
            parser.error("Min/Max must be powers of two.")

    @staticmethod
    def apply_post_round_op(testcase):  # pylint: disable=unused-argument
        """ Operations to be performed after each round
        Args:
            testcase (Testcase): Testcase to be reduced.
        Returns:
            bool: True if callback was performed successfully, False otherwise.
        """
        return False

    def main(self, testcase, interesting, tempFilename):  # pylint: disable=missing-return-doc,missing-return-type-doc
        log.info("The original testcase has %s.", quantity(len(testcase), testcase.atom))
        log.info("Checking that the original testcase is 'interesting'...")
        if not interesting(testcase, writeIt=False):
            log.info("Lithium result: the original testcase is not 'interesting'!")
            return 1

        if not testcase.parts:
            log.info("The file has %s so there's nothing for Lithium to try to remove!", quantity(0, testcase.atom))

        testcase.writeTestcase(tempFilename("original", False))

        origNumParts = len(testcase)  # pylint: disable=invalid-name
        result, anySingle, testcase = self.run(testcase, interesting, tempFilename)  # pylint: disable=invalid-name

        testcase.writeTestcase()

        summaryHeader()

        if anySingle:
            log.info("  Removing any single %s from the final file makes it uninteresting!", testcase.atom)

        log.info("  Initial size: %s", quantity(origNumParts, testcase.atom))
        log.info("  Final size: %s", quantity(len(testcase), testcase.atom))

        return result

    def run(self, testcase, interesting, tempFilename):  # pylint: disable=invalid-name,missing-docstring
        # pylint: disable=missing-return-doc,missing-return-type-doc
        # pylint: disable=too-many-branches,too-complex,too-many-statements
        chunk_size = min(self.minimizeMax, largestPowerOfTwoSmallerThan(len(testcase)))
        min_chunk_size = min(chunk_size, max(self.minimizeMin, 1))
        chunk_end = len(testcase)
        removed_chunks = self.minimizeRepeatFirstRound

        while True:
            if self.stopAfterTime and time.time() > self.stopAfterTime:
                # Not all switches will be copied!  Be sure to add --tempdir, --maxruntime if desired.
                # Not using shellify() here because of the strange requirements of bot.py's lithium-command.txt.
                log.info("Lithium result: please perform another pass using the same arguments")
                break

            if chunk_end - chunk_size < 0:
                testcase.writeTestcase(tempFilename("did-round-%d" % chunk_size))
                log.info("")

                # If the testcase is empty, end minimization
                if not testcase.parts:
                    log.info("Lithium result: succeeded, reduced to: %s", quantity(len(testcase), testcase.atom))
                    break

                # If the chunk_size is less than or equal to the min_chunk_size and...
                if chunk_size <= min_chunk_size:
                    # Repeat mode is last or always and at least one chunk was removed during the last round, repeat
                    if removed_chunks and (self.minimizeRepeat == "always" or self.minimizeRepeat == "last"):
                        log.info("Starting another round of chunk size %d", chunk_size)
                        chunk_end = len(testcase)
                    # Otherwise, end minimization
                    else:
                        log.info("Lithium result: succeeded, reduced to: %s", quantity(len(testcase), testcase.atom))
                        break
                # If none of the conditions apply, reduce the chunk_size and continue
                else:
                    chunk_end = len(testcase)
                    while chunk_size > 1:  # smallest valid chunk size is 1
                        chunk_size >>= 1
                        # To avoid testing with an empty testcase (wasting cycles) only break when
                        # chunkSize is less than the number of testcase parts available.
                        if chunk_size < len(testcase):
                            break

                    log.info("Reducing chunk size to %d", chunk_size)
                removed_chunks = False

                # Perform post round clean-up if defined
                test_to_try = testcase.copy()
                if self.apply_post_round_op(test_to_try):
                    log.info("Attempting to apply post round operations to testcase")
                    if interesting(test_to_try):
                        log.info("Post round operations were successful")
                        testcase = test_to_try
                    else:
                        log.info("Post round operations made the file uninteresting")

            chunk_start = max(0, chunk_end - chunk_size)
            status = "Removing chunk from %d to %d of %d" % (chunk_start, chunk_end, len(testcase))
            test_to_try = testcase.copy()
            test_to_try.parts = test_to_try.parts[:chunk_start] + test_to_try.parts[chunk_end:]

            if interesting(test_to_try):
                testcase = test_to_try
                log.info("%s was successful", status)
                removed_chunks = True
                chunk_end = chunk_start
            else:
                log.info("%s made the file uninteresting", status)
                # Decrement chunk_size
                # To ensure the file is fully reduced, decrement chunk_end by 1 when chunk_size <= 2
                if chunk_size <= 2:
                    chunk_end -= 1
                else:
                    chunk_end -= chunk_size

        return 0, (chunk_size == 1 and not removed_chunks and self.minimizeRepeat != "never"), testcase


class MinimizeSurroundingPairs(Minimize):
    """    This strategy attempts to remove pairs of chunks which might be surrounding
    interesting code, but which cannot be removed independently of the other.
    This happens frequently with patterns such as:

      a = 42;
      while (true) {
         b = foo(a);      <-- !!!
         interesting();
         a = bar(b);      <-- !!!
      }"""

    name = "minimize-around"

    def run(self, testcase, interesting, tempFilename):  # pylint: disable=missing-return-doc,missing-return-type-doc
        # pylint: disable=invalid-name
        chunkSize = min(self.minimizeMax, largestPowerOfTwoSmallerThan(len(testcase)))
        finalChunkSize = max(self.minimizeMin, 1)  # pylint: disable=invalid-name

        while 1:
            anyChunksRemoved, testcase = self.tryRemovingChunks(chunkSize, testcase, interesting, tempFilename)

            last = (chunkSize <= finalChunkSize)

            if anyChunksRemoved and (self.minimizeRepeat == "always" or (self.minimizeRepeat == "last" and last)):
                # Repeat with the same chunk size
                pass
            elif last:
                # Done
                break
            else:
                # Continue with the next smaller chunk size
                chunkSize >>= 1

        return 0, (finalChunkSize == 1 and self.minimizeRepeat != "never"), testcase

    @staticmethod
    def list_rindex(l, p, e):  # pylint: disable=invalid-name,missing-docstring
        # pylint: disable=missing-return-doc,missing-return-type-doc
        if p < 0 or p > len(l):
            raise ValueError("%s is not in list" % e)
        for index, item in enumerate(reversed(l[:p])):
            if item == e:
                return p - index - 1
        raise ValueError("%s is not in list" % e)

    @staticmethod
    def list_nindex(l, p, e):  # pylint: disable=invalid-name,missing-docstring
        # pylint: disable=missing-return-doc,missing-return-type-doc
        if p + 1 >= len(l):
            raise ValueError("%s is not in list" % e)
        return l[(p + 1):].index(e) + (p + 1)

    def tryRemovingChunks(self, chunkSize, testcase, interesting, tempFilename):  # pylint: disable=invalid-name
        # pylint: disable=missing-param-doc,missing-return-doc,missing-return-type-doc,missing-type-doc
        # pylint: disable=too-many-locals,too-many-statements
        """Make a single run through the testcase, trying to remove chunks of size chunkSize.

        Returns True iff any chunks were removed."""

        summary = ""

        chunksRemoved = 0  # pylint: disable=invalid-name
        atomsRemoved = 0  # pylint: disable=invalid-name

        atomsInitial = len(testcase)  # pylint: disable=invalid-name
        numChunks = divideRoundingUp(len(testcase), chunkSize)  # pylint: disable=invalid-name

        # Not enough chunks to remove surrounding blocks.
        if numChunks < 3:
            return False, testcase

        log.info("Starting a round with chunks of %s.", quantity(chunkSize, testcase.atom))

        summary = ["S" for _ in range(numChunks)]
        chunkStart = chunkSize  # pylint: disable=invalid-name
        beforeChunkIdx = 0  # pylint: disable=invalid-name
        keepChunkIdx = 1  # pylint: disable=invalid-name
        afterChunkIdx = 2  # pylint: disable=invalid-name

        try:
            while chunkStart + chunkSize < len(testcase):
                chunkBefStart = max(0, chunkStart - chunkSize)  # pylint: disable=invalid-name
                chunkBefEnd = chunkStart  # pylint: disable=invalid-name
                chunkAftStart = min(len(testcase), chunkStart + chunkSize)  # pylint: disable=invalid-name
                chunkAftEnd = min(len(testcase), chunkAftStart + chunkSize)  # pylint: disable=invalid-name
                description = "chunk #%d & #%d of %d chunks of size %d" % (
                    beforeChunkIdx, afterChunkIdx, numChunks, chunkSize)

                testcaseSuggestion = testcase.copy()  # pylint: disable=invalid-name
                testcaseSuggestion.parts = (testcaseSuggestion.parts[:chunkBefStart] +
                                            testcaseSuggestion.parts[chunkBefEnd:chunkAftStart] +
                                            testcaseSuggestion.parts[chunkAftEnd:])
                if interesting(testcaseSuggestion):
                    testcase = testcaseSuggestion
                    log.info("Yay, reduced it by removing %s :)", description)
                    chunksRemoved += 2  # pylint: disable=invalid-name
                    atomsRemoved += (chunkBefEnd - chunkBefStart)  # pylint: disable=invalid-name
                    atomsRemoved += (chunkAftEnd - chunkAftStart)  # pylint: disable=invalid-name
                    summary[beforeChunkIdx] = "-"
                    summary[afterChunkIdx] = "-"
                    # The start is now sooner since we remove the chunk which was before this one.
                    chunkStart -= chunkSize  # pylint: disable=invalid-name
                    try:
                        # Try to keep removing surrounding chunks of the same part.
                        beforeChunkIdx = self.list_rindex(summary, keepChunkIdx, "S")  # pylint: disable=invalid-name
                    except ValueError:
                        # There is no more survinving block on the left-hand-side of
                        # the current chunk, shift everything by one surviving
                        # block. Any ValueError from here means that there is no
                        # longer enough chunk.
                        beforeChunkIdx = keepChunkIdx  # pylint: disable=invalid-name
                        keepChunkIdx = self.list_nindex(summary, keepChunkIdx, "S")  # pylint: disable=invalid-name
                        chunkStart += chunkSize  # pylint: disable=invalid-name
                else:
                    log.info("Removing %s made the file 'uninteresting'.", description)
                    # Shift chunk indexes, and seek the next surviving chunk. ValueError
                    # from here means that there is no longer enough chunks.
                    beforeChunkIdx = keepChunkIdx  # pylint: disable=invalid-name
                    keepChunkIdx = afterChunkIdx  # pylint: disable=invalid-name
                    chunkStart += chunkSize  # pylint: disable=invalid-name

                afterChunkIdx = self.list_nindex(summary, keepChunkIdx, "S")  # pylint: disable=invalid-name

        except ValueError:
            # This is a valid loop exit point.
            chunkStart = len(testcase)  # pylint: disable=invalid-name

        atomsSurviving = atomsInitial - atomsRemoved  # pylint: disable=invalid-name
        printableSummary = " ".join(  # pylint: disable=invalid-name
            "".join(summary[(2 * i):min(2 * (i + 1), numChunks + 1)]) for i in range(numChunks // 2 + numChunks % 2))
        log.info("")
        log.info("Done with a round of chunk size %d!", chunkSize)
        log.info("%s survived; %s removed.",
                 quantity(summary.count("S"), "chunk"),
                 quantity(summary.count("-"), "chunk"))
        log.info("%s survived; %s removed.",
                 quantity(atomsSurviving, testcase.atom),
                 quantity(atomsRemoved, testcase.atom))
        log.info("Which chunks survived: %s", printableSummary)
        log.info("")

        testcase.writeTestcase(tempFilename("did-round-%d" % chunkSize))

        return bool(chunksRemoved), testcase


class MinimizeBalancedPairs(MinimizeSurroundingPairs):
    """    This strategy attempts to remove balanced chunks which might be surrounding
    interesting code, but which cannot be removed independently of the other.
    This happens frequently with patterns such as:

      ...;
      if (cond) {        <-- !!!
         ...;
         interesting();
         ...;
      }                  <-- !!!
      ...;

    The value of the condition might not be interesting, but in order to reach the
    interesting code we still have to compute it, and keep extra code alive."""

    name = "minimize-balanced"

    @staticmethod
    def list_fiveParts(lst, step, f, s, t):  # pylint: disable=invalid-name,missing-docstring
        # pylint: disable=missing-return-doc,missing-return-type-doc
        return (lst[:f], lst[f:s], lst[s:(s + step)], lst[(s + step):(t + step)], lst[(t + step):])

    def tryRemovingChunks(self, chunkSize, testcase, interesting, tempFilename):
        # pylint: disable=missing-param-doc,missing-return-doc,missing-return-type-doc,missing-type-doc
        # pylint: disable=too-many-branches,too-complex,too-many-locals,too-many-statements
        """Make a single run through the testcase, trying to remove chunks of size chunkSize.

        Returns True iff any chunks were removed."""

        summary = ""

        chunksRemoved = 0  # pylint: disable=invalid-name
        atomsRemoved = 0  # pylint: disable=invalid-name

        atomsInitial = len(testcase)  # pylint: disable=invalid-name
        numChunks = divideRoundingUp(len(testcase), chunkSize)  # pylint: disable=invalid-name

        # Not enough chunks to remove surrounding blocks.
        if numChunks < 2:
            return False, testcase

        log.info("Starting a round with chunks of %s.", quantity(chunkSize, testcase.atom))

        summary = ["S" for i in range(numChunks)]
        curly = [(testcase.parts[i].count(b"{") - testcase.parts[i].count(b"}")) for i in range(numChunks)]
        square = [(testcase.parts[i].count(b"[") - testcase.parts[i].count(b"]")) for i in range(numChunks)]
        normal = [(testcase.parts[i].count(b"(") - testcase.parts[i].count(b")")) for i in range(numChunks)]
        chunkStart = 0  # pylint: disable=invalid-name
        lhsChunkIdx = 0  # pylint: disable=invalid-name

        try:
            while chunkStart < len(testcase):

                description = "chunk #%d%s of %d chunks of size %d" % (
                    lhsChunkIdx, "".join(" " for i in range(len(str(lhsChunkIdx)) + 4)), numChunks, chunkSize)

                assert summary[:lhsChunkIdx].count("S") * chunkSize == chunkStart, (
                    "the chunkStart should correspond to the lhsChunkIdx modulo the removed chunks.")

                chunkLhsStart = chunkStart  # pylint: disable=invalid-name
                chunkLhsEnd = min(len(testcase), chunkLhsStart + chunkSize)  # pylint: disable=invalid-name

                nCurly = curly[lhsChunkIdx]  # pylint: disable=invalid-name
                nSquare = square[lhsChunkIdx]  # pylint: disable=invalid-name
                nNormal = normal[lhsChunkIdx]  # pylint: disable=invalid-name

                # If the chunk is already balanced, try to remove it.
                if not (nCurly or nSquare or nNormal):
                    testcaseSuggestion = testcase.copy()  # pylint: disable=invalid-name
                    testcaseSuggestion.parts = (testcaseSuggestion.parts[:chunkLhsStart] +
                                                testcaseSuggestion.parts[chunkLhsEnd:])
                    if interesting(testcaseSuggestion):
                        testcase = testcaseSuggestion
                        log.info("Yay, reduced it by removing %s :)", description)
                        chunksRemoved += 1  # pylint: disable=invalid-name
                        atomsRemoved += (chunkLhsEnd - chunkLhsStart)  # pylint: disable=invalid-name
                        summary[lhsChunkIdx] = "-"
                    else:
                        log.info("Removing %s made the file 'uninteresting'.", description)
                        chunkStart += chunkSize  # pylint: disable=invalid-name
                    lhsChunkIdx = self.list_nindex(summary, lhsChunkIdx, "S")  # pylint: disable=invalid-name
                    continue

                # Otherwise look for the corresponding chunk.
                rhsChunkIdx = lhsChunkIdx  # pylint: disable=invalid-name
                for item in summary[(lhsChunkIdx + 1):]:
                    rhsChunkIdx += 1  # pylint: disable=invalid-name
                    if item != "S":
                        continue
                    nCurly += curly[rhsChunkIdx]  # pylint: disable=invalid-name
                    nSquare += square[rhsChunkIdx]  # pylint: disable=invalid-name
                    nNormal += normal[rhsChunkIdx]  # pylint: disable=invalid-name
                    if nCurly < 0 or nSquare < 0 or nNormal < 0:
                        break
                    if not (nCurly or nSquare or nNormal):
                        break

                # If we have no match, then just skip this pair of chunks.
                if nCurly or nSquare or nNormal:
                    log.info("Skipping %s because it is 'uninteresting'.", description)
                    chunkStart += chunkSize  # pylint: disable=invalid-name
                    lhsChunkIdx = self.list_nindex(summary, lhsChunkIdx, "S")  # pylint: disable=invalid-name
                    continue

                # Otherwise we do have a match and we check if this is interesting to remove both.
                # pylint: disable=invalid-name
                chunkRhsStart = chunkLhsStart + chunkSize * summary[lhsChunkIdx:rhsChunkIdx].count("S")
                chunkRhsStart = min(len(testcase), chunkRhsStart)  # pylint: disable=invalid-name
                chunkRhsEnd = min(len(testcase), chunkRhsStart + chunkSize)  # pylint: disable=invalid-name

                description = "chunk #%d & #%d of %d chunks of size %d" % (
                    lhsChunkIdx, rhsChunkIdx, numChunks, chunkSize)

                testcaseSuggestion = testcase.copy()
                testcaseSuggestion.parts = (testcaseSuggestion.parts[:chunkLhsStart] +
                                            testcaseSuggestion.parts[chunkLhsEnd:chunkRhsStart] +
                                            testcaseSuggestion.parts[chunkRhsEnd:])
                if interesting(testcaseSuggestion):
                    testcase = testcaseSuggestion
                    log.info("Yay, reduced it by removing %s :)", description)
                    chunksRemoved += 2
                    atomsRemoved += (chunkLhsEnd - chunkLhsStart)
                    atomsRemoved += (chunkRhsEnd - chunkRhsStart)
                    summary[lhsChunkIdx] = "-"
                    summary[rhsChunkIdx] = "-"
                    lhsChunkIdx = self.list_nindex(summary, lhsChunkIdx, "S")
                    continue

                # Removing the braces make the failure disappear.  As we are looking
                # for removing chunk (braces), we need to make the content within
                # the braces as minimal as possible, so let us try to see if we can
                # move the chunks outside the braces.
                log.info("Removing %s made the file 'uninteresting'.", description)

                # Moving chunks is still a bit experimental, and it can introduce reducing loops.
                # If you want to try it, just replace this True by a False.
                if True:  # pylint: disable=using-constant-test
                    chunkStart += chunkSize
                    lhsChunkIdx = self.list_nindex(summary, lhsChunkIdx, "S")
                    continue

                origChunkIdx = lhsChunkIdx
                stayOnSameChunk = False
                chunkMidStart = chunkLhsEnd
                midChunkIdx = self.list_nindex(summary, lhsChunkIdx, "S")
                while chunkMidStart < chunkRhsStart:
                    assert summary[:midChunkIdx].count("S") * chunkSize == chunkMidStart, (
                        "the chunkMidStart should correspond to the midChunkIdx modulo the removed chunks.")
                    description = "chunk #%d%s of %d chunks of size %d" % (
                        midChunkIdx, "".join(" " for i in range(len(str(lhsChunkIdx)) + 4)), numChunks, chunkSize)

                    p = self.list_fiveParts(testcase.parts, chunkSize, chunkLhsStart, chunkMidStart, chunkRhsStart)

                    nCurly = curly[midChunkIdx]
                    nSquare = square[midChunkIdx]
                    nNormal = normal[midChunkIdx]
                    if nCurly or nSquare or nNormal:
                        log.info("Keeping %s because it is 'uninteresting'.", description)
                        chunkMidStart += chunkSize
                        midChunkIdx = self.list_nindex(summary, midChunkIdx, "S")
                        continue

                    # Try moving the chunk after.
                    testcaseSuggestion = testcase.copy()
                    testcaseSuggestion.parts = p[0] + p[1] + p[3] + p[2] + p[4]
                    if interesting(testcaseSuggestion):
                        testcase = testcaseSuggestion
                        log.info("->Moving %s kept the file 'interesting'.", description)
                        chunkRhsStart -= chunkSize
                        chunkRhsEnd -= chunkSize
                        # pylint: disable=bad-whitespace
                        tS = self.list_fiveParts(summary, 1, lhsChunkIdx, midChunkIdx, rhsChunkIdx)
                        tc = self.list_fiveParts(curly  , 1, lhsChunkIdx, midChunkIdx, rhsChunkIdx)  # noqa
                        ts = self.list_fiveParts(square , 1, lhsChunkIdx, midChunkIdx, rhsChunkIdx)  # noqa
                        tn = self.list_fiveParts(normal , 1, lhsChunkIdx, midChunkIdx, rhsChunkIdx)  # noqa
                        summary = tS[0] + tS[1] + tS[3] + tS[2] + tS[4]
                        curly =   tc[0] + tc[1] + tc[3] + tc[2] + tc[4]  # noqa
                        square =  ts[0] + ts[1] + ts[3] + ts[2] + ts[4]  # noqa
                        normal =  tn[0] + tn[1] + tn[3] + tn[2] + tn[4]  # noqa
                        rhsChunkIdx -= 1
                        midChunkIdx = summary[midChunkIdx:].index("S") + midChunkIdx
                        continue

                    # Try moving the chunk before.
                    testcaseSuggestion.parts = p[0] + p[2] + p[1] + p[3] + p[4]
                    if interesting(testcaseSuggestion):
                        testcase = testcaseSuggestion
                        log.info("<-Moving %s kept the file 'interesting'.", description)
                        chunkLhsStart += chunkSize
                        chunkLhsEnd += chunkSize
                        chunkMidStart += chunkSize
                        # pylint: disable=bad-whitespace
                        tS = self.list_fiveParts(summary, 1, lhsChunkIdx, midChunkIdx, rhsChunkIdx)
                        tc = self.list_fiveParts(curly  , 1, lhsChunkIdx, midChunkIdx, rhsChunkIdx)  # noqa
                        ts = self.list_fiveParts(square , 1, lhsChunkIdx, midChunkIdx, rhsChunkIdx)  # noqa
                        tn = self.list_fiveParts(normal , 1, lhsChunkIdx, midChunkIdx, rhsChunkIdx)  # noqa
                        summary = tS[0] + tS[2] + tS[1] + tS[3] + tS[4]
                        curly =   tc[0] + tc[2] + tc[1] + tc[3] + tc[4]  # noqa
                        square =  ts[0] + ts[2] + ts[1] + ts[3] + ts[4]  # noqa
                        normal =  tn[0] + tn[2] + tn[1] + tn[3] + tn[4]  # noqa
                        lhsChunkIdx += 1
                        midChunkIdx = self.list_nindex(summary, midChunkIdx, "S")
                        stayOnSameChunk = True
                        continue

                    log.info("..Moving %s made the file 'uninteresting'.", description)
                    chunkMidStart += chunkSize
                    midChunkIdx = self.list_nindex(summary, midChunkIdx, "S")

                lhsChunkIdx = origChunkIdx
                if not stayOnSameChunk:
                    chunkStart += chunkSize
                    lhsChunkIdx = self.list_nindex(summary, lhsChunkIdx, "S")

        except ValueError:
            # This is a valid loop exit point.
            chunkStart = len(testcase)  # pylint: disable=invalid-name

        atomsSurviving = atomsInitial - atomsRemoved  # pylint: disable=invalid-name
        printableSummary = " ".join(  # pylint: disable=invalid-name
            "".join(summary[(2 * i):min(2 * (i + 1), numChunks + 1)]) for i in range(numChunks // 2 + numChunks % 2))
        log.info("")
        log.info("Done with a round of chunk size %d!", chunkSize)
        log.info("%s survived; %s removed.",
                 quantity(summary.count("S"), "chunk"),
                 quantity(summary.count("-"), "chunk"))
        log.info("%s survived; %s removed.",
                 quantity(atomsSurviving, testcase.atom),
                 quantity(atomsRemoved, testcase.atom))
        log.info("Which chunks survived: %s", printableSummary)
        log.info("")

        testcase.writeTestcase(tempFilename("did-round-%d" % chunkSize))

        return bool(chunksRemoved), testcase


class ReplacePropertiesByGlobals(Minimize):
    """    This strategy attempts to remove members, such that other strategies can
    then move the lines outside the functions.  The goal is to rename
    variables at the same time, such that the program remains valid, while
    removing the dependency on the object on which the member is part of.

      function Foo() {
        this.list = [];
      }
      Foo.prototype.push = function(a) {
        this.list.push(a);
      }
      Foo.prototype.last = function() {
        return this.list.pop();
      }

    Which might transform the previous example to something like:

      function Foo() {
        list = [];
      }
      push = function(a) {
        list.push(a);
      }
      last = function() {
        return list.pop();
      }"""

    name = "replace-properties-by-globals"

    def run(self, testcase, interesting, tempFilename):  # pylint: disable=missing-return-doc,missing-return-type-doc
        # pylint: disable=invalid-name
        chunkSize = min(self.minimizeMax, 2 * largestPowerOfTwoSmallerThan(len(testcase)))
        finalChunkSize = max(self.minimizeMin, 1)

        origNumChars = 0
        for line in testcase.parts:
            origNumChars += len(line)

        numChars = origNumChars
        while 1:
            numRemovedChars, testcase = self.tryMakingGlobals(chunkSize, numChars, testcase, interesting, tempFilename)
            numChars -= numRemovedChars

            last = (chunkSize <= finalChunkSize)

            if numRemovedChars and (self.minimizeRepeat == "always" or (self.minimizeRepeat == "last" and last)):
                # Repeat with the same chunk size
                pass
            elif last:
                # Done
                break
            else:
                # Continue with the next smaller chunk size
                chunkSize >>= 1

        log.info("  Initial size: %s", quantity(origNumChars, "character"))
        log.info("  Final size: %s", quantity(numChars, "character"))

        return 0, (finalChunkSize == 1 and self.minimizeRepeat != "never"), testcase

    def tryMakingGlobals(self, chunkSize, numChars, testcase, interesting, tempFilename):
        # pylint: disable=invalid-name,missing-param-doc,missing-return-doc,missing-return-type-doc,missing-type-doc
        # pylint: disable=too-many-arguments,too-many-branches,too-complex,too-many-locals
        """Make a single run through the testcase, trying to remove chunks of size chunkSize.

        Returns True iff any chunks were removed."""

        numRemovedChars = 0
        numChunks = divideRoundingUp(len(testcase), chunkSize)
        finalChunkSize = max(self.minimizeMin, 1)

        # Map words to the chunk indexes in which they are present.
        words = {}
        for chunk, line in enumerate(testcase.parts):
            for match in re.finditer(br"(?<=[\w\d_])\.(\w+)", line):
                word = match.group(1)
                if word not in words:
                    words[word] = [chunk]
                else:
                    words[word] += [chunk]

        # All patterns have been removed sucessfully.
        if not words:
            return 0, testcase

        log.info("Starting a round with chunks of %s.", quantity(chunkSize, testcase.atom))
        summary = list("S" * numChunks)

        for word, chunks in list(words.items()):
            chunkIndexes = {}
            for chunkStart in chunks:
                chunkIdx = chunkStart // chunkSize
                if chunkIdx not in chunkIndexes:
                    chunkIndexes[chunkIdx] = [chunkStart]
                else:
                    chunkIndexes[chunkIdx] += [chunkStart]

            for chunkIdx, chunkStarts in chunkIndexes.items():
                # Unless this is the final size, let's try to remove couple of
                # prefixes, otherwise wait for the final size to remove each of them
                # individually.
                if len(chunkStarts) == 1 and finalChunkSize != chunkSize:
                    continue

                description = "'%s' in chunk #%d of %d chunks of size %d" % (
                    word.decode("utf-8", "replace"), chunkIdx, numChunks, chunkSize)

                maybeRemoved = 0
                newTC = testcase.copy()
                for chunkStart in chunkStarts:
                    subst = re.sub(br"[\w_.]+\." + word, word, newTC.parts[chunkStart])
                    maybeRemoved += len(newTC.parts[chunkStart]) - len(subst)
                    newTC.parts = newTC.parts[:chunkStart] + [subst] + newTC.parts[(chunkStart + 1):]

                if interesting(newTC):
                    testcase = newTC
                    log.info("Yay, reduced it by removing prefixes of %s :)", description)
                    numRemovedChars += maybeRemoved
                    summary[chunkIdx] = "s"
                    words[word] = [c for c in chunks if c not in chunkIndexes]
                    if not words[word]:
                        del words[word]
                else:
                    log.info("Removing prefixes of %s made the file 'uninteresting'.", description)

        numSurvivingChars = numChars - numRemovedChars
        printableSummary = " ".join(
            "".join(summary[(2 * i):min(2 * (i + 1), numChunks + 1)]) for i in range(numChunks // 2 + numChunks % 2))
        log.info("")
        log.info("Done with a round of chunk size %d!", chunkSize)
        log.info("%s survived; %s shortened.",
                 quantity(summary.count("S"), "chunk"),
                 quantity(summary.count("s"), "chunk"))
        log.info("%s survived; %s removed.",
                 quantity(numSurvivingChars, "character"),
                 quantity(numRemovedChars, "character"))
        log.info("Which chunks survived: %s", printableSummary)
        log.info("")

        testcase.writeTestcase(tempFilename("did-round-%d" % chunkSize))

        return numRemovedChars, testcase


class ReplaceArgumentsByGlobals(Minimize):
    """    This strategy attempts to replace arguments by globals, for each named
    argument of a function we add a setter of the global of the same name before
    the function call.  The goal is to remove functions by making empty arguments
    lists instead.

      function foo(a,b) {
        list = a + b;
      }
      foo(2, 3)

    becomes:

      function foo() {
        list = a + b;
      }
      a = 2;
      b = 3;
      foo()

    The next logical step is inlining the body of the function at the call site."""

    name = "replace-arguments-by-globals"

    def run(self, testcase, interesting, tempFilename):  # pylint: disable=missing-return-doc,missing-return-type-doc
        roundNum = 0  # pylint: disable=invalid-name
        while 1:
            # pylint: disable=invalid-name
            numRemovedArguments, testcase = self.tryArgumentsAsGlobals(roundNum, testcase, interesting, tempFilename)
            roundNum += 1  # pylint: disable=invalid-name

            if numRemovedArguments and (self.minimizeRepeat == "always" or self.minimizeRepeat == "last"):
                # Repeat with the same chunk size
                pass
            else:
                # Done
                break

        return 0, False, testcase

    @staticmethod
    def tryArgumentsAsGlobals(roundNum, testcase, interesting, tempFilename):  # pylint: disable=invalid-name
        # pylint: disable=missing-param-doc,missing-return-doc,missing-return-type-doc,missing-type-doc
        # pylint: disable=too-many-branches,too-complex,too-many-locals,too-many-statements
        """Make a single run through the testcase, trying to remove chunks of size chunkSize.

        Returns True iff any chunks were removed."""

        numMovedArguments = 0  # pylint: disable=invalid-name
        numSurvivedArguments = 0  # pylint: disable=invalid-name

        # Map words to the chunk indexes in which they are present.
        functions = {}
        anonymousQueue = []  # pylint: disable=invalid-name
        anonymousStack = []  # pylint: disable=invalid-name
        for chunk, line in enumerate(testcase.parts):
            # Match function definition with at least one argument.
            for match in re.finditer(br"(?:function\s+(\w+)|(\w+)\s*=\s*function)\s*\((\s*\w+\s*(?:,\s*\w+\s*)*)\)",
                                     line):
                fun = match.group(1)
                if fun is None:
                    fun = match.group(2)

                if match.group(3) == b"":
                    args = []
                else:
                    args = match.group(3).split(b",")

                if fun not in functions:
                    functions[fun] = {"defs": args, "argsPattern": match.group(3), "chunk": chunk, "uses": []}
                else:
                    functions[fun]["defs"] = args
                    functions[fun]["argsPattern"] = match.group(3)
                    functions[fun]["chunk"] = chunk

            # Match anonymous function definition, which are surrounded by parentheses.
            for match in re.finditer(br"\(function\s*\w*\s*\(((?:\s*\w+\s*(?:,\s*\w+\s*)*)?)\)\s*{", line):
                if match.group(1) == b"":
                    args = []
                else:
                    args = match.group(1).split(",")
                # pylint: disable=invalid-name
                anonymousStack += [{"defs": args, "chunk": chunk, "use": None, "useChunk": 0}]

            # Match calls of anonymous function.
            for match in re.finditer(br"}\s*\)\s*\(((?:[^()]|\([^,()]*\))*)\)", line):
                if not anonymousStack:
                    continue
                anon = anonymousStack[-1]
                anonymousStack = anonymousStack[:-1]  # pylint: disable=invalid-name
                if match.group(1) == b"" and not anon["defs"]:
                    continue
                if match.group(1) == b"":
                    args = []
                else:
                    args = match.group(1).split(b",")
                anon["use"] = args
                anon["useChunk"] = chunk
                anonymousQueue += [anon]  # pylint: disable=invalid-name

            # match function calls. (and some definitions)
            for match in re.finditer(br"((\w+)\s*\(((?:[^()]|\([^,()]*\))*)\))", line):
                pattern = match.group(1)
                fun = match.group(2)
                if match.group(3) == b"":
                    args = []
                else:
                    args = match.group(3).split(b",")
                if fun not in functions:
                    functions[fun] = {"uses": []}
                functions[fun]["uses"] += [{"values": args, "chunk": chunk, "pattern": pattern}]

        # All patterns have been removed sucessfully.
        if not functions and not anonymousQueue:
            return 0, testcase

        log.info("Starting removing function arguments.")

        for fun, argsMap in functions.items():  # pylint: disable=invalid-name
            description = "arguments of '%s'" % fun.decode("utf-8", "replace")
            if "defs" not in argsMap or not argsMap["uses"]:
                log.info("Ignoring %s because it is 'uninteresting'.", description)
                continue

            maybeMovedArguments = 0  # pylint: disable=invalid-name
            newTC = testcase.copy()  # pylint: disable=invalid-name

            # Remove the function definition arguments
            argDefs = argsMap["defs"]  # pylint: disable=invalid-name
            defChunk = argsMap["chunk"]  # pylint: disable=invalid-name
            subst = newTC.parts[defChunk].replace(argsMap["argsPattern"], b"", 1)
            newTC.parts = newTC.parts[:defChunk] + [subst] + newTC.parts[(defChunk + 1):]

            # Copy callers arguments to globals.
            for argUse in argsMap["uses"]:  # pylint: disable=invalid-name
                values = argUse["values"]
                chunk = argUse["chunk"]
                if chunk == defChunk and values == argDefs:
                    continue
                while len(values) < len(argDefs):
                    values = values + [b"undefined"]
                setters = b"".join((a + b" = " + v + b";\n") for (a, v) in zip(argDefs, values))
                subst = setters + newTC.parts[chunk]
                newTC.parts = newTC.parts[:chunk] + [subst] + newTC.parts[(chunk + 1):]
            maybeMovedArguments += len(argDefs)  # pylint: disable=invalid-name

            if interesting(newTC):
                testcase = newTC
                log.info("Yay, reduced it by removing %s :)", description)
                numMovedArguments += maybeMovedArguments  # pylint: disable=invalid-name
            else:
                numSurvivedArguments += maybeMovedArguments  # pylint: disable=invalid-name
                log.info("Removing %s made the file 'uninteresting'.", description)

            for argUse in argsMap["uses"]:  # pylint: disable=invalid-name
                chunk = argUse["chunk"]
                values = argUse["values"]
                if chunk == defChunk and values == argDefs:
                    continue

                newTC = testcase.copy()  # pylint: disable=invalid-name
                subst = newTC.parts[chunk].replace(argUse["pattern"], fun + b"()", 1)
                if newTC.parts[chunk] == subst:
                    continue
                newTC.parts = newTC.parts[:chunk] + [subst] + newTC.parts[(chunk + 1):]
                maybeMovedArguments = len(values)  # pylint: disable=invalid-name

                descriptionChunk = "%s at %s #%d" % (description, testcase.atom, chunk)  # pylint: disable=invalid-name
                if interesting(newTC):
                    testcase = newTC
                    log.info("Yay, reduced it by removing %s :)", descriptionChunk)
                    numMovedArguments += maybeMovedArguments  # pylint: disable=invalid-name
                else:
                    numSurvivedArguments += maybeMovedArguments  # pylint: disable=invalid-name
                    log.info("Removing %s made the file 'uninteresting'.", descriptionChunk)

        # Remove immediate anonymous function calls.
        for anon in anonymousQueue:
            noopChanges = 0  # pylint: disable=invalid-name
            maybeMovedArguments = 0  # pylint: disable=invalid-name
            newTC = testcase.copy()  # pylint: disable=invalid-name

            argDefs = anon["defs"]  # pylint: disable=invalid-name
            defChunk = anon["chunk"]  # pylint: disable=invalid-name
            values = anon["use"]
            chunk = anon["useChunk"]
            description = "arguments of anonymous function at #%s %d" % (testcase.atom, defChunk)

            # Remove arguments of the function.
            subst = newTC.parts[defChunk].replace(b",".join(argDefs), b"", 1)
            if newTC.parts[defChunk] == subst:
                noopChanges += 1  # pylint: disable=invalid-name
            newTC.parts = newTC.parts[:defChunk] + [subst] + newTC.parts[(defChunk + 1):]

            # Replace arguments by their value in the scope of the function.
            while len(values) < len(argDefs):
                values = values + [b"undefined"]
            setters = b"".join(b"var %s = %s;\n" % (a, v) for a, v in zip(argDefs, values))
            subst = newTC.parts[defChunk] + b"\n" + setters
            if newTC.parts[defChunk] == subst:
                noopChanges += 1  # pylint: disable=invalid-name
            newTC.parts = newTC.parts[:defChunk] + [subst] + newTC.parts[(defChunk + 1):]

            # Remove arguments of the anonymous function call.
            subst = newTC.parts[chunk].replace(b",".join(anon["use"]), b"", 1)
            if newTC.parts[chunk] == subst:
                noopChanges += 1  # pylint: disable=invalid-name
            newTC.parts = newTC.parts[:chunk] + [subst] + newTC.parts[(chunk + 1):]
            maybeMovedArguments += len(values)  # pylint: disable=invalid-name

            if noopChanges == 3:
                continue

            if interesting(newTC):
                testcase = newTC
                log.info("Yay, reduced it by removing %s :)", description)
                numMovedArguments += maybeMovedArguments  # pylint: disable=invalid-name
            else:
                numSurvivedArguments += maybeMovedArguments  # pylint: disable=invalid-name
                log.info("Removing %s made the file 'uninteresting'.", description)

        log.info("")
        log.info("Done with this round!")
        log.info("%s moved;", quantity(numMovedArguments, "argument"))
        log.info("%s survived.", quantity(numSurvivedArguments, "argument"))

        testcase.writeTestcase(tempFilename("did-round-%d" % roundNum))

        return numMovedArguments, testcase


class CollapseEmptyBraces(Minimize):
    """ Perform standard line based reduction but collapse empty braces at the end of each round
    This ensures that empty braces are reduced in a single pass of the reduction strategy

    Example:
        // Original
        function foo() {
        }

        // Post-processed
        function foo() { }
    """
    name = "minimize-collapse-brace"

    @staticmethod
    def apply_post_round_op(testcase):
        """ Collapse braces separated by whitespace
        Args:
            testcase (Testcase): Testcase to be reduced.
        Returns:
            bool: True if callback was performed successfully, False otherwise.
        """
        raw = b"".join(testcase.parts)
        modified = re.sub(br'{\s+}', b'{ }', raw)

        # Don't update the testcase if no changes were applied
        if raw != modified:
            with open(testcase.filename, 'wb') as f:
                f.write(testcase.before)
                f.write(modified)
                f.write(testcase.after)

            # Re-parse the modified testcase
            testcase.readTestcase(testcase.filename)

            return True

        return False


class Lithium(object):  # pylint: disable=missing-docstring,too-many-instance-attributes

    def __init__(self):

        self.strategy = None

        self.conditionScript = None  # pylint: disable=invalid-name
        self.conditionArgs = None  # pylint: disable=invalid-name

        self.testCount = 0  # pylint: disable=invalid-name
        self.testTotal = 0  # pylint: disable=invalid-name

        self.tempDir = None  # pylint: disable=invalid-name

        self.testcase = None
        self.lastInteresting = None  # pylint: disable=invalid-name

        self.tempFileCount = 1  # pylint: disable=invalid-name

        self.unique = False

        self.testcaseLock = threading.RLock()

    def main(self, args=None):  # pylint: disable=missing-docstring,missing-return-doc,missing-return-type-doc
        logging.basicConfig(format="%(message)s", level=logging.INFO)
        self.processArgs(args)

        try:
            return self.run()

        except LithiumError as e:  # pylint: disable=invalid-name
            summaryHeader()
            log.error(e)
            return 1

    def run(self):  # pylint: disable=missing-docstring,missing-return-doc,missing-return-type-doc
        if hasattr(self.conditionScript, "init"):
            self.conditionScript.init(self.conditionArgs)

        try:
            if not self.tempDir:
                self.createTempDir()
                log.info("Intermediate files will be stored in %s%s.", self.tempDir, os.sep)

            result = self.strategy.main(self.testcase, self.interesting, self.testcaseTempFilename)

            log.info("  Tests performed: %d", self.testCount)
            log.info("  Test total: %s", quantity(self.testTotal, self.testcase.atom))

            return result

        finally:
            if hasattr(self.conditionScript, "cleanup"):
                self.conditionScript.cleanup(self.conditionArgs)

            # Make sure we exit with an interesting testcase
            if self.lastInteresting is not None:
                self.lastInteresting.writeTestcase()

    def processArgs(self, argv=None):  # pylint: disable=invalid-name,missing-param-doc,missing-type-doc
        # pylint: disable=too-complex,too-many-locals
        """Build list of strategies and testcase types."""

        strategies = {}
        testcaseTypes = {}  # pylint: disable=invalid-name
        for globalValue in globals().values():  # pylint: disable=invalid-name
            if isinstance(globalValue, type):
                if globalValue is not Strategy and issubclass(globalValue, Strategy):
                    assert globalValue.name not in strategies
                    strategies[globalValue.name] = globalValue
                elif globalValue is not Testcase and issubclass(globalValue, Testcase):
                    assert globalValue.atom not in testcaseTypes
                    testcaseTypes[globalValue.atom] = globalValue

        # Try to parse --conflict before anything else
        class ArgParseTry(argparse.ArgumentParser):  # pylint: disable=missing-docstring
            def exit(subself, **kwds):  # pylint: disable=arguments-differ,no-self-argument
                pass

            def error(subself, message):  # pylint: disable=no-self-argument
                pass

        defaultStrategy = "minimize"  # pylint: disable=invalid-name
        assert defaultStrategy in strategies
        strategyParser = ArgParseTry(add_help=False)  # pylint: disable=invalid-name
        strategyParser.add_argument(
            "--strategy",
            default=defaultStrategy,
            choices=strategies.keys())
        args = strategyParser.parse_known_args(argv)
        self.strategy = strategies.get(args[0].strategy if args else None, strategies[defaultStrategy])()

        parser = argparse.ArgumentParser(
            description="Lithium, an automated testcase reduction tool",
            epilog="See docs/using-for-firefox.md for more information.",
            usage="python -m lithium [options] condition [condition options] file-to-reduce")
        grp_opt = parser.add_argument_group(description="Lithium options")
        grp_opt.add_argument(
            "--testcase",
            help="testcase file. default: last argument is used.")
        grp_opt.add_argument(
            "--tempdir",
            help="specify the directory to use as temporary directory.")
        grp_opt.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="enable verbose debug logging")
        grp_atoms = grp_opt.add_mutually_exclusive_group()
        grp_atoms.add_argument(
            "-c", "--char",
            action="store_true",
            help="Don't treat lines as atomic units; "
                 "treat the file as a sequence of characters rather than a sequence of lines.")
        grp_atoms.add_argument(
            "-j", "--js",
            action="store_true",
            help="Same as --char but only operate within JS strings, keeping escapes intact.")
        grp_atoms.add_argument(
            "-s", "--symbol",
            action="store_true",
            help="Treat the file as a sequence of strings separated by tokens. "
                 "The characters by which the strings are delimited are defined by "
                 "the --cutBefore, and --cutAfter options.")
        grp_opt.add_argument(
            "--cutBefore",
            default=TestcaseSymbol.DEFAULT_CUT_BEFORE,
            help="See --symbol. default: %s" % TestcaseSymbol.DEFAULT_CUT_BEFORE.decode("utf-8"))
        grp_opt.add_argument(
            "--cutAfter",
            default=TestcaseSymbol.DEFAULT_CUT_AFTER,
            help="See --symbol. default: %s" % TestcaseSymbol.DEFAULT_CUT_AFTER.decode("utf-8"))
        grp_opt.add_argument(
            "--strategy",
            default=self.strategy.name,  # this has already been parsed above, it's only here for the help message
            choices=strategies.keys(),
            help="reduction strategy to use. default: %s" % defaultStrategy)
        grp_opt.add_argument(
            "--unique", "-u",
            action="store_true",
            help="use a unique filename per-iteration (required for parallel reduction)")
        self.strategy.addArgs(parser)
        grp_ext = parser.add_argument_group(description="Condition, condition options and file-to-reduce")
        grp_ext.add_argument(
            "extra_args",
            action="append",
            nargs=argparse.REMAINDER,
            help="condition [condition options] file-to-reduce")

        args = parser.parse_args(argv)
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        self.strategy.processArgs(parser, args)

        self.tempDir = args.tempdir
        atom = TestcaseChar.atom if args.char else TestcaseLine.atom
        atom = TestcaseJsStr.atom if args.js else atom
        atom = TestcaseSymbol.atom if args.symbol else atom

        extra_args = args.extra_args[0]

        if args.testcase:
            testcaseFilename = args.testcase  # pylint: disable=invalid-name
        elif extra_args:
            # can be overridden by --testcase in processOptions
            testcaseFilename = extra_args[-1]  # pylint: disable=invalid-name
        else:
            parser.error("No testcase specified (use --testcase or last condition arg)")

        if args.unique:
            if args.testcase:
                parser.error("--unique and --testcase are not supported together")
            self.unique = True
        else:
            self.unique = False

        self.testcase = testcaseTypes[atom]()
        if args.symbol:
            self.testcase.cutBefore = args.cutBefore
            self.testcase.cutAfter = args.cutAfter
        self.testcase.readTestcase(testcaseFilename)

        self.conditionScript = rel_or_abs_import(extra_args[0])
        self.conditionArgs = extra_args[1:]

    def testcaseTempFilename(self, partialFilename, useNumber=True):  # pylint: disable=invalid-name,missing-docstring
        # pylint: disable=missing-return-doc,missing-return-type-doc
        if useNumber:
            with self.testcaseLock:
                partialFilename = "%d-%s" % (self.tempFileCount, partialFilename)
                self.tempFileCount += 1
        return os.path.join(self.tempDir, partialFilename + self.testcase.extension)

    def createTempDir(self):  # pylint: disable=invalid-name,missing-docstring
        i = 1
        while True:
            with self.testcaseLock:
                self.tempDir = "tmp%d" % i
            # To avoid race conditions, we use try/except instead of exists/create
            # Hopefully we don't get any errors other than "File exists" :)
            try:
                os.mkdir(self.tempDir)
                break
            except OSError:
                i += 1

    # If the file is still interesting after the change, changes "parts" and returns True.
    def interesting(self, testcaseSuggestion, writeIt=True):  # pylint: disable=invalid-name,missing-docstring
        # pylint: disable=missing-return-doc,missing-return-type-doc
        args = self.conditionArgs[:]
        if writeIt:
            if self.unique:
                base = os.path.splitext(os.path.basename(self.testcase.filename))[0]
                path = os.path.dirname(self.testcase.filename)
                hnd, filename = tempfile.mkstemp(suffix=self.testcase.extension, prefix=base, dir=path)
                os.close(hnd)
                args[-1] = filename
            else:
                filename = None
            testcaseSuggestion.writeTestcase(filename=filename)

        with self.testcaseLock:
            self.testCount += 1
            self.testTotal += len(testcaseSuggestion.parts)

            tempPrefix = os.path.join(self.tempDir, "%d" % self.tempFileCount)  # pylint: disable=invalid-name

        inter = self.conditionScript.interesting(args, tempPrefix)

        if writeIt and self.unique:
            os.unlink(filename)

        # Save an extra copy of the file inside the temp directory.
        # This is useful if you're reducing an assertion and encounter a crash:
        # it gives you a way to try to reproduce the crash.
        if self.tempDir:
            tempFileTag = "interesting" if inter else "boring"  # pylint: disable=invalid-name
            testcaseSuggestion.writeTestcase(self.testcaseTempFilename(tempFileTag))

        if inter:
            with self.testcaseLock:
                self.testcase = testcaseSuggestion
                self.lastInteresting = self.testcase

        return inter


# Helpers

def summaryHeader():  # pylint: disable=invalid-name,missing-docstring
    log.info("=== LITHIUM SUMMARY ===")


def divideRoundingUp(n, d):  # pylint: disable=invalid-name,missing-docstring,missing-return-doc,missing-return-type-doc
    return (n // d) + (1 if n % d != 0 else 0)


def isPowerOfTwo(n):  # pylint: disable=invalid-name,missing-docstring,missing-return-doc,missing-return-type-doc
    return (1 << max(n.bit_length() - 1, 0)) == n


def largestPowerOfTwoSmallerThan(n):  # pylint: disable=invalid-name,missing-docstring
    # pylint: disable=missing-return-doc,missing-return-type-doc
    result = 1 << max(n.bit_length() - 1, 0)
    if result == n and n > 1:
        result >>= 1
    return result


def quantity(n, unit):  # pylint: disable=invalid-name,missing-param-doc
    # pylint: disable=missing-return-doc,missing-return-type-doc,missing-type-doc
    """Convert a quantity to a string, with correct pluralization."""
    r = "%d %s" % (n, unit)  # pylint: disable=invalid-name
    if n != 1:
        r += "s"  # pylint: disable=invalid-name
    return r


def main():
    exit(Lithium().main())


if __name__ == "__main__":
    main()
