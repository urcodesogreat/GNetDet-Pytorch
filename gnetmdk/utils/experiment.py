import os
import sys
import time
import contextlib
import bisect
from collections import deque

from gnetmdk.utils.colorize import colored

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def monitor(i, total, suffix="", length=24, bar='=', tip='>', tail='.'):
    """Print progress bar. Assume `i` starts at 0."""
    assert i < total, f"Incorrect progress: {i}/{total}"
    a = int((i + 1) / total * length)
    b = length - a
    b, c = (b - 1, 1) if b > 0 else (0, 0)
    print(f"\r{int(100 * (i + 1) / total):>2d}% [" +
          bar * a + tip * c + tail * b + f"] {i + 1:>4d}/{total:>4d} " +
          suffix, end='')
    if i == total - 1: print()


def progress_bar(freq=1, rank=0):
    freq = max(1, freq)
    t1 = t2 = st = time.time()
    avg_time = global_average()

    def get_color(ratio: float, breakpoints=(.4, .8), colors=("red", "yellow", "green")):
        idx = bisect.bisect(breakpoints, ratio)
        return colors[idx]

    def monitor(i, total, suffix="", length=24, bar='=', tip='>', tail='.'):
        """Print progress bar. Assume `i` starts at 0."""
        assert i < total, f"Incorrect progress: {i}/{total}"
        nonlocal t1, t2, st

        t1, t2 = t2, time.time()
        elapsed = avg_time(t2 - t1)

        if rank == 0 and ((i + 1) % freq == 0 or i == total - 1 or i == 0):
            a = int((i + 1) / total * length)
            b = length - a
            b, c = (b - 1, 1) if b > 0 else (0, 0)
            est = elapsed * (total - i - 1) / freq
            it_per_sec = freq / elapsed
            color = get_color((i + 1) / total)
            print(colored(f"\r{int(100 * (i + 1) / total):>3d}%", color) + ' [' +
                  bar * a + tip * c + tail * b + f"] {i + 1:>4d}/{total:>4d} " +
                  f"[{time_format(t2 - st)}<{time_format(est)}, {it_per_sec:.2f}it/s] " +
                  suffix, end='')
        if rank == 0 and i == total - 1:
            print()

    return monitor


class ProgressBar(object):
    _PREFIX_FMT = "{{percent}} [{{progress}}] {{step:{width}d}}/{total} [{{estimation}}]"
    _TITLE_FMT = "f\"{{{{' ':^{{space}}}}}} | {{{{'loss':^{width}}}}} | {{{{'loc':^{width}}}}} | " \
                 "{{{{'cls':^{width}}}}} | {{{{'cont':^{width}}}}} | {{{{'nocont':^{width}}}}} | " \
                 "{{{{'noobj':^{width}}}}}\""
    _LOSS_FMT = "| {loss:<.4f} | {loc_loss:<.4f} | {cls_loss:<.4f} " \
                "| {contain_loss:<.4f} | {not_contain_loss:<.4f} | {noobj_loss:<.4f} "

    def __init__(self, total: int, freq: int = 10, length: int = 20,
                 rank: int = 0, bar='=', tip='>', tail='.', out=sys.stderr):
        self._PREFIX_FMT = self._PREFIX_FMT.format(width=len(str(total)), total=total)
        self._TITLE_FMT = self._TITLE_FMT.format(width=6)
        self.total = max(1, total)
        self.freq = max(1, freq)
        self.rank = max(0, rank)
        self._prog_len = max(10, length)
        self._prex_len = 0
        self.bar = bar
        self.tip = tip
        self.tail = tail
        self._write = out.write
        self._avg_fn = global_average()
        self._start_time = time.time()
        self._last_time = time.time()
        self._elapsed_time = 0
        self._logged_error = False

    def _print_title(self):
        title = self._TITLE_FMT.format(space=self._prex_len - 1)
        title = eval(title)
        self._write(title)
        self._write('\n')

    @staticmethod
    def _get_color(ratio: float, breakpoints=(.4, .8), colors=("red", "yellow", "green")):
        idx = bisect.bisect(breakpoints, ratio)
        return colors[idx]

    def _update_timer(self, i: int):
        t1, t2 = self._last_time, time.time()
        self._elapsed_time = self._avg_fn(t2 - t1)
        self._last_time = t2
        if i < 8: self._start_time = time.time()

    def _get_progress(self, ratio: float):
        a = int(ratio * self._prog_len)
        c = self._prog_len - a
        a, b = (a - 1, 1) if 0 < a < self._prog_len - 1 else (a, 0)
        b, c = (1, c - 1) if a == self._prog_len - 1 else (b, c)
        progress = self.bar * a + self.tip * b + self.tail * c
        return colored(progress, self._get_color(ratio))

    def _get_estimation(self, i: int):
        running = self._last_time - self._start_time
        if i > 5:
            est = self._elapsed_time * (self.total - i - 1)
            its = self.freq / self._elapsed_time
        else:
            est = its = 0
        estimation = f"{time_format(running)}<{time_format(est)}, {its:4.1f}it/s"
        return estimation

    def _get_prefix(self, i: int):
        i = max(0, min(i, self.total - 1))
        ratio = min(1.0, max(0.0, round(i / (self.total - 1), ndigits=2)))
        percent = colored(f"\r{int(100 * ratio):>3d}%", self._get_color(ratio))
        progress = self._get_progress(ratio)
        estimation = self._get_estimation(i)
        prefix = self._PREFIX_FMT.format(percent=percent, progress=progress, step=i + 1, estimation=estimation)
        self._prex_len = len(prefix) - 2 * (len(colored("_", "yellow")) - 1)
        return prefix

    def format_loss(self, losses: dict, **kwargs):
        for key, value in kwargs.items():
            losses[key] = value
        try:
            return self._LOSS_FMT.format(**losses)
        except KeyError as e:
            if not self._logged_error:
                self._logged_error = True
                sys.stderr.write(str(e))
            return ""

    def update(self, i: int, suffix: str = ""):
        """
        Assume i ranges from 0 to total - 1
        """
        self._update_timer(i)
        if self.rank == 0:
            if (i + 1) % self.freq == 0 or i == self.total - 1 or i == 0:
                prefix = self._get_prefix(i)
                if i == 0:
                    self._start_time = time.time()
                    self._print_title()
                    self._write('-' * (self._prex_len + len(suffix)))
                    self._write('\n')
                # Log information
                self._write(f"{prefix} {suffix}")
                if i == self.total - 1: self._write('\n')


def moving_average(window_size=5):
    buffer = deque(maxlen=window_size)

    def call(value):
        buffer.append(value)
        return sum(buffer) / len(buffer)

    return call


def global_average():
    total = 0.
    count = 0.
    def call(value):
        nonlocal total, count
        total += value
        count += 1.
        return total / count
    return call


def time_format(sec):
    sec = int(sec)
    hour, sec = sec // 3600, sec % 3600
    minute, sec = sec // 60, sec % 60
    result = f"{minute:02d}:{sec:02d}"
    if hour:
        result = f"{hour:02d}:" + result
    return result


@contextlib.contextmanager
def silent(no_error: bool = False):
    """A context to silent all console outputs."""
    import sys

    def stdout_to_null(*args, **kwargs):
        pass

    stdout_write = sys.stdout.write
    stderr_write = sys.stderr.write
    sys.stdout.write = stdout_to_null
    if no_error:
        sys.stderr.write = stdout_to_null
    yield
    sys.stdout.write = stdout_write
    if no_error:
        sys.stderr.write = stderr_write
