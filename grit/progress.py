"""Text progress that is visible in terminals, subprocess streams and Modal logs."""

from contextlib import contextmanager
import sys
import threading

from tqdm import tqdm


class Progress:
    def __init__(self, enabled=True, prefix=""):
        self.enabled = enabled
        self.prefix = prefix

    def bar(self, desc, *, total=None, unit="item", **kwargs):
        return tqdm(total=total, desc=f"{self.prefix}{desc}", unit=unit,
                    file=sys.stdout, disable=not self.enabled, mininterval=1,
                    dynamic_ncols=False, **kwargs)

    def track(self, items, desc, *, total=None, unit="item"):
        if total is None and hasattr(items, "__len__"):
            total = len(items)
        with self.bar(desc, total=total, unit=unit) as bar:
            for item in items:
                yield item
                bar.update()

    def log(self, message):
        if self.enabled:
            tqdm.write(f"{self.prefix}{message}", file=sys.stdout)
            sys.stdout.flush()

    @contextmanager
    def phase(self, desc):
        """Show elapsed time for blocking work whose completion % is unknown."""
        if not self.enabled:
            yield
            return
        stop = threading.Event()
        with self.bar(desc, bar_format="{desc} | elapsed {elapsed}") as bar:
            def heartbeat():
                while not stop.wait(5):
                    bar.refresh()
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                yield
            except BaseException:
                bar.set_description_str(f"{self.prefix}{desc} — failed/interrupted")
                raise
            else:
                bar.set_description_str(f"{self.prefix}{desc} — done")
            finally:
                stop.set()
                thread.join()
