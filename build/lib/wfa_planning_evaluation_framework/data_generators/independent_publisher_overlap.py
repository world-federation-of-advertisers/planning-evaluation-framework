"""TODO(jiayupeng): DO NOT SUBMIT without one-line documentation for independent_publisher_overlap.

TODO(jiayupeng): DO NOT SUBMIT without a detailed description of independent_publisher_overlap.
"""

from typing import Sequence

from absl import app


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
