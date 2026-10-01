"""Keep transmission imports safe without the RF extra."""

import subprocess
import sys


def test_import_without_skrf():
    script = """
import sys
sys.modules['skrf'] = None
import gsim.palace.transmission as transmission
assert transmission.PropagationResult
try:
    transmission.extract_propagation(
        None, None, length_difference_m=1, maximum_phase_index=1
    )
except ImportError as error:
    assert 'gsim[rf]' in str(error)
else:
    raise AssertionError('Expected the optional dependency error')
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603
