"""Conditional runtime pins must select exactly one dependency on every platform."""
import unittest
from pathlib import Path
from packaging.markers import default_environment
from packaging.requirements import Requirement


class RuntimeRequirementsTests(unittest.TestCase):
    def test_platform_markers_keep_a_complete_unambiguous_runtime(self):
        path = Path(__file__).resolve().parents[2] / 'requirements-runtime.txt'
        requirements = [Requirement(line) for line in path.read_text().splitlines()
                        if line.strip() and not line.startswith('#')]
        names = {requirement.name for requirement in requirements}
        for system, machine in [('darwin', 'x86_64'), ('darwin', 'arm64'),
                                ('linux', 'x86_64'), ('win32', 'AMD64')]:
            with self.subTest(system=system, machine=machine):
                environment = {**default_environment(), 'sys_platform': system,
                               'platform_machine': machine, 'python_version': '3.12'}
                selected = [requirement for requirement in requirements
                            if requirement.marker is None or requirement.marker.evaluate(environment)]
                self.assertEqual({requirement.name for requirement in selected}, names)
                self.assertEqual(len(selected), len(names), 'Each dependency must have one applicable pin')
                for requirement in selected:
                    pins = list(requirement.specifier)
                    self.assertEqual(len(pins), 1)
                    self.assertEqual(pins[0].operator, '==')
                    self.assertNotIn('*', pins[0].version)
