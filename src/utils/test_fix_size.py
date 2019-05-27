from unittest import TestCase
import numpy as np

from utils.inference_utils import fix_size


class TestFix_size(TestCase):
    def test_fix_size(self):
        arr = np.random.random((1000, 456))
        ret1 = fix_size(arr, 500)
        self.assertEqual(arr[:500], ret1)
        ret2 = fix_size(arr, 2000)
        self.assertEqual(arr, ret2[:1000])

