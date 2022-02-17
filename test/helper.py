import unittest
import torch
from torch import Tensor


class TestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.atol = 1e-6

    def assertTensorsClose(self, expected: Tensor, actual: Tensor, atol=None, msg=""):
        atol = self.atol if atol is None else atol
        if not torch.allclose(expected, actual, atol=atol):
            norm_inf = (expected - actual).abs().max()
            self.fail(f'Not equal by {norm_inf}: {msg}')

    def assertEqualShape(self, expected: Tensor, actual: Tensor, msg=""):        
        if expected.shape != actual.shape:
            self.fail(f'Not equal shape: {msg}')

