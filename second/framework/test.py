import unittest
import numpy as np


class TestCase(unittest.TestCase):
    def _GetNdArray(self, a):
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        return a

    def assertAllEqual(self, a, b):
        """Asserts that two numpy arrays have the same values.
        Args:
        a: the expected numpy ndarray or anything can be converted to one.
        b: the actual numpy ndarray or anything can be converted to one.
        """
        a = self._GetNdArray(a)
        b = self._GetNdArray(b)
        self.assertEqual(a.shape, b.shape,
                         "Shape mismatch: expected %s, got %s." % (a.shape,
                                                                   b.shape))
        same = (a == b)

        if a.dtype == np.float32 or a.dtype == np.float64:
            same = np.logical_or(same, np.logical_and(
                np.isnan(a), np.isnan(b)))
        if not np.all(same):
            # Prints more details than np.testing.assert_array_equal.
            diff = np.logical_not(same)
            if a.ndim:
                x = a[np.where(diff)]
                y = b[np.where(diff)]
                print("not equal where = ", np.where(diff))
            else:
                # np.where is broken for scalars
                x, y = a, b
            print("not equal lhs = ", x)
            print("not equal rhs = ", y)
            np.testing.assert_array_equal(a, b)

    def assertAllClose(self, a, b, rtol=1e-6, atol=1e-6):
        """Asserts that two numpy arrays, or dicts of same, have near values.
        This does not support nested dicts.
        Args:
        a: The expected numpy ndarray (or anything can be converted to one), or
            dict of same. Must be a dict iff `b` is a dict.
        b: The actual numpy ndarray (or anything can be converted to one), or
            dict of same. Must be a dict iff `a` is a dict.
        rtol: relative tolerance.
        atol: absolute tolerance.
        Raises:
        ValueError: if only one of `a` and `b` is a dict.
        """
        is_a_dict = isinstance(a, dict)
        if is_a_dict != isinstance(b, dict):
            raise ValueError("Can't compare dict to non-dict, %s vs %s." % (a,
                                                                            b))
        if is_a_dict:
            self.assertCountEqual(
                a.keys(),
                b.keys(),
                msg="mismatched keys, expected %s, got %s" % (a.keys(),
                                                              b.keys()))
            for k in a:
                self._assertArrayLikeAllClose(
                    a[k],
                    b[k],
                    rtol=rtol,
                    atol=atol,
                    msg="%s: expected %s, got %s." % (k, a, b))
        else:
            self._assertArrayLikeAllClose(a, b, rtol=rtol, atol=atol)

    def _assertArrayLikeAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
        a = self._GetNdArray(a)
        b = self._GetNdArray(b)
        self.assertEqual(a.shape, b.shape,
                         "Shape mismatch: expected %s, got %s." % (a.shape,
                                                                   b.shape))
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            # Prints more details than np.testing.assert_allclose.
            #
            # NOTE: numpy.allclose (and numpy.testing.assert_allclose)
            # checks whether two arrays are element-wise equal within a
            # tolerance. The relative difference (rtol * abs(b)) and the
            # absolute difference atol are added together to compare against
            # the absolute difference between a and b.  Here, we want to
            # print out which elements violate such conditions.
            cond = np.logical_or(
                np.abs(a - b) > atol + rtol * np.abs(b),
                np.isnan(a) != np.isnan(b))
            if a.ndim:
                x = a[np.where(cond)]
                y = b[np.where(cond)]
                print("not close where = ", np.where(cond))
            else:
                # np.where is broken for scalars
                x, y = a, b
            print("not close lhs = ", x)
            print("not close rhs = ", y)
            print("not close dif = ", np.abs(x - y))
            print("not close tol = ", atol + rtol * np.abs(y))
            print("dtype = %s, shape = %s" % (a.dtype, a.shape))
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)
