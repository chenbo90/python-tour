import unittest

import add


class MyTestCase(unittest.TestCase):
    def test_add(self):
        a = 1
        b = 2
        self.assertEqual(add.add(a,b),3,'测试结果1')  # add assertion here
        self.assertEqual(add.add(a,b),4,'测试结果2')  # add assertion here


# if __name__ == '__main__':
#     unittest.main(argv=['first-arg-ignored'],exit=False)
