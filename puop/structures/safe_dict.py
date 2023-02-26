import loguru
import warnings
from easydict import EasyDict

from puop.utils.os_utils import red

safe_level = 1  # 1 warn 2 error


# class SafeDict(EasyDict):
#
#     def __setattr__(self, name, value):
#         if name in self.keys():
#             err_msg = f'{name} has been existing.'
#             if safe_level == 1:
#                 loguru.logger.warning(err_msg)
#             else:
#                 loguru.logger.error(err_msg)
#                 raise RuntimeError(err_msg)
#         super(SafeDict, self).__setattr__(name, value)
#
#     __setitem__ = __setattr__
#
#     def pop_if_present(self, k):
#         if k in self.keys():
#             self.pop(k)

class SafeDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop', 'pop_if_present'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if name in self.keys():
            err_msg = f'{name} has been existing.'
            if safe_level == 1:
                warnings.warn(red(err_msg))
            else:
                loguru.logger.error(err_msg)
                raise RuntimeError(err_msg)
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(SafeDict, self).__setattr__(name, value)
        super(SafeDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(SafeDict, self).pop(k, d)

    def pop_if_present(self, k):
        if k in self.keys():
            self.pop(k)


def main():
    d = {'a': 1, 'b': 2}
    ed = SafeDict(d)
    # ed['a'] = 3
    # ed.a = 3
    ed2 = SafeDict(ed)
    ed2.a = 4
    print()


if __name__ == '__main__':
    main()
