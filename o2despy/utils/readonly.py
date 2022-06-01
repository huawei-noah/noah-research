# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


class ReadOnlyList(list):
    def __init__(self, data=None):
        self._data = []
        if data is not None:
            for item in data:
                if isinstance(item, dict):
                    self._data.append( ReadOnlyDict(item) )
                elif isinstance(item, (list, set)):
                    self._data.append( ReadOnlyList(item) )
                elif hasattr(item, "as_read_only"):
                    self._data.append( item.as_read_only() )
                else:
                    self._data.append(item)
        super().__init__(self._data)

    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("Cannot modify ReadOnly Object")

    __setitem__ = __readonly__
    __delitem__ = __readonly__
    # __add__, __mul__, __rmul__
    __iadd__ = __readonly__
    __imul__ = __readonly__
    __reversed__ = __readonly__
    append = __readonly__
    insert = __readonly__
    extend = __readonly__
    pop = __readonly__
    remove = __readonly__
    clear = __readonly__
    reverse = __readonly__
    sort = __readonly__
    del __readonly__


class ReadOnlyDict(dict):
    def __init__(self, data):
        self._data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                self._data[key] = ReadOnlyDict(value)
            elif isinstance(value, (list, set)):
                self._data[key] = ReadOnlyList(value)
            elif hasattr(value, "as_read_only"):
                self._data[key] = value.as_read_only()
            else:
                self._data[key] = value
        super().__init__(self._data)
    
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("Cannot modify ReadOnly Object")
    
    __setitem__ = __readonly__
    __delitem__ = __readonly__
    setdefault = __readonly__
    pop = __readonly__
    popitem = __readonly__
    update = __readonly__
    clear = __readonly__
    del __readonly__
