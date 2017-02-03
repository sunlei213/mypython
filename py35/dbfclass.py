#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.5
@author:  sunlei
@license: Apache Licence 
@contact: 12166056@qq.com
@site: 
@software: PyCharm Community Edition
@file: dbfclass.py
@time: 2017/2/2 13:55
"""

import collections as col
import struct
import datetime


FIELD = col.namedtuple('Field', "name type lenth dec")
HEAD_FORMAT = '<BBBBLHH20x'
FIELD_FORMAT = '<11sc4xBB14x'


def recode_to_stream(f, fields, recodes):
    for reno, recode in enumerate(recodes, start=1):
        for i, field in enumerate(fields):
            if i == 0:
                del_flag = b'\x42' if recode[i] else b'\x32'
                f.write(del_flag)
                continue
            if len(recode[i]) > field.lenth:
                raise "记录{3:d}长度：{1:d}大于字段{0}设定长度：{2:d}，内容：{4}".format(
                    field.name, len(recode[i]), field.lenth, reno, recode[i])
            if field.type == 'N' or field.type == 'F':
                value = str(recode[i]).rjust(field.lenth, ' ').encode()
            elif field.type == 'D':
                value = recode[i].strftime('%Y%m%d').encode()
            elif field.type == 'L':
                value = str(recode[i])[0].upper().encode()
            else:
                value = str(recode[i])[:field.lenth].ljust(field.lenth, ' ').encode()
            f.write(value)


def head_to_stream(f, fields, recnum):
    """f is ByteIO, fields is list of field, recnum is total recode"""
    # 写Dbf文件头
    ver = 3
    now = datetime.datetime.now()
    yr, mon, day = now.year - 1900, now.month, now.day
    numfields = len(fields)
    lenheader = numfields * 32 + 33
    lenrecord = sum(field.lenth for field in fields)
    hdr = struct.pack(HEAD_FORMAT, ver, yr, mon, day, recnum, lenheader, lenrecord)
    f.write(hdr)

    # 写Dbf字段
    for field in fields:
        name = field.name.ljust(11, '\x00').encode()
        fld = struct.pack(FIELD_FORMAT, name, field.type[0].encode(), field.lenth, field.dec)
        f.write(fld)

    # 结束文件头
    f.write(b'\r')
