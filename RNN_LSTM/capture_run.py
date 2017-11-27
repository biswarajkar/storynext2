#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Reference: https://stackoverflow.com/questions/38338823/python-stdout-to-both-console-and-textfile-including-errors
# Script to capture console output to screen

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()
