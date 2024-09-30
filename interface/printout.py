#!/bin/python


def print_box(name: str, width=80):
    print('┏' +    "━"*width       + "┓")
    print('┃' + name.center(width) + '┃')
    print('┗' +    "━"*width       + "┛")
