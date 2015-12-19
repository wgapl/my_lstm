#! /usr/bin/env python

import os, shutil

def look(folder, s):
    cwd = os.getcwd()
    go_back = False
    if folder != cwd:
        os.chdir(folder)
        go_back = True
    for x in os.listdir('.'):
        if os.path.isdir(x):
            look(x, s)
        else:
            fext = os.path.splitext(x)[-1]
            if fext == '.c' or fext == '.h':
                fh = open(x,'r')
                txt = fh.read()
                fh.close()
                s.append(txt)
    if go_back:
        os.chdir(cwd)

if __name__ == "__main__":
    s = []
    cwd = os.getcwd()
    look(cwd,s)
    bg_txt = ''.join(s)
    fh = open('all_linux_text.txt','w')
    fh.write(bg_txt)
    fh.close()
