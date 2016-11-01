# -*- coding: utf-8 -*-

"""

    stamaimer 10/21/16

"""

from glob import glob
from subprocess import call


for image in glob("images/*.jpg"):

    print image

    call(["python", "tubuledetect.py", image, "5", "5", "5"])
