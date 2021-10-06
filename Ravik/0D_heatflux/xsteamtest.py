# -*- coding: utf-8 -*-
#tests xsteam python

from pyXSteam.XSteam import XSteam

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)

p=1.013

Tsat=steamTable.tsat_p(p)
print(Tsat)
rhof=steamTable.rhoL_p(p)
print(rhof)

muf=steamTable.my_pt(100,p)
print(muf)
