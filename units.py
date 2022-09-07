######################################################################
# FIlamEntary STructure Analysis (FIESTA) -- Pre-release
# Authors: Jiten Dhandha, Zoe Faes and Rowan Smith
# Last edited: September 2022
######################################################################

######################################################################
#                            AREPO units                             #
######################################################################

#Base units (in CGS)
ulength = 1.0000000e+17
umass = 1.991e33
uvel = 36447.268

#Derived units
utime = ulength/uvel
udensity = umass/ulength/ulength/ulength
uenergy= umass*uvel*uvel
ucolumn = umass/ulength/ulength
uparsec=ulength/3.0856e18

######################################################################
#                      Fundamental constants                         #
######################################################################

mp = 1.6726231e-24
kb = 1.3806485e-16