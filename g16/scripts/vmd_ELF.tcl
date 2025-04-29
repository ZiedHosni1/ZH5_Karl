mol new ELF.cube type cube waitfor all
mol delrep 0 top
mol representation Isosurface 0.03 0 0 1 1  ;# 0.03 isovalue — adjust if needed
mol color Name
mol addrep top
display projection Orthographic
render TachyonInternal ELF.eps
quit
