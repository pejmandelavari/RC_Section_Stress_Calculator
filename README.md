RC Cracked Service Section Solver – One-Page Quick
Start
1) What the app does
• Cracked service stress analysis of RC sections under N, Mx, My.
• Plane sections remain plane; concrete tension neglected; steel linear elastic.
• Reports governing steel tension (fs,t,max) and concrete compression (fc,max).
2) Sign & axis conventions
• x right (+), y up (+), axes at concrete centroid.
• Compression positive, tension negative internally.
• Mx > 0 ® bottom in compression; My > 0 ® right side in compression.
• Reported steel stress fs: tension positive, compression negative.
3) Required inputs
Section file (.txt, SPColumn style)
Units: mm, mm².
Blocks: SOLIDS, OPENINGS, REINFORCEMENT.
Load combinations (.xlsx or .csv)
Columns: Combo, N_kN, Mx_kNm, My_kNm.
4) Typical settings
Ec = 25–35 GPa, Es = 200 GPa, Mesh = 20 mm.
Tolerances: DN = 2 kN, DMx = DMy = 2 kN·m.
5) Workflow
Upload files ® set parameters ® Run analysis ® review results ® download Excel & PDF.
6) Status meaning
OK: within tolerances.
NEAR: within 1.5× tolerances (usually acceptable).
FAIL: check load signs, units, or reference point.
7) Quick troubleshooting
Many FAILs ® check moment sign convention & units.
Mirrored plots ® try Swap X/Y.
Blocky contours ® reduce mesh size.
