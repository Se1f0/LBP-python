import numpy as np
M = [[120,32,150],
    [255,110,180],
    [230,55,69]
]
i_ref = int(len(M)/2)
j_ref = int(len(M)/2)
ref_value = M[i_ref][j_ref]
final_val = ""

#1
if M[i_ref-1][j_ref-1] >= ref_value :
    final_val += "1"
else :
    final_val += "0"
#2
if M[i_ref-1][j_ref] >= ref_value :
    final_val += "1"
else :
    final_val += "0"
#3
if M[i_ref-1][j_ref+1] >= ref_value :
    final_val += "1"
else :
    final_val += "0"
#4
if M[i_ref][j_ref+1] >= ref_value :
    final_val += "1"
else :
    final_val += "0"
#5
if M[i_ref+1][j_ref+1] >= ref_value :
    final_val += "1"
else :
    final_val += "0"
#6
if M[i_ref+1][j_ref] >= ref_value :
    final_val += "1"
else :
    final_val += "0"
#7
if M[i_ref+1][j_ref-1] >= ref_value :
    final_val += "1"
else :
    final_val += "0"
#8    
if M[i_ref][j_ref-1] >= ref_value :
    final_val += "1"
else :
    final_val += "0"

print(int(final_val,2))
