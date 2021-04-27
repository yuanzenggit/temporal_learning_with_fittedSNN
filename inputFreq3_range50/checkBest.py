import numpy as np
import os
for name in ['fit_LIF1', 'fit_LIF2', 'fit_LIF3', 'fit_ALIF1', 'fit_ALIF2', 'fit_ALIF3']:
	os.system('grep \'finished with values\' '+str(name)+'.txt > temp.txt')
	os.system('cut -d\" \" -f 9,10 temp.txt > temp1.txt')
	os.system('sed -i "s/\[//g" temp1.txt')
	os.system('sed -i "s/\]//g" temp1.txt')
	data = np.loadtxt("temp1.txt", delimiter=',')
	good = np.where((data[:200, 1]<10) & (data[:200, 1]>0))
	minNum=10
	index=0
	for g in good[0]:
		#print(g, data[g][0])
		if data[g][0]<minNum:
			minNum=data[g][0]
			index=g

	print(name)
	os.system('sed -n '+str(index+1)+'p temp.txt')
