import numpy as np
import lumapi
It=lumapi.INTERCONNECT("BackPropergation.icp")
Row=4
SweepResult=np.zeros(Row)
for i in range (Row):
    MeterName="OPWM_"+str(i+1)
        #temp=getresultdata(MeterName,"sum/power");
    SweepResult[i]=It.getresultdata(MeterName,"sum/power")
print(SweepResult)

0.0036+0.076+0.032+0.00168