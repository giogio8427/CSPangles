import numpy as np
from timeUtil import timeZoneToUTC
import sunposition as sp
sp.disable_jit()
from compSub import comp


def solarAnglesDay(year, month, day, latitude, longitude, timeZone,axRot_az, compIncAngle, minTimeStep=15):
    """
    Calculate the solar angles based on azimuth and zenith angles.

    Parameters:
    azimuth (float): Azimuth angle in radians.
    zenith (float): Zenith angle in radians.

    Returns:
    tuple: A tuple containing the solar angles (azimuth, zenith).
    """


    year, month, day, timeCurr=timeZoneToUTC(year, month, day, 0, timeZone)

    # Convert year, month, day to strings with proper formatting
    year_str = str(year)
    month_str = f"{month:02d}"  # Ensure two digits with leading zero if needed
    day_str = f"{day:02d}"      # Ensure two digits with leading zero if needed
    
    hours = timeCurr // 60
    minutes = timeCurr % 60
    
    # Format the time part with leading zeros
    time_str = f"{hours:02d}:{minutes:02d}:00"
    # Create the complete datetime64 string
    stringDay = f"{year_str}-{month_str}-{day_str}"
    
    # For using in sunpos function
    #time_str = "00:00:00"
    inTime=np.datetime64(f"{stringDay}T{time_str}")
    currTime = inTime
    azimuth = np.zeros(int(60*24/minTimeStep))+99.
    zenith = np.zeros(int(60*24/minTimeStep)) +99. 
    incAngleArr = np.zeros(int(60*24/minTimeStep))
    trackAngleArr = np.zeros(int(60*24/minTimeStep))
    # Create the datetime64 object
    for ii in range(0,int(60*24/minTimeStep)):
        currTime = inTime + np.timedelta64(ii*minTimeStep, 'm')
        # Call the sunpos function
        aTemp,bTemp = sp.sunpos(currTime, latitude, longitude,0)[:2]
        azimuth[int(ii)] = aTemp
        zenith[int(ii)] = bTemp
        if compIncAngle:
            incAngle, trackAngle=comp(np.deg2rad(bTemp), np.deg2rad(aTemp), axRot_az)
            incAngleArr[int(ii)]=incAngle
            trackAngleArr[int(ii)]=trackAngle
    hourArrayDiscr = np.linspace(0, 24, 24*(60)//minTimeStep)
    return azimuth, zenith, hourArrayDiscr,incAngleArr, trackAngleArr