import sys, os
import time
import datetime

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger

os.environ['TZ'] = "UTC"
time.tzset()

leaps=[ # /* leap seconds (y,m,d,h,m,s,utc-gpst) */
    [2017,1,1,0,0,0,-18],
    [2015,7,1,0,0,0,-17],
    [2012,7,1,0,0,0,-16],
    [2009,1,1,0,0,0,-15],
    [2006,1,1,0,0,0,-14],
    [1999,1,1,0,0,0,-13],
    [1997,7,1,0,0,0,-12],
    [1996,1,1,0,0,0,-11],
    [1994,7,1,0,0,0,-10],
    [1993,7,1,0,0,0, -9],
    [1992,7,1,0,0,0, -8],
    [1991,1,1,0,0,0, -7],
    [1990,1,1,0,0,0, -6],
    [1988,1,1,0,0,0, -5],
    [1985,7,1,0,0,0, -4],
    [1983,7,1,0,0,0, -3],
    [1982,7,1,0,0,0, -2],
    [1981,7,1,0,0,0, -1],
    [0]
]
gpst0 = [1980,1, 6,0,0,0] # /* gps time reference */
TIME_FMT = "%Y-%m-%d-%H-%M-%S.%f"

class UnixTime:
    class UnixGnssTime:
        def __init__(self) -> None:
            self.time = 0       # time (s) expressed by standard time_t
            self.sec = 0.0      # fraction of second under 1 s
        
        def toSec(self) -> float:
            return self.time + self.sec
        
        def fromUnix(self, ts: float):
            self.time = int(ts)
            self.sec = ts - int(ts)
            return self
        
        def toWeek(self):
            week = int(self.toSec() / (7 * 24 * 3600))
            sow = self.toSec() - week * (7 * 24 * 3600.0)
            return week, sow

    def __init__(self):
        pass
    
    def toGPS(self, ts: float) -> UnixGnssTime:
        """convert UTC to GPS time (expressed by Unix timestamp)

        Args:
            ts (float): UTC expressed by Unix

        Returns:
            UnixGnssTime: GPS time
        """
        # logger.info(leaps)
        
        gps_unix = 0
        for i in range(len(leaps)):
            # if (timediff(t,epoch2time(leaps[i]))>=0.0) return timeadd(t,-leaps[i][6]);
            if ts - self.fromEpoch(leaps[i]) >= 0: 
                gps_unix = ts - leaps[i][6]
                break
        
        return UnixTime.UnixGnssTime().fromUnix(gps_unix)

    def toGPSWeek(self, t: float) -> tuple[int, float]:
        t0= self.fromEpoch(gpst0)
        sec = t - t0
        frac = sec - int(sec)
        week = int(int(sec) / (7 * 24 * 3600))
        sow = int(sec) - week * 7 * 24 * 3600 + frac

        return week, sow
    
    def fromEpoch(self, epoch:list[int]) -> float:
        """from %Y-%m-%d-%H-%M-%S to Unix timestamp

        Args:
            epoch (list[int]): YMDHMS respectively

        Returns:
            float: unix timestamp
        """

        time_str = []
        for count, item in enumerate(epoch):
            if count > 5: break
            if count == 5:
                time_str.append("{:.6f}".format(item))
            else:
                time_str.append(str(item))

        time_str = '-'.join(time_str)
        timestamp = datetime.datetime.strptime(time_str, TIME_FMT).timestamp()

        # doy = [1,32,60,91,121,152,182,213,244,274,305,335]
        # time1 = UnixTime.UnixGnssTime()
        # year, mon, day = int(epoch[0]), int(epoch[1]), int(epoch[2])
        
        # if (year<1970 or 2099<year or mon<1 or 12<mon): return None
        
        # # /* leap year if year%4==0 in 1901-2099 */
        # tmp = 1 if year % 4 == 0 and mon >= 3 else 0
        # days = (year - 1970) * 365 + (year - 1969) / 4 + doy[mon - 1] + day - 2 + tmp
        # sec = int(epoch[5])
        # time1.time = int(days)*86400+ int(epoch[3])*3600 + int(epoch[4]) * 60 + sec
        # time1.sec = epoch[5] - sec

        # logger.info("test at: {}".format(timestamp - time1.toSec()))


        return timestamp # time1.toSec()

    def toEpoch(self, t:UnixGnssTime):
        # mday = [ 
        #     31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,
        #     31,29,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31
        # ]
        # days=int(t.time/86400)
        # sec=int(t.time-days*86400)
        # day, mon = 0, 0
        # ep = [0, 0, 0, 0, 0, 0]
        # day = days % 1461
        # while mon < 48:
        #     if day >= mday[mon]: day -= mday[mon]
        #     else: break
        #     mon += 1
        # ep[0]=1970+int(days/1461)*4+int(mon/12)
        # ep[1]=mon%12+1
        # ep[2]=day + 1
        # ep[3]=int(sec/3600)
        # ep[4]=int(sec%3600/60)
        # ep[5]=sec % 60 + t.sec

        # logger.info(datetime.datetime.fromtimestamp(t.toSec()).strftime(TIME_FMT))
        time_epoch_str = datetime.datetime.fromtimestamp(t.toSec()).strftime(TIME_FMT).split("-")
        time_epoch = []
        for i, item in enumerate(time_epoch_str):
            if i < 5: time_epoch.append(int(item))
            else: time_epoch.append(float(item))
        # logger.info(time_epoch)
        # logger.info(ep)
        return time_epoch

if __name__ == "__main__":
    test_time = 1679903098
    gps_time = UnixTime().toGPS(test_time)
    epoch_time = UnixTime().toEpoch(gps_time)
    stamp = UnixTime().fromEpoch(epoch_time)
    week, sow = UnixTime().toGPSWeek(stamp)

    logger.warning(gps_time.toSec())
    logger.warning(stamp)
    logger.warning(week)
    logger.warning(sow)
    logger.warning(week * 7 * 24 * 3600 + sow - stamp + UnixTime().fromEpoch(gpst0))



    logger.info(UnixTime().toGPSWeek(UnixTime().fromEpoch([2023, 3, 27, 7, 45, 16])))
