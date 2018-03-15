'''
Created on Mar 15, 2018
'''
from datetime import datetime, timedelta

def calculate_nsm(dt_obj):
    """Get the number of seconds until midnight."""
    tomorrow = dt_obj + timedelta(1)
    midnight = datetime(year=tomorrow.year, month=tomorrow.month, 
                        day=tomorrow.day, hour=0, minute=0, second=0)
    return (midnight - dt_obj).seconds

def get_day_of_week(dt_obj):
    current_day = dt_obj.weekday()
    if current_day == 0 :
        return "Monday"
    elif current_day == 1:
        return "Tuesday"
    elif current_day == 2:
        return "Wednesday"
    elif current_day == 3:
        return "Thursday"
    elif current_day == 4:
        return "Friday"
    elif current_day == 5:
        return "Saturday"
    elif current_day == 6:
        return "Sunday"

def get_week_status(dt_obj):
    """Identify whether the data is weekend or weekday"""
    current_day = dt_obj.weekday()
    if current_day == 0 or current_day == 1 or current_day == 2 or current_day == 3 or current_day == 4:
        return "workday"
    else:
        return "weekend"
    
def add_attributes_from_dates(df):
    # add number of seconds from midnight for each day (NSM), the week status (weekend or workday) and the day of the week
    nsm = []
    day_of_week = []
    week_status = []
    df["nsm"] = ""
    df["day_of_week"] = ""
    df["week_status"] = ""
    
    for _,row in df.iterrows():
        dt_str = row["date"]
        dt_obj = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        nsm.append(calculate_nsm(dt_obj))
        day_of_week.append(get_day_of_week(dt_obj))
        week_status.append(get_week_status(dt_obj))
    df["nsm"] = nsm
    df["day_of_week"] = day_of_week
    df["week_status"] = week_status
    return df