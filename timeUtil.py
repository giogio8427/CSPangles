# Function to calculate days in month considering leap years
def days_in_month(month, year):
    """Return the number of days in a month for a given year."""
    month = int(month)
    year = int(year)
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    elif month == 2:
        # Check for leap year
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29
        else:
            return 28
    else:
        return 31  # Default fallback
    
def timeZoneToUTC(year, month, day, timeCurr, timeZone):
# Adjust the time for the specified time zone to compute UTC
    timeCurr = timeCurr - int(timeZone * 60)
    # Adjust the date for UTC if the time adjustment crosses day boundaries
    if timeCurr < 0:
        # Move to the previous day
        timeCurr += 24 * 60
        day -= 1
        if day < 1:
            month -= 1
            if month < 1:
                month = 12
                year -= 1
            day = days_in_month(month, year)
    elif timeCurr >= 24 * 60:
        # Move to the next day
        timeCurr -= 24 * 60
        day += 1
        if day > days_in_month(month, year):
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
    return year, month, day, timeCurr