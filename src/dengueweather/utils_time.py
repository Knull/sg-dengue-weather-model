
from datetime import date, datetime
from typing import Tuple


def date_to_iso_week(dt: date) -> Tuple[int, int]:
    iso_calendar = dt.isocalendar()
    return (iso_calendar.year, iso_calendar.week)


def iso_week_to_date(iso_year: int, iso_week: int) -> date:
    return datetime.fromisocalendar(iso_year, iso_week, 1).date()