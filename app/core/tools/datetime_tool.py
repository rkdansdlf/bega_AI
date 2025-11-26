from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any

def get_current_datetime() -> Dict[str, Any]:
    """현재 날짜와 시간을 반환합니다 (한국 시간 기준)"""
    kst = ZoneInfo('Asia/Seoul')
    now = datetime.now(kst)

    return {
        "current_time": now.isoformat(),
        "formatted_date": now.strftime("%Y년 %m월 %d일"),
        "formatted_time": now.strftime("%H시 %M분"),
        "day_of_week": now.strftime("%A"),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute
    }

def get_baseball_season_info() -> Dict[str, Any]:
    """현재 야구 시즌 정보를 반환합니다"""
    kst = ZoneInfo('Asia/Seoul')
    now = datetime.now(kst)

    # KBO 시즌은 보통 3월~10월
    if 3 <= now.month <= 10:
        season_status = "정규시즌"
    elif now.month in [11, 12]:
        season_status = "포스트시즌/오프시즌"
    else:
        season_status = "오프시즌"

    return {
        "current_year": now.year,
        "season_status": season_status,
        "is_season_active": 3 <= now.month <= 10
    }

