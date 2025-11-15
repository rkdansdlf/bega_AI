"""KBO 리그의 세이버메트릭스 지표 계산을 위한 유틸리티 함수 및 클래스 모음.

이 모듈은 OPS, BABIP, wOBA, wRC+, WAR 등 야구의 고급 통계 지표를
계산하는 함수를 제공합니다. 계산에 필요한 시즌별 가중치 및 상수는
`LeagueContext` 클래스를 통해 주입받습니다.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class LeagueContext:
    """
    시즌별 리그 상수를 저장하는 데이터 클래스.
    wOBA, wRC+, WAR 등 리그 평균에 의존하는 지표 계산에 사용됩니다.
    """
    wBB: float = 0.69
    wHBP: float = 0.72
    w1B: float = 0.88
    w2B: float = 1.25
    w3B: float = 1.60
    wHR: float = 2.05
    wOBA_scale: float = 1.20
    lg_wOBA: float = 0.330
    lg_R_per_PA: float = 0.115
    lg_OBP: float = 0.330
    lg_SLG: float = 0.400
    lg_ERA: float = 4.10
    lg_FIP: float = 4.20
    park_factor: float = 1.00
    runs_per_win: float = 10.0
    lg_ra9: float = 4.50
    fip_const: float = 3.10


def slg(H:int, doubles:int, triples:int, HR:int, AB:int) -> Optional[float]:
    if AB <= 0:
        return None
    _1B = H - doubles - triples - HR
    TB = _1B + (2 * doubles) + (3 * triples) + (4 * HR)
    return TB / AB

def ops(
    H: int, BB: int, HBP: int, AB: int, SF: int, _2B: int, _3B: int, HR: int
) -> Optional[float]:
    """OPS (On-base Plus Slugging)를 계산합니다."""
    try:
        obp_denominator = AB + BB + HBP + SF
        obp = (H + BB + HBP) / obp_denominator if obp_denominator > 0 else 0.0
        slg_val = slg(H, _2B, _3B, HR, AB)
        if slg_val is None: return None
        return obp + slg_val
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def babip(H: int, AB: int, HR: int, K: int, SF: int) -> Optional[float]:
    """BABIP (Batting Average on Balls In Play)를 계산합니다."""
    denominator = AB - K - HR + SF
    if denominator <= 0:
        return None
    return (H - HR) / denominator


def woba(
    BB: int, IBB: int, HBP: int, H: int, _2B: int, _3B: int, HR: int, AB: int, SF: int, ctx: LeagueContext
) -> Optional[float]:
    """wOBA (weighted On-Base Average)를 계산합니다."""
    try:
        uBB = BB - IBB
        _1B = H - _2B - _3B - HR
        numerator = (
            ctx.wBB * uBB
            + ctx.wHBP * HBP
            + ctx.w1B * _1B
            + ctx.w2B * _2B
            + ctx.w3B * _3B
            + ctx.wHR * HR
        )
        denominator = AB + uBB + SF + HBP
        return numerator / denominator if denominator > 0 else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def wrc_plus(woba_val: float, PA: int, ctx: LeagueContext) -> Optional[float]:
    """wRC+ (weighted Runs Created Plus)를 계산합니다."""
    if ctx.wOBA_scale == 0 or ctx.lg_R_per_PA == 0:
        return None
    park_adj = ctx.park_factor
    try:
        return 100 * (
            ((woba_val - ctx.lg_wOBA) / ctx.wOBA_scale) + ctx.lg_R_per_PA
        ) / (park_adj * ctx.lg_R_per_PA)
    except ZeroDivisionError:
        return None


def war_batter(
    woba_val: float,
    PA: int,
    baserunning_runs: float,
    fielding_runs: float,
    positional_runs: float,
    league_adj_runs: float,
    ctx: LeagueContext,
) -> Optional[float]:
    """타자 WAR (Wins Above Replacement)를 단순화된 공식으로 계산합니다."""
    if ctx.wOBA_scale == 0 or ctx.runs_per_win == 0:
        return None
    
    try:
        wraa = ((woba_val - ctx.lg_wOBA) / ctx.wOBA_scale) * PA
        replacement_runs = (20 / 600) * PA
        war = (
            wraa
            + baserunning_runs
            + fielding_runs
            + positional_runs
            + league_adj_runs
            + replacement_runs
        ) / ctx.runs_per_win
        return war
    except ZeroDivisionError:
        return None


def fip(HR: int, BB: int, HBP: int, K: int, IP: float, ctx: LeagueContext) -> Optional[float]:
    """FIP (Fielding Independent Pitching)를 계산합니다."""
    if IP == 0:
        return None
    
    numerator = (13 * HR) + (3 * (BB + HBP)) - (2 * K)
    return (numerator / IP) + ctx.fip_const


def war_pitcher(fip_val: float, IP: float, ctx: LeagueContext) -> Optional[float]:
    """FIP 기반의 투수 WAR (Wins Above Replacement)를 단순화된 공식으로 계산합니다."""
    if ctx.runs_per_win == 0:
        return None
        
    try:
        runs_above_replacement = (
            (ctx.lg_ra9 * ctx.park_factor - fip_val) * (IP / 9)
        )
        return runs_above_replacement / ctx.runs_per_win
    except ZeroDivisionError:
        return None


# 한국어 설명 
_METRIC_KO = {
    # 투수 지표
    "ERA": ("평균자책점", "9이닝당 자책점, 낮을수록 좋음"),
    "ERA-": ("ERA-", "리그 평균 대비 방어율, 100 미만이 좋음"),
    "FIP": ("수비무관 평균자책(FIP)", "볼넷·사구·삼진·홈런 기반, 낮을수록 좋음"),
    "FIP-": ("FIP-", "리그 평균 대비 FIP, 100 미만이 좋음"),
    "WHIP": ("이닝당 출루 허용(WHIP)", "이닝당 볼넷+피안타, 낮을수록 좋음"),
    "K/9": ("9이닝당 삼진(K/9)", "삼진 능력"),
    "BB/9": ("9이닝당 볼넷(BB/9)", "제구 안정성"),
    "K-BB%": ("K-BB%", "삼진율 - 볼넷율, 높을수록 좋음"),
    
    # 타자 지표
    "AVG": ("타율", "안타/타수, 높을수록 좋음"),
    "OBP": ("출루율", "출루/타석수, 높을수록 좋음"),
    "SLG": ("장타율", "루타/타수, 높을수록 좋음"),
    "OPS": ("OPS(출루+장타)", "출루율과 장타율의 합, 높을수록 좋음"),
    "OPS+": ("OPS+", "리그 평균 대비 OPS, 100 초과가 좋음"),
    "WRC+": ("wRC+", "구장 보정 공격지표, 100이 리그 평균"),
    "WAR": ("대체선수 대비 승리기여(WAR)", "팀 승리에 기여한 승수"),
    "ISO": ("순수 장타율(ISO)", "장타력을 나타내는 지표"),
    "BABIP": ("인플레이 타구 타율(BABIP)", "운/수비 영향이 큰 지표"),
    
    # 기본 기록 지표
    "HR": ("홈런", "홈런 개수"),
    "RBI": ("타점", "팀 득점에 기여한 타점"),
    "SB": ("도루", "도루 성공 횟수"),
    "R": ("득점", "득점 횟수"),
    "H": ("안타", "안타 개수"),
    "2B": ("2루타", "2루타 개수"),
    "3B": ("3루타", "3루타 개수"),
    "BB": ("볼넷", "볼넷 횟수"),
    "SO": ("삼진", "삼진 횟수"),
    "PA": ("타석", "총 타석수"),
    "AB": ("타수", "타수"),
    
    # 투수 기본 기록
    "W": ("승", "승리 횟수"),
    "L": ("패", "패배 횟수"),
    "SV": ("세이브", "세이브 횟수"),
    "HLD": ("홀드", "홀드 횟수"),
    "IP": ("이닝", "투구 이닝수"),
    "K": ("삼진", "삼진 탈삼진 수"),
    "HA": ("피안타", "피안타 개수"),
    "BBA": ("볼넷허용", "허용 볼넷수"),
    "CSW%": ("CSW%", "콜+헛스윙 비율, 투수의 구위를 나타내는 지표"),
}

@dataclass
class LeagueGradeBoundaries:
    """시즌별, 리그별 특성을 반영한 동적 등급 기준을 정의합니다."""
    era_ace: float = 2.50
    era_top: float = 3.50
    era_avg: float = 4.50
    era_low: float = 5.00
    whip_excellent: float = 1.10
    whip_good: float = 1.30
    whip_avg: float = 1.50
    ops_elite: float = 0.900
    ops_top: float = 0.800
    ops_avg: float = 0.700
    wrc_plus_mvp: float = 160
    wrc_plus_elite: float = 140
    wrc_plus_top: float = 120
    wrc_plus_avg: float = 100
    wrc_plus_low: float = 80
    war_mvp: float = 6.0
    war_allstar: float = 4.0
    war_starter: float = 2.0
    war_role: float = 0.0
    fip_ace: float = 3.00
    fip_top: float = 3.70
    fip_avg: float = 4.30
    k_per_9_excellent: float = 9.0
    k_per_9_good: float = 7.0
    k_per_9_avg: float = 5.0
    bb_per_9_excellent: float = 2.0
    bb_per_9_good: float = 3.0
    bb_per_9_avg: float = 4.0

def grade_metric_ko(key: str, value: float, grades: Optional[LeagueGradeBoundaries] = None) -> str:
    if value is None:
        return "데이터 부족"
    
    if grades is None:
        grades = LeagueGradeBoundaries()

    k = key.upper()
    v = value
    if k == "ERA":
        if v <= grades.era_ace: return "에이스급"
        if v <= grades.era_top: return "상위권"
        if v <= grades.era_avg: return "평균권"
        if v <= grades.era_low: return "하락세"
        return "재정비 필요"
    if k == "WHIP":
        if v <= grades.whip_excellent: return "출루 억제 우수"
        if v <= grades.whip_good: return "양호"
        if v <= grades.whip_avg: return "평균권"
        return "불안정"
    if k == "OPS":
        if v >= grades.ops_elite: return "엘리트"
        if v >= grades.ops_top: return "상위권"
        if v >= grades.ops_avg: return "보통"
        return "낮음"
    if k == "WRC+":
        if v >= grades.wrc_plus_mvp: return "MVP급"
        if v >= grades.wrc_plus_elite: return "엘리트"
        if v >= grades.wrc_plus_top: return "상위권"
        if v >= grades.wrc_plus_avg: return "평균권"
        if v >= grades.wrc_plus_low:  return "하락세"
        return "저조"
    if k == "WAR":
        if v >= grades.war_mvp: return "MVP 후보"
        if v >= grades.war_allstar: return "올스타급"
        if v >= grades.war_starter: return "선발 레귤러급"
        if v >= grades.war_role: return "롤플레이어"
        return "대체선수 이하"
    if k == "FIP":
        if v <= grades.fip_ace: return "에이스급"
        if v <= grades.fip_top: return "상위권"
        if v <= grades.fip_avg: return "평균권"
        return "재정비 필요"
    if k == "K/9":
        if v >= grades.k_per_9_excellent: return "삼진 능력 탁월"
        if v >= grades.k_per_9_good: return "양호"
        if v >= grades.k_per_9_avg: return "평균권"
        return "낮음"
    if k == "BB/9":
        if v <= grades.bb_per_9_excellent: return "제구 매우 안정"
        if v <= grades.bb_per_9_good: return "양호"
        if v <= grades.bb_per_9_avg: return "평균권"
        return "불안정"
    if k == "ERA-" or k == "FIP-":
        if v <= 75: return "엘리트"
        if v <= 85: return "상위권"
        if v <= 100: return "평균권"
        if v <= 115: return "평균 이하"
        return "재정비 필요"
    if k == "K-BB%" or k == "K-BB":
        if v >= 20: return "탁월"
        if v >= 15: return "우수"
        if v >= 10: return "양호"
        if v >= 5: return "평균권"
        return "개선 필요"
    if k == "OPS+":
        if v >= 140: return "MVP급"
        if v >= 120: return "엘리트"
        if v >= 110: return "상위권"
        if v >= 100: return "평균권"
        if v >= 85: return "평균 이하"
        return "저조"
    if k == "AVG":
        if v >= 0.320: return "최상급"
        if v >= 0.300: return "우수"
        if v >= 0.280: return "양호"
        if v >= 0.250: return "평균권"
        return "개선 필요"
    return ""

def describe_metric_ko(key: str, value: float, precision: int = 2, grades: Optional[LeagueGradeBoundaries] = None) -> str:
    name, note = _METRIC_KO.get(key.upper(), (key, ""))
    if value is None:
        return f"{name}: 데이터 부족"
    grade = grade_metric_ko(key, value, grades)
    val = f"{value:.{precision}f}" if isinstance(value, (int, float)) else str(value)
    extra = f" — {note}" if note else ""
    if grade:
        return f"{name}: {val} · {grade}{extra}"
    return f"{name}: {val}{extra}"

# 이닝 포맷
def ip_to_outs(ip: float) -> int:
    """야구 이닝(e.g., 52.1)을 아웃카운트로 변환합니다."""
    full_innings = int(ip)
    partial_outs = round((ip - full_innings) * 10)
    return (full_innings * 3) + partial_outs

def outs_to_ip_float(outs: int) -> float:
    """아웃카운트를 계산용 소수점 이닝으로 변환합니다."""
    return outs / 3.0

def format_ip(ip: float) -> str:
    """소수점 이닝을 야구 표준 표기법(e.g., 52.2)으로 변환합니다."""
    if ip is None: return "0.0"
    try:
        # Handle potential floating point inaccuracies e.g. 52.666667 -> 52.2
        outs = round(ip * 3)
        full_innings = outs // 3
        partial_outs = outs % 3
        return f"{full_innings}.{partial_outs}"
    except (TypeError, ValueError):
        return "0.0"


#  추가 타격/투구 지표 및 보조 기능
def avg(H:int, AB:int) -> Optional[float]:
    if AB <= 0:
        return None
    return H / AB

def iso(H:int, doubles:int, triples:int, HR:int, AB:int) -> Optional[float]:
    slg_val = slg(H, doubles, triples, HR, AB)
    avg_val = avg(H, AB)
    if slg_val is None or avg_val is None:
        return None
    return slg_val - avg_val

def k_per_nine(K:int, IP:float) -> Optional[float]:
    if IP <= 0:
        return None
    return 9.0 * K / IP

def bb_per_nine(BB:int, IP:float) -> Optional[float]:
    if IP <= 0:
        return None
    return 9.0 * BB / IP

def k_rate(K:int, PA:int) -> Optional[float]:
    if PA <= 0:
        return None
    return K / PA

def bb_rate(BB:int, PA:int) -> Optional[float]:
    if PA <= 0:
        return None
    return BB / PA

def k_minus_bb_pct(K:int, BB:int, PA:int) -> Optional[float]:
    Kp = k_rate(K, PA)
    BBp = bb_rate(BB, PA)
    if Kp is None or BBp is None:
        return None
    return 100.0 * (Kp - BBp)

def ops_plus(OBP:float, SLG:float, ctx: LeagueContext) -> Optional[float]:
    """
    Simplified OPS+: 100 * ( (OBP/lgOBP + SLG/lgSLG - 1) / park_factor )
    """
    if ctx.lg_OBP <= 0 or ctx.lg_SLG <= 0 or ctx.park_factor <= 0:
        return None
    return 100.0 * ((OBP / ctx.lg_OBP + SLG / ctx.lg_SLG - 1.0) / ctx.park_factor)

def era_minus(ERA:float, ctx: LeagueContext) -> Optional[float]:
    """ERA-: 100 * ERA / (lgERA * park_factor). Lower is better."""
    if ctx.lg_ERA <= 0 or ctx.park_factor <= 0:
        return None
    return 100.0 * (ERA / (ctx.lg_ERA * ctx.park_factor))

def fip_minus(FIP_value:float, ctx: LeagueContext) -> Optional[float]:
    """FIP-: 100 * FIP / (lgFIP * park_factor). Lower is better."""
    if ctx.lg_FIP <= 0 or ctx.park_factor <= 0:
        return None
    return 100.0 * (FIP_value / (ctx.lg_FIP * ctx.park_factor))

def csw_rate(called_strikes:int, swinging_strikes:int, total_pitches:int) -> Optional[float]:
    """CSW% = (Called Strikes + Whiffs) / Total Pitches."""
    if total_pitches <= 0:
        return None
    return (called_strikes + swinging_strikes) / total_pitches

def safe_pct(x: Optional[float]) -> Optional[float]:
    return None if x is None else 100.0 * x

# 일관된 형식으로 정리하고 출력
def classify_game_comment(score_a:int, score_b:int) -> str:
    diff = abs(score_a - score_b)
    total = score_a + score_b
    if diff >= 5:
        return "완승"
    if diff == 1:
        return "접전"
    if total >= 12:
        return "타격전"
    if total <= 3:
        return "투수전"
    if min(score_a, score_b) <= 2:
        return "상대 투수 공략 실패"
    return "균형 잡힌 경기"

def format_game_line(date:str, team_a:str, score_a:int, team_b:str, score_b:int, sp_a:str, sp_b:str, comment:Optional[str]=None) -> str:
    if not comment:
        comment = classify_game_comment(score_a, score_b)
    return f"{date} {team_a} {score_a}–{score_b} {team_b} — 선발: {team_a} {sp_a}, {team_b} {sp_b} · 한줄평: {comment}"

def pitcher_rank_score(era_minus_v, fip_minus_v, kbb_pct, whip, ip):
    # 낮을수록 좋은 지표(ERA-, FIP-)는 그대로, K-BB%는 높을수록 좋으니 100-KBB%로 반전
    base = 0.4*era_minus_v + 0.3*fip_minus_v + 0.2*(100 - kbb_pct) + 0.1*whip*100
    # IP 가중치 (시그모이드): IP≈60을 기준으로 점차 가중
    weight = 1 / (1 + math.exp(-(ip - 60)/20))
    return base / max(0.0001, weight)

def scope_header(year:int, covered_teams:int, role:str, min_value:int):
    if role == "SP":
        role_txt = "선발"
        qualifier = f"IP≥{min_value}"
    elif role == "RP":
        role_txt = "불펜"
        qualifier = f"IP≥{min_value}"
    elif role == "BAT":
        role_txt = "타자"
        qualifier = f"PA≥{min_value}"
    else:
        role_txt = role
        qualifier = f"MIN≥{min_value}"
    return f"범위: {year} 정규시즌 · 커버 구단 {covered_teams} · {role_txt} · 최소 {qualifier}"
