import json
import random
import os

# Constants
OUTPUT_FILE = "AI/data/stress_test_queries.json"
TOTAL_QUERIES = 1000
HOT_RATIO = 0.1  # 100 hot queries (will be requested 70% of the time)
LONG_TAIL_RATIO = 0.9 # 900 long tail queries

# KBO Data Source
TEAMS = ["KIA", "LG", "두산", "롯데", "삼성", "키움", "한화", "KT", "NC", "SSG"]
PLAYERS_HOT = ["김도영", "류현진", "양현종", "구자욱", "손아섭", "최정", "강백호", "이정후", "김광현", "원태인"]
PLAYERS_TAIL = [f"선수{i}" for i in range(100)] # Synthetic long tail players
STATS = ["타율", "홈런", "타점", "ERA", "다승", "삼진", "OPS", "wRC+", "WAR"]
YEARS = range(2015, 2026)
RULES = ["FA 자격", "비디오 판독", "2차 드래프트", "샐러리캡", "외국인 선수", "트레이드 마감", "엔트리 확대"]

def generate_hot_queries(count):
    queries = []
    for _ in range(count):
        type_ = random.choice(["player_stat", "team_stat", "rule"])
        if type_ == "player_stat":
            q = f"{random.choice(YEARS)}년 {random.choice(PLAYERS_HOT)} {random.choice(STATS)}"
        elif type_ == "team_stat":
            q = f"{random.choice(YEARS)}년 {random.choice(TEAMS)} {random.choice(STATS)}"
        else:
            q = f"{random.choice(RULES)} 설명해줘"
        queries.append(q)
    return list(set(queries)) # Dedup initially but we might need to fill up

def generate_long_tail_queries(count):
    queries = []
    for i in range(count):
        # Make them very specific to ensure uniqueness
        type_ = random.choice(["complex", "specific_date", "obscure_rule"])
        if type_ == "complex":
            q = f"{random.choice(YEARS)}년 {random.choice(TEAMS)}에서 {random.choice(STATS)} 1위는 누구야? (ID:{i})"
        elif type_ == "specific_date":
            q = f"{random.choice(YEARS)}월 {random.randint(4,10)}월 {random.randint(1,30)}일 {random.choice(TEAMS)} 경기 결과 알려줘 (ID:{i})"
        else:
            q = f"야구 규칙 조항 {random.randint(1, 100)}조 {random.randint(1,5)}항 설명 (ID:{i})"
        queries.append(q)
    return queries

def main():
    print("Generating synthetic queries...")
    
    num_hot = int(TOTAL_QUERIES * HOT_RATIO)
    num_tail = TOTAL_QUERIES - num_hot
    
    hot_queries = generate_hot_queries(num_hot)
    # Ensure we have exactly num_hot (fill if dedup reduced count)
    while len(hot_queries) < num_hot:
        hot_queries.append(f"{random.choice(PLAYERS_HOT)}의 {random.choice(STATS)} 기록은?")
        
    tail_queries = generate_long_tail_queries(num_tail)
    
    all_queries = {
        "hot": hot_queries,
        "long_tail": tail_queries,
        "all": hot_queries + tail_queries
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=2)
        
    print(f"Generated {len(all_queries['all'])} queries.")
    print(f"- Hot: {len(hot_queries)}")
    print(f"- Long-tail: {len(tail_queries)}")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
