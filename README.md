# Lab

import numpy as np                             # 引入 numpy，方便處理陣列與數值運算
import itertools                               # 引入 itertools，提供排列/組合等工具

# -------------------
# 條件參數（可調）
L_REL_TOL = 0.04      # L 差距相對容差 < 4%（判斷相鄰片亮度差是否允許）
AB_TOL = 2            # a、b 相鄰差距 < 2（判斷相鄰片色度差是否允許）
AB_RANGE_LIMIT = 5    # 每套內 a、b 的最大範圍（max - min） ≤ 5
GROUP_SIZE = 8        # 每套產品包含 8 片
TOTAL_GROUPS = 5      # 目標要選出的套數（5 套）
# -------------------

# 檢查兩片是否可以相鄰（能不能排在一起）
def can_follow(p1, p2):
    L1,a1,b1 = p1                               # 把第一片的 L,a,b 解包成變數
    L2,a2,b2 = p2                               # 把第二片的 L,a,b 解包成變數
    return (abs(L1-L2)/L1 < L_REL_TOL and       # 檢查 L 的相對差是否小於 L_REL_TOL（4%）
            abs(a1-a2) < AB_TOL and             # 檢查 a 差是否小於 AB_TOL（2）
            abs(b1-b2) < AB_TOL)                # 檢查 b 差是否小於 AB_TOL（2）

# 給定一組 8 個片的索引，檢查是否存在一個合法的排列順序（左到右）
def valid_sequence(seq, data):
    """
    seq: list 或 tuple，包含 8 個樣品的索引（例如 (0,3,5,...)）
    data: 數據陣列，shape = (44,3)，每列是 [L,a,b]
    回傳：若存在合法排列，回傳該排列（list）；若不存在，回傳 None
    """
    for perm in itertools.permutations(seq):    # 列舉 seq 的所有排列（8! = 40,320 種可能）
        ok = True                               # 標記此排列是否每個相鄰都合法
        for i in range(len(perm)-1):            # 檢查排列中每一對相鄰元素（從第0對到第6對）
            if not can_follow(data[perm[i]], data[perm[i+1]]):  # 若任一相鄰不合格
                ok = False                      # 標記為不合法
                break                            # 跳出內層相鄰檢查，改試下一個排列
        if ok:                                  # 如果相鄰檢查都通過
            a_vals = [data[i][1] for i in perm] # 取排列中每片的 a 值
            b_vals = [data[i][2] for i in perm] # 取排列中每片的 b 值
            # 再檢查整套的 a,b 範圍是否在限制內（max-min <= AB_RANGE_LIMIT）
            if max(a_vals)-min(a_vals) <= AB_RANGE_LIMIT and max(b_vals)-min(b_vals) <= AB_RANGE_LIMIT:
                return list(perm)               # 找到一個合格排列時，回傳該排列（list 型式）
    return None                                 # 若所有排列都不合格，回傳 None

# 產生所有「候選套」：也就是所有能組成一套且能找到一個合法內部排列的 8 片組
def generate_candidates(data):
    n = len(data)                              # 樣本總數（期望為 44）
    candidates = []                            # 存放所有候選套的清單（每個元素是已排好的 8 個索引 list）
    for combo in itertools.combinations(range(n), GROUP_SIZE):  # 從 n 個樣品選 8 個（組合，順序不考慮）
        seq = valid_sequence(combo, data)      # 嘗試為這 8 片找一個合法的排列（若存在）
        if seq is not None:                    # 若找到了合法排列
            candidates.append(seq)             # 把該已排好的序列加入候選清單
    return candidates                           # 回傳所有候選套（每個是 list of 8 索引）

# 從候選套中挑出 5 套互不重疊（每片只能被用一次）的組合，並回傳前 max_solutions 個解
def find_5_sets(candidates, max_solutions=10):
    """
    candidates: list，每項是已排好的 8 片索引（list）
    max_solutions: 最多儲存多少個解（避免結果數爆炸）
    回傳：solutions（list of 解），每個解是由 5 個套組成，每個套是 8 個索引的 list
    """
    solutions = []                             # 儲存找到的解（每個解包含 5 套）
    used = set()                               # 追蹤已被選用的樣片索引集合，確保互不重疊

    def dfs(start, path):
        # start: 從 candidates 的哪個索引開始往後嘗試（避免重複排列，例如只考 i+1 之後的候選）
        # path: 當前已選的套的 list（內容是每套的 list of 8 索引）
        if len(path) == TOTAL_GROUPS:          # 若已經選到 TOTAL_GROUPS（5）套
            solutions.append([p[:] for p in path])  # 存一份 path 的深複本到 solutions
            return                               # 回溯（此路徑已完成一個解）
        for i in range(start, len(candidates)):  # 從 start 開始逐一試每個候選套
            cand = candidates[i]                 # 取出第 i 個候選套（已排好內部順序）
            # 檢查候選套是否與已用樣片重疊（cand 中的每個索引都不能在 used 裡）
            if all(idx not in used for idx in cand):
                path.append(cand)                # 選這個套加入 path
                used.update(cand)                # 把此套的所有索引加入 used（標記為已用）
                dfs(i+1, path)                   # 繼續遞迴，且 i+1 可避免重複排列的順序
                path.pop()                       # 回溯：把剛選的套移除
                for idx in cand:                 # 回溯：把 cand 的索引從 used 中移除
                    used.remove(idx)
            # 若已經找到足夠多的解，提早結束迴圈與遞迴
            if len(solutions) >= max_solutions:
                return

    dfs(0, [])                                  # 從候選清單索引 0 開始搜尋，初始 path 為空
    return solutions                             # 回傳找到的 solutions（最多 max_solutions 個）

#改良，用鄰近法去找
def generate_candidates(data, group_size=8):
    n = len(data)
    candidates = set()  # 用 set 避免重複

    for start in range(n):
        stack = [[start]]  # 以每個樣品為起點

        while stack:
            group = stack.pop()

            if len(group) == group_size:
                # 檢查 a、b 差值範圍
                a_vals = [data[i][1] for i in group]
                b_vals = [data[i][2] for i in group]
                if max(a_vals) - min(a_vals) <= 5 and max(b_vals) - min(b_vals) <= 5:
                    candidates.add(tuple(sorted(group)))
                continue

            last = group[-1]
            L_last, a_last, b_last = data[last]

            for nxt in range(n):
                if nxt in group:  
                    continue
                L, a, b = data[nxt]
                # 鄰近條件檢查
                if abs(L - L_last) <= 0.04 * L_last and abs(a - a_last) < 2 and abs(b - b_last) < 2:
                    new_group = group + [nxt]
                    stack.append(new_group)

    return [list(cand) for cand in candidates]

#限制窮舉數
def generate_candidates(data, group_size=8, max_candidates=1000):
    """
    根據鄰近條件生成候選套裝（每套 group_size 片）
    data: [(L,a,b), ...]
    group_size: 每套片數（預設 8）
    max_candidates: 候選套裝最多收集多少組
    """
    n = len(data)
    candidates = set()  # 用 set 避免重複

    for start in range(n):
        stack = [[start]]  # 每片樣品當起點

        while stack and len(candidates) < max_candidates:
            group = stack.pop()

            if len(group) == group_size:
                # 檢查 a、b 差值範圍
                a_vals = [data[i][1] for i in group]
                b_vals = [data[i][2] for i in group]
                if max(a_vals) - min(a_vals) <= 5 and max(b_vals) - min(b_vals) <= 5:
                    candidates.add(tuple(sorted(group)))
                continue

            last = group[-1]
            L_last, a_last, b_last = data[last]

            for nxt in range(n):
                if nxt in group:
                    continue
                L, a, b = data[nxt]
                # 鄰近條件檢查
                if abs(L - L_last) <= 0.04 * L_last and abs(a - a_last) < 2 and abs(b - b_last) < 2:
                    new_group = group + [nxt]
                    stack.append(new_group)

    return [list(cand) for cand in candidates]
