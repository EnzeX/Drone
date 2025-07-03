#!/usr/bin/env python3
import os
import pickle
import glob

DATA_DIR = os.path.expanduser("~/bc_data")
MERGED_FILE = os.path.join(DATA_DIR, "tello_merged_expert_data.pkl")

# 匹配所有以 tello_expert_data_ 开头的 pkl 文件
file_list = sorted(glob.glob(os.path.join(DATA_DIR, "tello_expert_data_*.pkl")))

all_data = []
print(f"🔍 找到 {len(file_list)} 个数据文件：")
for f in file_list:
    print(f" - {os.path.basename(f)}")
    with open(f, "rb") as pf:
        data = pickle.load(pf)
        all_data.extend(data)

# 保存合并结果
with open(MERGED_FILE, "wb") as out:
    pickle.dump(all_data, out)

print(f"✅ 已合并保存 {len(all_data)} 条数据至 {MERGED_FILE}")

