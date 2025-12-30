import numpy as np

real_time = np.load('data/processed_12/mimiciv_con_time_12.npy', mmap_mode='r')

# 关键检查
print(f"总样本数: {real_time.size}")
print(f"-1的数量: {(real_time == -1).sum()}")
print(f"-1的比例: {(real_time == -1).sum() / real_time.size * 100:.2f}%")
print(f"负值数量: {(real_time < 0).sum()}")
print(f"零值数量: {(real_time == 0).sum()}")

# 查看第一列（第一个事件）的分布
print(f"\n第一个事件的时间统计:")
first_event_times = real_time[:, 0, :]
print(f"  Min: {first_event_times.min()}")
print(f"  Max: {first_event_times.max()}")
print(f"  -1的比例: {(first_event_times == -1).sum() / first_event_times.size * 100:.2f}%")

# 查看非第一个事件的分布
print(f"\n非第一个事件的时间统计:")
other_events_times = real_time[:, 1:, :]
print(f"  Min: {other_events_times.min()}")
print(f"  Max: {other_events_times.max()}")
print(f"  -1的比例: {(other_events_times == -1).sum() / other_events_times.size * 100:.2f}%")