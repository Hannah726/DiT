# 加载真实数据，看看长什么样
import numpy as np

# 采样参数：只对部分数据进行统计以加快速度
SAMPLE_SIZE = 1000  # 采样病人数量，如果数据量小可以设为None使用全部数据

def calc_avg_valid_tokens(data, sample_size=SAMPLE_SIZE):
    """计算平均有效token数，使用采样以加快速度"""
    if sample_size and data.shape[0] > sample_size:
        sample_indices = np.random.choice(data.shape[0], sample_size, replace=False)
        sample_data = data[sample_indices]
        return (sample_data > 0).sum(axis=-1).mean()
    else:
        return (data > 0).sum(axis=-1).mean()

def calc_time_stats(data, sample_size=SAMPLE_SIZE):
    """计算时间统计信息，使用采样以加快速度"""
    if sample_size and data.shape[0] > sample_size:
        sample_indices = np.random.choice(data.shape[0], sample_size, replace=False)
        sample_data = data[sample_indices]
        return sample_data.min(), sample_data.max(), sample_data.mean()
    else:
        return data.min(), data.max(), data.mean()

def calc_value_stats(data, sample_size=SAMPLE_SIZE):
    """计算数值统计信息，使用采样以加快速度"""
    if sample_size and data.shape[0] > sample_size:
        # 对于min/max，使用采样
        sample_indices = np.random.choice(data.shape[0], min(sample_size, data.shape[0]), replace=False)
        sample_data = data[sample_indices]
        # 对于unique，只对采样数据的前几个事件计算（因为unique很慢）
        unique_sample = sample_data[:, :min(10, sample_data.shape[1]), :].flatten()
        return sample_data.min(), sample_data.max(), len(np.unique(unique_sample))
    else:
        # 如果数据不大，计算unique时也只对部分数据计算
        unique_sample = data[:, :min(10, data.shape[1]), :].flatten()
        return data.min(), data.max(), len(np.unique(unique_sample))

# 使用内存映射模式，按需加载数据，避免一次性加载整个文件
real_input = np.load('data/processed_12/mimiciv_hi_input.npy', mmap_mode='r')  
real_type = np.load('data/processed_12/mimiciv_hi_type.npy', mmap_mode='r')
real_dpe = np.load('data/processed_12/mimiciv_hi_dpe.npy', mmap_mode='r')
real_time = np.load('data/processed_12/mimiciv_con_time_12.npy', mmap_mode='r')

# 看第一个病人的前5个事件
print("Real data:")
print("Patient 0, Event 0:")
print(f"  Input tokens: {real_input[0, 0, :]}")
print(f"  Type tokens: {real_type[0, 0, :]}")
print(f"  DPE tokens: {real_dpe[0, 0, :]}")
print(f"  Time: {real_time[0, 0]}")

# 统计有效token数量（使用采样以加快速度）
if SAMPLE_SIZE and real_input.shape[0] > SAMPLE_SIZE:
    sample_indices = np.random.choice(real_input.shape[0], SAMPLE_SIZE, replace=False)
    sample_data = real_input[sample_indices]
    valid_mask_real = sample_data > 0
    avg_valid = valid_mask_real.sum(axis=-1).mean()
    print(f"\nAverage valid tokens per event (sampled {SAMPLE_SIZE} patients): {avg_valid:.2f}")
else:
    valid_mask_real = real_input > 0
    avg_valid = valid_mask_real.sum(axis=-1).mean()
    print(f"\nAverage valid tokens per event: {avg_valid:.2f}")

generate_input = np.load('outputs/generated_samples/con/12627401.pbs101/mimiciv_input.npy', mmap_mode='r')  
generate_type = np.load('outputs/generated_samples/con/12627401.pbs101/mimiciv_type.npy', mmap_mode='r')
generate_dpe = np.load('outputs/generated_samples/con/12627401.pbs101/mimiciv_dpe.npy', mmap_mode='r')
generate_time = np.load('outputs/generated_samples/con/12627401.pbs101/mimiciv_con_time_12.npy', mmap_mode='r')

# 看第一个条件控制病人的前5个事件
print("Generated con data:")
print("Patient 0, Event 0:")
print(f"  Input tokens: {generate_input[0, 0, :]}")
print(f"  Type tokens: {generate_type[0, 0, :]}")
print(f"  DPE tokens: {generate_dpe[0, 0, :]}")
print(f"  Time: {generate_time[0, 0]}")

# 统计有效token数量（使用采样以加快速度）
if SAMPLE_SIZE and generate_input.shape[0] > SAMPLE_SIZE:
    sample_indices = np.random.choice(generate_input.shape[0], SAMPLE_SIZE, replace=False)
    sample_data = generate_input[sample_indices]
    valid_mask_con = sample_data > 0
    avg_valid = valid_mask_con.sum(axis=-1).mean()
    print(f"\nAverage con valid tokens per event (sampled {SAMPLE_SIZE} patients): {avg_valid:.2f}")
else:
    valid_mask_con = generate_input > 0
    avg_valid = valid_mask_con.sum(axis=-1).mean()
    print(f"\nAverage con valid tokens per event: {avg_valid:.2f}")

generate_uncon_input = np.load('outputs/generated_samples/uncon/12625605.pbs101/mimiciv_input.npy', mmap_mode='r')  
generate_uncon_type = np.load('outputs/generated_samples/uncon/12625605.pbs101/mimiciv_type.npy', mmap_mode='r')
generate_uncon_dpe = np.load('outputs/generated_samples/uncon/12625605.pbs101/mimiciv_dpe.npy', mmap_mode='r')
generate_uncon_time = np.load('outputs/generated_samples/uncon/12625605.pbs101/mimiciv_con_time_12.npy', mmap_mode='r')

# 看第一个没有条件控制病人的前5个事件
print("Generated uncon data:")
print("Patient 0, Event 0:")
print(f"  Input tokens: {generate_uncon_input[0, 0, :]}")
print(f"  Type tokens: {generate_uncon_type[0, 0, :]}")
print(f"  DPE tokens: {generate_uncon_dpe[0, 0, :]}")
print(f"  Time: {generate_uncon_time[0, 0]}")

# 统计有效token数量（使用采样以加快速度）
if SAMPLE_SIZE and generate_uncon_input.shape[0] > SAMPLE_SIZE:
    sample_indices = np.random.choice(generate_uncon_input.shape[0], SAMPLE_SIZE, replace=False)
    sample_data = generate_uncon_input[sample_indices]
    valid_mask_uncon = sample_data > 0
    avg_valid = valid_mask_uncon.sum(axis=-1).mean()
    print(f"\nAverage uncon valid tokens per event (sampled {SAMPLE_SIZE} patients): {avg_valid:.2f}")
else:
    valid_mask_uncon = generate_uncon_input > 0
    avg_valid = valid_mask_uncon.sum(axis=-1).mean()
    print(f"\nAverage uncon valid tokens per event: {avg_valid:.2f}")

# ========== 数据对比分析 ==========
print("\n" + "="*60)
print("数据对比分析")
print("="*60)

# 1. 数据维度对比
print("\n1. 数据维度对比:")
print(f"  Real data shape: {real_input.shape}")
print(f"  Generated con shape: {generate_input.shape}")
print(f"  Generated uncon shape: {generate_uncon_input.shape}")

# 2. 统计信息对比（使用采样）
print("\n2. 统计信息对比:")
print(f"  Real - Average valid tokens: {calc_avg_valid_tokens(real_input):.2f}")
print(f"  Con - Average valid tokens: {calc_avg_valid_tokens(generate_input):.2f}")
print(f"  Uncon - Average valid tokens: {calc_avg_valid_tokens(generate_uncon_input):.2f}")

# 3. 时间分布对比（使用采样）
print("\n3. 时间分布对比:")
r_min, r_max, r_mean = calc_time_stats(real_time)
c_min, c_max, c_mean = calc_time_stats(generate_time)
u_min, u_max, u_mean = calc_time_stats(generate_uncon_time)
print(f"  Real time - min: {r_min:.2f}, max: {r_max:.2f}, mean: {r_mean:.2f}")
print(f"  Con time - min: {c_min:.2f}, max: {c_max:.2f}, mean: {c_mean:.2f}")
print(f"  Uncon time - min: {u_min:.2f}, max: {u_max:.2f}, mean: {u_mean:.2f}")

# 4. 数值范围对比（使用采样，unique值计算较慢）
print("\n4. 数值范围对比:")
r_min, r_max, r_unique = calc_value_stats(real_input)
c_min, c_max, c_unique = calc_value_stats(generate_input)
u_min, u_max, u_unique = calc_value_stats(generate_uncon_input)
print(f"  Real input - min: {r_min}, max: {r_max}, unique values (sampled): {r_unique}")
print(f"  Con input - min: {c_min}, max: {c_max}, unique values (sampled): {c_unique}")
print(f"  Uncon input - min: {u_min}, max: {u_max}, unique values (sampled): {u_unique}")

# 5. 前几个事件的详细对比
print("\n5. 前3个事件详细对比 (Patient 0):")
for event_idx in range(min(3, real_input.shape[1])):
    print(f"\n  Event {event_idx}:")
    print(f"    Real input[:5]: {real_input[0, event_idx, :5]}")
    print(f"    Con input[:5]:  {generate_input[0, event_idx, :5]}")
    print(f"    Uncon input[:5]: {generate_uncon_input[0, event_idx, :5]}")
    print(f"    Real time: {real_time[0, event_idx]:.2f}, Con time: {generate_time[0, event_idx]:.2f}, Uncon time: {generate_uncon_time[0, event_idx]:.2f}")

print("\n" + "="*60)
print("分析完成")
print("="*60)