#!/usr/bin/env python3
import os
import numpy as np
import faiss

ROOT = "/root/autodl-tmp/PathWeaver/_datasets"

def read_fbin_fixed(filename, dtype=np.float32):
    """修复的.fbin文件读取函数"""
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.int32)
        n, d = header[0], header[1]
        
        print(f"文件头解析: 向量数量 n={n}, 维度 d={d}")
        
        data = np.frombuffer(f.read(), dtype=dtype)
        actual_elements = len(data)
        expected_elements = n * d
        
        print(f"期望数据元素: {expected_elements}, 实际读取: {actual_elements}")
        
        if actual_elements == expected_elements:
            xb = data.reshape(n, d)
            print(f"数据重塑成功: 形状 {xb.shape}")
            return xb
        else:
            raise ValueError(f"数据不完整: 期望{expected_elements}个元素，实际{actual_elements}个")

def learn(dataset: str, dim: int, M: int = 8, Ks: int = 256):
    """学习PQ码本（修复Float32Vector长度问题）"""
    data_path = os.path.join(ROOT, dataset, "base.fbin")
    
    if not os.path.exists(data_path):
        raise RuntimeError(f"找不到 {data_path}")
    
    print(f"加载数据: {data_path}")
    
    # 读取数据
    xb = read_fbin_fixed(data_path)
    xb = xb.astype(np.float32)
    print(f"数据形状: {xb.shape}")
    
    # 验证维度
    if xb.shape[1] != dim:
        print(f"使用数据实际维度: {xb.shape[1]}")
        dim = xb.shape[1]
    
    print(f"学习PQ码本 M={M}")
    
    # 创建并训练乘积量化器
    pq = faiss.ProductQuantizer(dim, M, 8)
    
    print("开始训练PQ量化器...")
    pq.train(xb)
    print("训练完成!")
    
    print("正在提取码本...")
    print(f"centroids类型: {type(pq.centroids)}")
    
    # 修复：正确转换Float32Vector为numpy数组
    # 方法1：使用faiss.vector_to_array（推荐）
    try:
        # 这是最标准的方法
        codebook_array = faiss.vector_to_array(pq.centroids)
        print(f"使用faiss.vector_to_array转换成功")
    except AttributeError:
        # 方法2：如果faiss.vector_to_array不存在，使用np.array
        print("faiss.vector_to_array不可用，使用np.array转换")
        codebook_array = np.array(pq.centroids, dtype=np.float32)
    
    print(f"转换后码本元素总数: {codebook_array.size}")
    
    # 计算预期的子空间参数
    dsub = dim // M  # 每个子空间的维度
    ksub = 256       # 每个子空间的聚类中心数 (2^8 = 256)
    
    expected_size = M * ksub * dsub
    print(f"预期码本元素总数: {expected_size}")
    
    # 验证尺寸匹配
    if codebook_array.size == expected_size:
        # 重塑为正确的三维形状 (M, ksub, dsub)
        codebook_reshaped = codebook_array.reshape(M, ksub, dsub)
        print(f"码本重塑成功: 形状 {codebook_reshaped.shape}")
    else:
        # 如果尺寸不匹配，尝试自动计算正确的ksub
        print("尺寸不匹配，尝试自动调整...")
        actual_ksub = codebook_array.size // (M * dsub)
        if actual_ksub * M * dsub == codebook_array.size:
            print(f"自动计算ksub={actual_ksub}")
            codebook_reshaped = codebook_array.reshape(M, actual_ksub, dsub)
            print(f"调整后码本形状: {codebook_reshaped.shape}")
        else:
            raise ValueError(f"码本尺寸不匹配: 期望{expected_size}, 实际{codebook_array.size}")
    
    # 保存码本
    out_file = os.path.join(ROOT, dataset, "pq_codebook.npy")
    np.save(out_file, codebook_reshaped)
    print(f"码本已保存 -> {out_file}")
    
    # 验证码本
    codebook_loaded = np.load(out_file)
    print(f"验证: 加载的码本形状: {codebook_loaded.shape}")
    
    return codebook_reshaped

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="数据集名称")
    parser.add_argument("--dim", type=int, required=True, help="向量维度")
    parser.add_argument("--M", type=int, default=8, help="子空间数量")
    parser.add_argument("--Ks", type=int, default=256, help="每子空间聚类中心数")
    args = parser.parse_args()
    
    learn(args.dataset, args.dim, args.M, args.Ks)
