# 创建测试脚本 test_fix.py
import numpy as np

def quick_test():
    filepath = "/root/autodl-tmp/PathWeaver/_datasets/sift-128-euclidean/base.fbin"
    
    # 读取文件头
    with open(filepath, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.int32)
        n, d = header[0], header[1]
        print(f"文件头: n={n}, d={d}")
        
        # 检查文件大小
        file_size = 512000008
        data_size = file_size - 8  # 减去文件头
        element_count = data_size // 4  # 每个float32是4字节
        
        print(f"文件总大小: {file_size}")
        print(f"数据部分大小: {data_size}")
        print(f"float32元素数量: {element_count}")
        print(f"基于d={d}计算的n: {element_count // d}")
        print(f"基于n={n}计算的d: {element_count // n}")

if __name__ == "__main__":
    quick_test()