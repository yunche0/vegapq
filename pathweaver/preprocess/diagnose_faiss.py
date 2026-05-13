# 诊断脚本：diagnose_faiss.py
import faiss
import numpy as np

print(f"Faiss版本: {faiss.__version__}")

# 创建一个简单的测试
dim, M = 64, 4
pq = faiss.ProductQuantizer(dim, M, 8)

# 生成测试数据
data = np.random.rand(1000, dim).astype(np.float32)
pq.train(data)

# 检查可用属性
print("ProductQuantizer对象的属性:")
for attr in dir(pq):
    if not attr.startswith('_'):
        try:
            value = getattr(pq, attr)
            if not callable(value):  # 只检查非方法属性
                if hasattr(value, 'shape'):
                    print(f"  {attr}: {value.shape}")
                else:
                    print(f"  {attr}: {type(value)}")
        except:
            print(f"  {attr}: <无法访问>")

# 特别检查码本相关属性
print("\n特别检查码本属性:")
for attr_name in ['centroids', 'codebook', 'centroids', 'codes']:
    if hasattr(pq, attr_name):
        value = getattr(pq, attr_name)
        if hasattr(value, 'shape'):
            print(f"  {attr_name} 存在: {value.shape}")
        else:
            print(f"  {attr_name} 存在: {type(value)}")
    else:
        print(f"  {attr_name} 不存在")