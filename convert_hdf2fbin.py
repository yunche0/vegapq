import h5py, numpy as np, os

def convert(name):
    with h5py.File(name, 'r') as f:
        base  = f['train'][:]          # (N, 128)
        query = f['test'][:]           # (Q, 128)
        gt    = f['neighbors'][:]      # (Q, K)
    os.makedirs('sift-128-euclidean', exist_ok=True)

    # PathWeaver 真格式：N（int64）→ dim（int64）→ 数据（float32）
    with open('sift-128-euclidean/base.fbin', 'wb') as fb:
        np.array(base.shape[0], dtype=np.int64).tofile(fb)   # N
        np.array(base.shape[1], dtype=np.int64).tofile(fb)   # 128
        base.astype(np.float32).tofile(fb)

    with open('sift-128-euclidean/query.fbin', 'wb') as fq:
        np.array(query.shape[0], dtype=np.int64).tofile(fq)
        np.array(query.shape[1], dtype=np.int64).tofile(fq)
        query.astype(np.float32).tofile(fq)

    gt.astype(np.int32).tofile('sift-128-euclidean/ground_truth.ivecs')
    print('convert done: base/query/gt shape =', base.shape, query.shape, gt.shape)

if __name__ == '__main__':
    import sys
    convert(sys.argv[1])