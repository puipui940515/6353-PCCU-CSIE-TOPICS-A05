# 1. 重生資料（用 v2 + pyroomacoustics，這是重點）
python gen_dataset.py --n 200000 --seed 0   --use-v2 --out data/train.npz
python gen_dataset.py --n 20000  --seed 999 --use-v2 --out data/eval.npz

# 2. 雙 head 重訓（會自動偵測 range_labels）
python train_gpu.py --steps 5000 --tag range_v1

# 3. 看 go/no-go：tensorboard 的 eval/range_hit_rate
#    明顯 > 25% → source_range 可進 obs；接近 25% → 維持關閉，距離靠 base_to_tcp_dist