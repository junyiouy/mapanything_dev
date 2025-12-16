#!/bin/bash

# ================= 配置区域 =================
ROOT_PATH="/wekafs/ict/junyiouy/map_anything_data/scannetppv2"
NUM_GPUS=8                       # 物理 GPU 数量
TASKS_PER_GPU=8                  # 【新增】每个 GPU 上跑几个进程？
LOG_DIR="/wekafs/ict/junyiouy/map-anything/misc_files/render_logs"

# 计算总进程数 (World Size)
WORLD_SIZE=$((NUM_GPUS * TASKS_PER_GPU))
# ===========================================

# 1. 准备工作
mkdir -p $LOG_DIR
echo "=================================================="
echo "准备启动并行渲染任务..."
echo "物理 GPU 数量: $NUM_GPUS"
echo "每卡任务数:    $TASKS_PER_GPU"
echo "总进程数 (World Size): $WORLD_SIZE"
echo "日志目录: $LOG_DIR"
echo "按 Ctrl+C 可以一次性终止所有任务"
echo "=================================================="

# 2. 定义清理函数
cleanup() {
    echo ""
    echo "🚨 检测到中断信号 (Ctrl+C)！"
    echo "正在终止所有后台 Worker 进程 (PPID=$$)..."
    pkill -P $$ 
    sleep 1
    echo "✅ 所有进程已清理完毕，退出。"
    exit 1
}

# 3. 注册信号捕获
trap cleanup SIGINT SIGTERM

# 4. 循环启动子进程
# 现在的循环是针对 "Rank" (任务ID)，而不是物理 GPU ID
for ((rank=0; rank<WORLD_SIZE; rank++)); do
    
    # === 关键修改：计算当前 Rank 应该分配到哪个 GPU ===
    # 使用取模运算，均匀将任务撒到各个 GPU 上
    # 例如：8卡，Rank 0->GPU0, Rank 1->GPU1 ... Rank 8->GPU0, Rank 9->GPU1
    gpu_id=$((rank % NUM_GPUS))
    
    # 如果你想让同一个 GPU 的任务 ID 连续 (0,1->GPU0; 2,3->GPU1)，可以用除法：
    # gpu_id=$((rank / TASKS_PER_GPU))

    # 物理隔离 GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # 这里的 & 让 python 在后台运行
    python -m wai_processing.scripts.run_rendering \
        root="$ROOT_PATH" \
        rank=$rank \
        world_size=$WORLD_SIZE \
        > "$LOG_DIR/render_rank_${rank}_gpu_${gpu_id}.log" 2>&1 &
    
    PID=$!
    echo "[启动] Rank $rank on GPU $gpu_id (PID: $PID)"
    
done

echo "=================================================="
echo "🚀 所有 $WORLD_SIZE 个任务已启动！"
echo "你可以使用 'tail -f $LOG_DIR/render_rank_0_gpu_0.log' 查看进度"
echo "保持此 tmux 窗口开启，按 Ctrl+C 停止所有任务。"
echo "=================================================="

# 5. 挂起主进程
wait