#!/bin/bash
#SBATCH --job-name=gan                         # 任务名
#SBATCH --output=logs/batch_%j.out
#SBATCH --error=logs/batch_%j.err
#SBATCH --partition=compute                       # 分区
#SBATCH --nodes=4                                 # 请求4个节点
#SBATCH --ntasks=4                                # 共4个MPI进程
#SBATCH --ntasks-per-node=1                       # 每个节点1个MPI进程
#SBATCH --cpus-per-task=16                        # 每个进程分配16核线程（OpenMP）
#SBATCH --mem=32G                                 # 每节点内存
#SBATCH --time=02:00:00                           # 最长运行时间2小时

# --- 加载MPI模块（根据集群设置调整） ---
module load mpi/latest

# --- 设置OpenMP线程数为每进程CPU核数 ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- 创建日志目录（如果不存在） ---
mkdir -p logs

# --- 打印资源信息（可选） ---
echo "Running on $SLURM_NTASKS MPI tasks across $SLURM_JOB_NUM_NODES nodes"
echo "Each MPI task uses $OMP_NUM_THREADS OpenMP threads"

# 关键两行：绕开 qib0
export I_MPI_FABRICS=shm:tcp
# （Intel MPI + Slurm）
srun --mpi=pmi2 -n $SLURM_NTASKS -u --label \
  --output=logs/out_%j_rank%t.log --error=logs/err_%j_rank%t.err \
  ./simulation gan gan_scenarios.csv
